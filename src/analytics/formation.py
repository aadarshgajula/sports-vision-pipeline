"""Formation recognition: classify a team's shape (e.g. "4-4-2") from its
players' tracked positions during out-of-possession (defensive) play.

Deliberately geometric, not learned — by the time this runs, the hard ML
problems (detection, team classification, pixel-to-pitch calibration) are
already solved, and a formation label is just a counting/clustering
question over real-world coordinates: how many players sit in each line
from goal to goal. No dataset pairing tracked positions with ground-truth
formation labels exists (or would be easy to build), and this kind of label
doesn't benefit from a learned model the way visual detection does.

Every heuristic here was tuned against actual output from
outputs/position_log_sample3_calibrated.csv, not just reasoned about in the
abstract — see inline comments for what that data actually looked like and
why a naive version of each step would have gotten it wrong:

- ByteTrack ID fragmentation means the raw candidate pool per team is much
  larger (50+) than the ~10-11 real players — a naive "one row per
  track_id" grouping doesn't converge to real players at all.
- Comparing a team's mean position to the pitch *midline* to infer which
  goal it defends fails outright on a short clip capturing one phase of
  play near one end — both teams' means can sit on the same side. Comparing
  the two teams' means to *each other* is required instead.
- The goalkeeper search must run on the full candidate pool *before*
  de-fragmentation (picking the most-persistently-tracked players) — a real
  goalkeeper can have very low frame presence (play stayed away from their
  box) and would otherwise be discarded before ever being considered,
  silently misidentifying a real outfield defender as the keeper instead.
- A team's true base shape only really holds when it's out of possession —
  in possession, fullbacks overlap, wingers cut inside, a striker drops
  deep, all deliberately breaking formation to create attacking options.
  Averaging those moments in with defensive ones (an earlier version of
  this module used the whole clip regardless of phase) smears a real,
  well-defined formation into something that doesn't band cleanly at all —
  confirmed directly: the same clip that reports a clean formation from
  defensive-phase frames alone reports "indeterminate" from the whole clip.
  _defensive_phase_df() restricts to frames where the other team is closer
  to the ball, falling back to the whole clip (with a caveat) if too few
  such frames exist.
- Even restricted to defensive phases, line clustering can still legitimately
  fail to find clean bands on a short or unusual clip. This is reported
  honestly (low confidence, an "indeterminate" label), not forced into a
  plausible-looking but made-up grouping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from configs.field_config import PENALTY_AREA_DEPTH, PITCH_LENGTH, PITCH_WIDTH
from src.utils.ball_track_cleaning import clean_ball_track
from src.utils.pitch_drawing import TEAM_COLORS_BGR, draw_pitch_base, m2px

MIN_TRACK_FRAMES = 5  # reject one-off noise / a stray misclassified row
GK_DISTANCE_THRESHOLD_M = 20.0  # PENALTY_AREA_DEPTH (16.5) + buffer for sweeper-keeper/noise
EXPECTED_OUTFIELD_COUNT = 10  # standard XI minus keeper
AMBIGUOUS_CUTOFF_RATIO = 0.15  # flag a near-tie at the top-K inclusion boundary
LINE_GAP_CANDIDATES_M = (4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0)
BOUNDS_MARGIN_M = 5.0  # tolerate boundary-line players + minor homography noise
DEFENSIVE_LINE_SIZES = (3, 4, 5)  # real back-lines are never 1, 2, or 6+ players
MIN_DEFENSIVE_PHASE_FRAMES = 50  # below this, the out-of-possession filter has too little to work with

# Used only as a last-resort fallback when no threshold in LINE_GAP_CANDIDATES_M
# produces a naturally gap-separated, plausible grouping. Rather than surface
# a diagnostic "bands found: 1-9"-looking string (which reads as a real, if
# bizarre, formation and was reported as exactly that confusion), force-fit
# the sorted depths into whichever of these real, commonly-used formations
# has the lowest within-line variance. Always a real formation name; accuracy
# at this point is a best-effort guess over a naturally messy distribution,
# not a confirmed grouping — always tagged confidence="low" with a caveat.
COMMON_FORMATIONS: tuple[tuple[int, ...], ...] = (
    (4, 4, 2), (4, 3, 3), (4, 2, 3, 1), (4, 1, 4, 1), (4, 5, 1), (4, 1, 3, 2), (4, 3, 1, 2),
    (3, 5, 2), (3, 4, 3), (3, 4, 1, 2),
    (5, 3, 2), (5, 4, 1), (5, 2, 3),
)


@dataclass
class FormationLine:
    index: int  # 0 = deepest/most defensive line
    role_label: str
    track_ids: list[int]
    positions: dict[int, tuple[float, float]]  # track_id -> (pitch_x, pitch_y)
    avg_depth_m: float


@dataclass
class FormationResult:
    team: int
    formation_label: str
    confidence: str  # "high" | "medium" | "low"
    lines: list[FormationLine]
    goalkeeper_track_id: int | None
    goalkeeper_position: tuple[float, float] | None
    attacking_direction: float  # +1.0 (defends x=0) or -1.0 (defends x=PITCH_LENGTH)
    own_goal_x: float
    n_candidate_tracks: int
    n_included_tracks: int
    line_gap_threshold_used_m: float | None  # None when a common-formation best-fit was used instead of a gap threshold
    caveats: list[str] = field(default_factory=list)


def _defensive_phase_df(df: pd.DataFrame, team: int) -> tuple[pd.DataFrame, str | None]:
    """Restricts df to frames where `team` is out of possession (a ball-
    distance proxy: whichever team has the closer player to the ball is
    "in possession" that frame). A team's formation is only meaningful as
    its organized defensive shape — in possession, fullbacks overlap,
    wingers cut inside, and a striker drops deep, all deliberately breaking
    the base shape to create attacking options. Averaging those moments in
    with defensive ones (the previous behavior: whole-clip median,
    regardless of phase) smears a real, well-defined formation into
    something that doesn't band cleanly at all, which is exactly the
    "doesn't work when the team is mid-play" failure this was built to fix.

    Returns (filtered_df, caveat_or_None). Falls back to the full df
    (caveat explaining why) if too few frames survive — a short clip could
    be almost entirely one team attacking, in which case there's no
    meaningful defensive-phase sample to restrict to."""
    players = df[df["class_name"] == "person"].dropna(subset=["pitch_x", "pitch_y", "bbox_height"])
    if players.empty or "team" not in players or players["team"].dropna().nunique() < 2:
        return df, None

    player_scale = players.groupby("frame")["bbox_height"].median()
    ball = clean_ball_track(df, "pitch_x", "pitch_y", player_scale).dropna(subset=["pitch_x"])
    if ball.empty:
        return df, "Could not determine ball position to isolate defensive-phase frames; used all frames instead."

    by_frame_team = players.groupby(["frame", "team"])[["pitch_x", "pitch_y"]]
    closest_dist = {}
    for (frame, frame_team), group in by_frame_team:
        if frame not in ball.index:
            continue
        bx, by = ball.at[frame, "pitch_x"], ball.at[frame, "pitch_y"]
        d = np.hypot(group["pitch_x"] - bx, group["pitch_y"] - by).min()
        closest_dist.setdefault(frame, {})[int(frame_team)] = d

    defensive_frames = {
        frame for frame, dists in closest_dist.items()
        if len(dists) == 2 and min(dists, key=dists.get) != team
    }
    filtered = df[df["frame"].isin(defensive_frames)]
    n_team_rows = len(filtered[(filtered["class_name"] == "person") & (filtered["team"] == team)])
    if len(defensive_frames) < MIN_DEFENSIVE_PHASE_FRAMES or n_team_rows < MIN_TRACK_FRAMES * EXPECTED_OUTFIELD_COUNT:
        return df, (
            f"Only {len(defensive_frames)} frames of clear out-of-possession play found for this "
            f"team — not enough to isolate a defensive shape, so this used the whole clip instead "
            f"(may be less accurate during open, dynamic play)."
        )
    return filtered, None


def _candidate_pool(df: pd.DataFrame, team: int) -> pd.DataFrame:
    """One row per track_id: median pitch position + frame count, for
    tracks with enough presence and in-bounds-ish coordinates. Median, not
    mean — robust to the occasional bad homography-extrapolated point
    within an otherwise-good track."""
    data = df[(df["class_name"] == "person") & (df["team"] == team)]
    data = data.dropna(subset=["pitch_x", "pitch_y"])
    data = data[
        data["pitch_x"].between(-BOUNDS_MARGIN_M, PITCH_LENGTH + BOUNDS_MARGIN_M)
        & data["pitch_y"].between(-BOUNDS_MARGIN_M, PITCH_WIDTH + BOUNDS_MARGIN_M)
    ]
    grouped = data.groupby("track_id").agg(
        pitch_x=("pitch_x", "median"),
        pitch_y=("pitch_y", "median"),
        n_frames=("frame", "count"),
    )
    return grouped[grouped["n_frames"] >= MIN_TRACK_FRAMES].reset_index()


def _infer_attacking_direction(df: pd.DataFrame, team: int) -> tuple[float, float]:
    """Returns (attacking_direction, own_goal_x). Compares this team's mean
    pitch_x to the OTHER team's mean pitch_x, not to the pitch midline —
    validated against real data that a per-team midline comparison breaks
    when a short clip captures one phase of play near one end (both teams'
    means can land on the same side of 52.5m). Known limitation: breaks
    across a halftime end-swap, and is a coin flip if both teams' mean
    depths are nearly identical."""
    pool = _candidate_pool(df, team)
    other_team = 1 - team if team in (0, 1) else None
    other_pool = _candidate_pool(df, other_team) if other_team is not None else pd.DataFrame()

    this_mean_x = pool["pitch_x"].mean() if not pool.empty else PITCH_LENGTH / 2
    other_mean_x = other_pool["pitch_x"].mean() if not other_pool.empty else PITCH_LENGTH / 2

    if this_mean_x <= other_mean_x:
        return 1.0, 0.0  # defends x=0, attacks toward PITCH_LENGTH
    return -1.0, PITCH_LENGTH  # defends x=PITCH_LENGTH, attacks toward 0


def _find_goalkeeper(df: pd.DataFrame, team: int, own_goal_x: float) -> tuple[int | None, tuple[float, float] | None, float]:
    """Searches the FULL candidate pool (strict in-bounds only), before any
    de-fragmentation cut — a real goalkeeper can have very low frame
    presence if play stayed away from their box, and a top-K-by-persistence
    cut done first would discard them before they're ever considered,
    silently letting a real outfield defender be mistaken for the keeper.
    Strict in-bounds (not the general BOUNDS_MARGIN_M-padded pool) so a
    marginally-out-of-bounds homography artifact just past the goal line
    can't out-compete a real in-bounds candidate for "closest to goal."""
    data = df[(df["class_name"] == "person") & (df["team"] == team)]
    data = data.dropna(subset=["pitch_x", "pitch_y"])
    data = data[data["pitch_x"].between(0, PITCH_LENGTH) & data["pitch_y"].between(0, PITCH_WIDTH)]
    if data.empty:
        return None, None, float("inf")

    grouped = data.groupby("track_id").agg(
        pitch_x=("pitch_x", "median"), pitch_y=("pitch_y", "median"), n_frames=("frame", "count"),
    )
    grouped = grouped[grouped["n_frames"] >= MIN_TRACK_FRAMES]
    if grouped.empty:
        return None, None, float("inf")

    distance = grouped["pitch_x"] if own_goal_x == 0.0 else (PITCH_LENGTH - grouped["pitch_x"])
    best_track_id = distance.idxmin()
    best_distance = distance.loc[best_track_id]
    if best_distance <= GK_DISTANCE_THRESHOLD_M:
        row = grouped.loc[best_track_id]
        return int(best_track_id), (float(row["pitch_x"]), float(row["pitch_y"])), float(best_distance)
    return None, None, float(best_distance)


def _cluster_lines(depths: np.ndarray, threshold: float) -> list[list[int]]:
    """1D gap-based clustering on sorted distance-from-own-goal values:
    start a new line whenever the gap to the next player exceeds
    `threshold`. Chosen over k-means because k-means needs the number of
    clusters up front — exactly what we're trying to discover; gap
    clustering finds however many lines the data actually supports (3 for a
    back-3, 4 for a back-4, etc.)."""
    order = np.argsort(depths)
    lines: list[list[int]] = [[int(order[0])]]
    for i in range(1, len(order)):
        gap = depths[order[i]] - depths[order[i - 1]]
        if gap > threshold:
            lines.append([])
        lines[-1].append(int(order[i]))
    return lines


def _plausible(lines: list[list[int]]) -> bool:
    n_lines = len(lines)
    if not (2 <= n_lines <= 5):
        return False
    sizes = [len(line) for line in lines]
    total = sum(sizes)
    if any(size < 1 or size > 6 for size in sizes):
        return False
    if max(sizes) / total > 0.7:
        return False
    # lines[0] is always the deepest/most-defensive group — _cluster_lines
    # walks players sorted ascending by depth-from-own-goal and appends into
    # lines[-1] sequentially, so the first line it ever creates holds the
    # shallowest depths. A real defensive line is never a lone player or a
    # pair (nor 6+): this is what rejects the "1-2-3-4"/"1-6-2-1"-style
    # labels a large leading gap used to produce, isolating one outlier
    # player closest to goal as its own "line" with nothing to stop it.
    if len(lines[0]) not in DEFENSIVE_LINE_SIZES:
        return False
    return True


def _best_fit_common_formation(depth: np.ndarray) -> tuple[list[list[int]], tuple[int, ...]] | None:
    """Force-fits the sorted depths into whichever whitelisted formation
    shape has the lowest total within-line variance (sum of squared
    deviations from each line's own mean depth) — a deterministic,
    always-real-formation fallback for when no gap threshold produced a
    naturally separated, plausible grouping. None only if no whitelist
    entry's player count matches (e.g. fewer than 10 outfield tracks were
    available; there's no meaningful "the 4-4-2 with 2 defenders" case)."""
    n = len(depth)
    order = np.argsort(depth)
    sorted_depth = depth[order]

    best_shape, best_sse, best_partition = None, float("inf"), None
    for shape in COMMON_FORMATIONS:
        if sum(shape) != n:
            continue
        partition, sse, start = [], 0.0, 0
        for size in shape:
            idx = order[start:start + size]
            group_depth = sorted_depth[start:start + size]
            sse += float(np.sum((group_depth - group_depth.mean()) ** 2))
            partition.append(idx.tolist())
            start += size
        if sse < best_sse:
            best_shape, best_sse, best_partition = shape, sse, partition
    if best_shape is None:
        return None
    return best_partition, best_shape


def detect_formation(
    df: pd.DataFrame,
    team: int,
    expected_outfield_count: int = EXPECTED_OUTFIELD_COUNT,
    gk_distance_threshold_m: float = GK_DISTANCE_THRESHOLD_M,
) -> FormationResult:
    if df["pitch_x"].notna().sum() == 0:
        raise ValueError("No pitch_x/pitch_y in this position log — calibrate first (scripts/calibrate_pitch.py)")

    caveats: list[str] = []
    attacking_direction, own_goal_x = _infer_attacking_direction(df, team)

    gk_track_id, gk_position, gk_distance = _find_goalkeeper(df, team, own_goal_x)
    if gk_track_id is None:
        caveats.append(
            f"No player found within {gk_distance_threshold_m:.0f}m of the goal line "
            f"(closest was {gk_distance:.1f}m) — goalkeeper likely not tracked/visible "
            f"during this clip (play may have stayed away from this team's defensive third)."
        )

    defensive_df, defensive_phase_caveat = _defensive_phase_df(df, team)
    if defensive_phase_caveat:
        caveats.append(defensive_phase_caveat)

    pool = _candidate_pool(defensive_df, team)
    n_candidate_tracks = len(pool)
    if gk_track_id is not None:
        pool = pool[pool["track_id"] != gk_track_id]

    if pool.empty:
        raise ValueError(f"No outfield players found for team {team} with calibrated positions")

    if n_candidate_tracks > expected_outfield_count + 2:
        caveats.append(
            f"{n_candidate_tracks} candidate track_ids found, far more than the "
            f"~{expected_outfield_count} expected outfield players — likely tracker ID "
            f"fragmentation (a player losing and regaining an ID). Kept the "
            f"{expected_outfield_count} most persistently tracked as a best-effort proxy; "
            f"this does not verify these are {expected_outfield_count} distinct real players."
        )
    elif n_candidate_tracks < expected_outfield_count:
        caveats.append(
            f"Only {n_candidate_tracks} of the expected {expected_outfield_count} outfield "
            f"tracks are available — detection/tracking gaps likely mean some players are "
            f"missing from this formation."
        )

    pool = pool.sort_values("n_frames", ascending=False)
    included = pool.head(expected_outfield_count)
    n_included_tracks = len(included)

    if len(pool) > expected_outfield_count:
        last_included = pool.iloc[expected_outfield_count - 1]
        first_excluded = pool.iloc[expected_outfield_count]
        if last_included["n_frames"] > 0:
            ratio = (last_included["n_frames"] - first_excluded["n_frames"]) / last_included["n_frames"]
            if ratio < AMBIGUOUS_CUTOFF_RATIO:
                caveats.append(
                    f"Track selection cutoff is ambiguous: the last included track "
                    f"(id {int(last_included['track_id'])}, {int(last_included['n_frames'])} frames) "
                    f"is barely more persistent than the next excluded one "
                    f"(id {int(first_excluded['track_id'])}, {int(first_excluded['n_frames'])} frames) "
                    f"— membership at this rank is uncertain."
                )

    depth = (
        included["pitch_x"].to_numpy() if own_goal_x == 0.0
        else (PITCH_LENGTH - included["pitch_x"].to_numpy())
    )
    track_ids = included["track_id"].to_numpy()
    positions = {int(tid): (float(x), float(y)) for tid, x, y in zip(track_ids, included["pitch_x"], included["pitch_y"])}

    chosen_lines, chosen_threshold, confidence = None, None, "low"
    for threshold in LINE_GAP_CANDIDATES_M:
        grouping = _cluster_lines(depth, threshold)
        if _plausible(grouping):
            chosen_lines, chosen_threshold = grouping, threshold
            # near-tie check: would the very next threshold candidate change the grouping?
            next_idx = LINE_GAP_CANDIDATES_M.index(threshold) + 1
            if next_idx < len(LINE_GAP_CANDIDATES_M):
                next_grouping = _cluster_lines(depth, LINE_GAP_CANDIDATES_M[next_idx])
                confidence = "high" if [len(l) for l in next_grouping] == [len(l) for l in grouping] else "medium"
            else:
                confidence = "high"
            break

    if chosen_lines is None:
        fallback = _best_fit_common_formation(depth)
        if fallback is not None:
            chosen_lines, chosen_shape = fallback
            chosen_threshold = None
            confidence = "low"
            formation_label = "-".join(str(s) for s in chosen_shape)
            caveats.append(
                "No gap threshold (4-12m) produced a naturally separated grouping — this "
                f"clip's positions don't band cleanly along the pitch length axis (open, "
                f"bunched-up play, or a short window without a settled defensive shape). "
                f"'{formation_label}' is a best-fit match to the closest common formation, "
                f"not a grouping that separated on its own — treat it as a low-confidence "
                f"guess, not a confirmed shape."
            )
        else:
            chosen_lines = _cluster_lines(depth, 8.0)
            chosen_threshold = 8.0
            confidence = "low"
            sizes = "-".join(str(len(l)) for l in sorted(chosen_lines, key=lambda l: depth[l].mean()))
            formation_label = f"indeterminate ({n_included_tracks} players, bands found: {sizes})"
            caveats.append(
                f"No plausible line grouping found, and no common formation has exactly "
                f"{n_included_tracks} outfield players to fit against (fewer than the expected "
                f"{expected_outfield_count} were tracked) — this can happen with a short clip "
                f"of open, bunched-up play even when the team's real formation is well-defined; "
                f"not necessarily a tracking error."
            )
    else:
        chosen_lines = sorted(chosen_lines, key=lambda l: depth[l].mean())
        formation_label = "-".join(str(len(l)) for l in chosen_lines)

    role_labels = []
    for i in range(len(chosen_lines)):
        if i == 0:
            role_labels.append("defense")
        elif i == len(chosen_lines) - 1:
            role_labels.append("attack")
        else:
            role_labels.append("midfield" if len(chosen_lines) <= 3 else f"line_{i + 1}")

    lines = [
        FormationLine(
            index=i,
            role_label=role_labels[i],
            track_ids=[int(track_ids[j]) for j in idxs],
            positions={int(track_ids[j]): positions[int(track_ids[j])] for j in idxs},
            avg_depth_m=float(depth[idxs].mean()),
        )
        for i, idxs in enumerate(chosen_lines)
    ]

    return FormationResult(
        team=team,
        formation_label=formation_label,
        confidence=confidence,
        lines=lines,
        goalkeeper_track_id=gk_track_id,
        goalkeeper_position=gk_position,
        attacking_direction=attacking_direction,
        own_goal_x=own_goal_x,
        n_candidate_tracks=n_candidate_tracks,
        n_included_tracks=n_included_tracks,
        line_gap_threshold_used_m=chosen_threshold,
        caveats=caveats,
    )


def draw_formation(result: FormationResult, out_path: str, title: str | None = None) -> None:
    pitch_canvas = draw_pitch_base()
    color = TEAM_COLORS_BGR.get(result.team, (160, 160, 160))
    canvas = pitch_canvas  # caveats get appended as a footer panel below, once wrapped (see bottom of function)

    for line in result.lines:
        for track_id, (x, y) in line.positions.items():
            pt = m2px(x, y)
            cv2.circle(canvas, pt, 8, color, -1)
            cv2.circle(canvas, pt, 8, (255, 255, 255), 1)
            cv2.putText(canvas, str(track_id), (pt[0] + 10, pt[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if line.positions:
            cx = sum(p[0] for p in line.positions.values()) / len(line.positions)
            cy = sum(p[1] for p in line.positions.values()) / len(line.positions)
            label_pt = m2px(cx, cy - 6)
            cv2.putText(canvas, f"L{line.index + 1}", label_pt,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    if result.goalkeeper_track_id is not None and result.goalkeeper_position is not None:
        pt = m2px(*result.goalkeeper_position)
        size = 9
        cv2.rectangle(canvas, (pt[0] - size, pt[1] - size), (pt[0] + size, pt[1] + size), color, -1)
        cv2.rectangle(canvas, (pt[0] - size, pt[1] - size), (pt[0] + size, pt[1] + size), (255, 255, 255), 1)
        cv2.putText(canvas, "GK?", (pt[0] + 12, pt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    else:
        goal_pt = m2px(result.own_goal_x, PITCH_WIDTH / 2)
        cv2.putText(canvas, "no goalkeeper detected", (max(goal_pt[0] - 60, 5), goal_pt[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    header = title or f"Team {result.team}: {result.formation_label} (confidence: {result.confidence})"
    cv2.putText(canvas, header, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Caveats are still returned on FormationResult (and printed to the
    # console by generate_analytics.py) — just not drawn on the image itself.
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, canvas)
