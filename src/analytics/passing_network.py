"""Passing network inferred from ball-possession changes.

Heuristic, not ground truth: each frame, whichever player is closest to the
ball (within a possession radius) is the "possessor." A pass is logged when
possession moves from one player to a different player on the SAME team and
holds for at least `min_possession_frames` before and after the switch (this
filters out single-frame jitter when two players are shoulder-to-shoulder).
A switch to a different team is a turnover, not a pass, and isn't counted.

Pixel-space possession radius scales with the camera's apparent zoom level
(`radius_scale * median player bbox_height in that frame`), rather than a
fixed pixel count — a broadcast feed constantly cuts between tight close-ups
(a player fills ~400px of frame height) and wide shots (~100px), and a fixed
threshold is badly wrong at one end or the other. Pitch-space (calibrated)
distances are already zoom-invariant real-world meters, so they keep a fixed
`possession_radius`.

The ball track also gets cleaned before any of this: short gaps (the COCO
fallback ball detector, not fine-tuned for football, drops out constantly
during fast play) are linearly interpolated, and single-frame jumps too fast
to be the real ball (a false positive latching onto a crowd/ad-board object)
are rejected rather than treated as a real ball position.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

DEFAULT_PIXEL_RADIUS_SCALE = 0.6  # possession radius = this * median player bbox_height
DEFAULT_MAX_BALL_GAP_FRAMES = 10  # interpolate ball position across gaps up to this long
DEFAULT_MAX_BALL_JUMP_SCALE = 5.0  # reject a ball detection that jumps > this * local player scale


def _coords(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    if df["pitch_x"].notna().any():
        return df, "pitch_x", "pitch_y"
    return df, "pixel_x", "pixel_y"


def _clean_ball_track(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    player_scale: pd.Series,
    max_gap_frames: int,
    max_jump_scale: float,
) -> pd.DataFrame:
    """Returns a frame-indexed ball position series with outlier jumps
    dropped and short gaps linearly interpolated."""
    ball = df[df["class_name"] == "sports ball"][["frame", xcol, ycol]].dropna()
    if ball.empty:
        return ball.set_index("frame")
    ball = ball.sort_values("frame").set_index("frame")
    global_scale = player_scale.median() if not player_scale.empty else 1.0

    cleaned_rows = []
    last_good = None
    last_good_frame = None
    for frame, row in ball.iterrows():
        point = np.array([row[xcol], row[ycol]])
        if last_good is not None:
            scale = player_scale.get(frame, global_scale) or global_scale
            dist = np.hypot(*(point - last_good))
            frame_gap = max(frame - last_good_frame, 1)
            if dist > max_jump_scale * scale * frame_gap:
                continue  # outlier: too fast to be the real ball, drop this detection
        cleaned_rows.append((frame, point[0], point[1]))
        last_good, last_good_frame = point, frame

    if not cleaned_rows:
        return pd.DataFrame(columns=[xcol, ycol])

    cleaned = pd.DataFrame(cleaned_rows, columns=["frame", xcol, ycol]).set_index("frame")
    full_index = pd.RangeIndex(cleaned.index.min(), cleaned.index.max() + 1, name="frame")
    return cleaned.reindex(full_index).interpolate(method="linear", limit=max_gap_frames, limit_area="inside")


def infer_possessor_per_frame(
    df: pd.DataFrame,
    possession_radius: float = 3.0,
    radius_scale: float = DEFAULT_PIXEL_RADIUS_SCALE,
    max_ball_gap_frames: int = DEFAULT_MAX_BALL_GAP_FRAMES,
    max_ball_jump_scale: float = DEFAULT_MAX_BALL_JUMP_SCALE,
) -> pd.DataFrame:
    df, xcol, ycol = _coords(df)
    is_pitch_space = xcol == "pitch_x"
    players = df[df["class_name"] != "sports ball"].dropna(subset=[xcol, ycol])

    player_scale = players.groupby("frame")["bbox_height"].median() if "bbox_height" in players else pd.Series(dtype=float)
    ball = _clean_ball_track(df, xcol, ycol, player_scale, max_ball_gap_frames, max_ball_jump_scale)
    ball = ball.rename(columns={xcol: "ball_x", ycol: "ball_y"}).reset_index()

    merged = players.merge(ball, on="frame", how="inner")
    merged["dist"] = np.hypot(merged[xcol] - merged["ball_x"], merged[ycol] - merged["ball_y"])

    if is_pitch_space:
        radius = pd.Series(possession_radius, index=merged["frame"].unique())
    else:
        global_scale = player_scale.median() if not player_scale.empty else 1.0
        radius = (player_scale * radius_scale).reindex(merged["frame"].unique()).fillna(global_scale * radius_scale)
    merged = merged[merged["dist"] <= merged["frame"].map(radius)]

    if merged.empty:
        return pd.DataFrame(columns=["frame", "track_id", "team"])
    idx = merged.groupby("frame")["dist"].idxmin()
    return merged.loc[idx, ["frame", "track_id", "team"]].sort_values("frame").reset_index(drop=True)


def compute_possession_runs(
    df: pd.DataFrame,
    possession_radius: float = 3.0,
    min_possession_frames: int = 3,
    **possessor_kwargs,
) -> pd.DataFrame:
    """Collapse consecutive frames held by the same player into single
    "touches" (a run), dropping runs shorter than `min_possession_frames`
    (jitter between two players standing shoulder-to-shoulder). Shared by
    build_passing_network (match-level summary) and compute_pass_events
    (per-moment events for a video overlay)."""
    possession = infer_possessor_per_frame(df, possession_radius, **possessor_kwargs)
    possession["run_id"] = (possession["track_id"] != possession["track_id"].shift()).cumsum()
    runs = possession.groupby("run_id").agg(
        track_id=("track_id", "first"),
        team=("team", "first"),
        length=("frame", "count"),
        start_frame=("frame", "first"),
        end_frame=("frame", "last"),
    )
    return runs[runs["length"] >= min_possession_frames].reset_index(drop=True)


def compute_pass_events(
    df: pd.DataFrame,
    possession_radius: float = 3.0,
    min_possession_frames: int = 3,
    **possessor_kwargs,
) -> pd.DataFrame:
    """One row per detected pass: the moment possession switched from one
    player to a different SAME-TEAM player. `event_frame` is when the new
    receiver's run starts (turnovers — a switch to the other team — are
    excluded, same rule as build_passing_network). from_x/y and to_x/y are
    always pixel coordinates (for drawing on the actual video frame), even
    if possession matching itself used pitch meters."""
    runs = compute_possession_runs(df, possession_radius, min_possession_frames, **possessor_kwargs)
    pos_lookup = df.set_index(["track_id", "frame"])[["pixel_x", "pixel_y"]]

    events = []
    for (_, prev), (_, curr) in zip(runs.iterrows(), runs.iloc[1:].iterrows()):
        if prev["team"] != curr["team"]:
            continue  # turnover, not a pass
        from_id, to_id = int(prev["track_id"]), int(curr["track_id"])
        if from_id == to_id:
            continue
        try:
            from_pos = pos_lookup.loc[(from_id, int(prev["end_frame"]))]
            to_pos = pos_lookup.loc[(to_id, int(curr["start_frame"]))]
        except KeyError:
            continue  # shouldn't happen, but don't let a lookup gap crash the whole pass
        events.append({
            "event_frame": int(curr["start_frame"]),
            "from_track": from_id,
            "to_track": to_id,
            "team": curr["team"],
            "from_x": from_pos["pixel_x"], "from_y": from_pos["pixel_y"],
            "to_x": to_pos["pixel_x"], "to_y": to_pos["pixel_y"],
        })
    return pd.DataFrame(events, columns=["event_frame", "from_track", "to_track", "team",
                                          "from_x", "from_y", "to_x", "to_y"])


def build_passing_network(
    df: pd.DataFrame,
    possession_radius: float = 3.0,
    min_possession_frames: int = 3,
    **possessor_kwargs,
) -> nx.Graph:
    runs = compute_possession_runs(df, possession_radius, min_possession_frames, **possessor_kwargs)

    df, xcol, ycol = _coords(df)
    avg_pos = df[df["class_name"] != "sports ball"].groupby("track_id")[[xcol, ycol]].mean()

    # .iterrows() upcasts every value in a row to a common dtype, so track_id
    # (int64) would silently become float64 next to the float64 team column —
    # cast back to int explicitly rather than storing float-keyed nodes.
    graph = nx.Graph()
    for _, row in runs.iterrows():
        track_id = int(row["track_id"])
        pos = avg_pos.loc[track_id] if track_id in avg_pos.index else (0.0, 0.0)
        graph.add_node(track_id, team=row["team"], pos=(pos[xcol], pos[ycol]), touches=0)
        graph.nodes[track_id]["touches"] += 1

    for (_, prev), (_, curr) in zip(runs.iterrows(), runs.iloc[1:].iterrows()):
        if prev["team"] != curr["team"]:
            continue  # turnover, not a pass
        a, b = int(prev["track_id"]), int(curr["track_id"])
        if a == b:
            continue
        if graph.has_edge(a, b):
            graph[a][b]["weight"] += 1
        else:
            graph.add_edge(a, b, weight=1)

    return graph


def draw_passing_network(graph: nx.Graph, out_path: str, title: str | None = None):
    import matplotlib.pyplot as plt

    if graph.number_of_nodes() == 0:
        raise ValueError("Passing network has no nodes — nothing to draw")

    pos = nx.get_node_attributes(graph, "pos")
    touches = nx.get_node_attributes(graph, "touches")
    teams = nx.get_node_attributes(graph, "team")

    team_colors = {0: "#e63946", 1: "#457b9d"}
    node_colors = [team_colors.get(teams.get(n), "#999999") for n in graph.nodes]
    node_sizes = [200 + 100 * touches.get(n, 1) for n in graph.nodes]
    edge_widths = [1 + graph[u][v]["weight"] for u, v in graph.edges]

    fig, ax = plt.subplots(figsize=(10, 7))
    nx.draw(
        graph, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
        width=edge_widths, edge_color="#cccccc", with_labels=True, font_size=8,
        font_color="white",
    )
    if title:
        ax.set_title(title)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
