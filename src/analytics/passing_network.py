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

from src.utils.ball_track_cleaning import (
    DEFAULT_MAX_BALL_GAP_FRAMES,
    DEFAULT_MAX_BALL_JUMP_SCALE,
    clean_ball_track,
)

DEFAULT_PIXEL_RADIUS_SCALE = 0.6  # possession radius = this * median player bbox_height


def _coords(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    if df["pitch_x"].notna().any():
        return df, "pitch_x", "pitch_y"
    return df, "pixel_x", "pixel_y"


def infer_possessor_per_frame(
    df: pd.DataFrame,
    possession_radius: float = 3.0,
    radius_scale: float = DEFAULT_PIXEL_RADIUS_SCALE,
    max_ball_gap_frames: int = DEFAULT_MAX_BALL_GAP_FRAMES,
    max_ball_jump_scale: float = DEFAULT_MAX_BALL_JUMP_SCALE,
) -> pd.DataFrame:
    df, xcol, ycol = _coords(df)
    is_pitch_space = xcol == "pitch_x"
    # Restrict to real players, not just "not the ball": class_name != "sports
    # ball" also lets the referee through as a possession candidate whenever
    # they're closest to the ball, which they often are (they run near play
    # by definition) — a referee doesn't "pass" the ball to anyone, so this
    # was corrupting real passing sequences into false turnovers/phantom runs.
    players = df[df["class_name"] == "person"].dropna(subset=[xcol, ycol])

    player_scale = players.groupby("frame")["bbox_height"].median() if "bbox_height" in players else pd.Series(dtype=float)
    ball = clean_ball_track(df, xcol, ycol, player_scale, max_ball_gap_frames, max_ball_jump_scale)
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
    avg_pos = df[df["class_name"] == "person"].groupby("track_id")[[xcol, ycol]].mean()

    # .iterrows() upcasts every value in a row to a common dtype, so track_id
    # (int64) would silently become float64 next to the float64 team column —
    # cast back to int explicitly rather than storing float-keyed nodes.
    graph = nx.Graph()
    graph.graph["is_pitch_space"] = xcol == "pitch_x"
    for _, row in runs.iterrows():
        track_id = int(row["track_id"])
        if track_id not in graph:
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

    if graph.graph.get("is_pitch_space", False):
        # Draw on the same schematic pitch used everywhere else in this
        # project (formation diagrams, the top-down view) instead of a bare
        # floating graph with no spatial reference — node positions are real
        # pitch meters, meaningless without the pitch itself for context.
        # Converting to the pitch canvas's pixel space (via m2px) also fixes
        # a second problem: pitch_y increases downward (image convention)
        # but matplotlib's y-axis increases upward by default, so plotting
        # raw meters directly would render the whole network upside-down
        # relative to the real pitch — imshow displays array rows top-to-
        # bottom, which already matches pixel-space "y increases downward."
        from src.utils.pitch_drawing import draw_pitch_base, m2px

        pitch_img = draw_pitch_base()
        ax.imshow(pitch_img[:, :, ::-1])  # BGR -> RGB
        pos = {n: m2px(x, y) for n, (x, y) in pos.items()}
        ax.set_xlim(0, pitch_img.shape[1])
        ax.set_ylim(pitch_img.shape[0], 0)
        ax.axis("off")

    nx.draw(
        graph, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
        width=edge_widths, edge_color="#cccccc", with_labels=False,
    )
    if title:
        ax.set_title(title)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
