"""Frame-by-frame team shape metrics: centroid, compactness, width/depth.

Needs pitch_x/pitch_y (a homography must be loaded) — these numbers are only
meaningful in real-world meters, not raw pixels, since pixel-space distances
change with camera zoom.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


def _team_frame_positions(df: pd.DataFrame) -> pd.DataFrame:
    data = df.dropna(subset=["pitch_x", "pitch_y"])
    return data[data["class_name"] != "sports ball"]


def team_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """Mean (x, y) position per team per frame."""
    data = _team_frame_positions(df)
    return data.groupby(["frame", "team"])[["pitch_x", "pitch_y"]].mean().reset_index()


def team_compactness(df: pd.DataFrame) -> pd.DataFrame:
    """Convex hull area (m^2) spanned by each team per frame — a pressing/
    compactness proxy: smaller area means a tighter defensive/attacking shape.
    Frames with fewer than 3 players for a team are skipped (no hull)."""
    data = _team_frame_positions(df)
    rows = []
    for (frame, team), group in data.groupby(["frame", "team"]):
        points = group[["pitch_x", "pitch_y"]].to_numpy()
        if len(points) < 3:
            continue
        area = ConvexHull(points).volume  # 2D hull: .volume is the enclosed area
        rows.append({"frame": frame, "team": team, "area_m2": area})
    return pd.DataFrame(rows, columns=["frame", "team", "area_m2"])


def team_width_depth(df: pd.DataFrame) -> pd.DataFrame:
    """Team shape width (pitch_y spread) and depth (pitch_x spread) per frame."""
    data = _team_frame_positions(df)
    grouped = data.groupby(["frame", "team"])
    result = grouped.agg(
        width_m=("pitch_y", lambda s: s.max() - s.min()),
        depth_m=("pitch_x", lambda s: s.max() - s.min()),
    ).reset_index()
    return result


def average_teammate_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Mean pairwise distance between teammates per frame — another
    compactness proxy, more robust than hull area with few players visible."""
    data = _team_frame_positions(df)
    rows = []
    for (frame, team), group in data.groupby(["frame", "team"]):
        points = group[["pitch_x", "pitch_y"]].to_numpy()
        if len(points) < 2:
            continue
        diffs = points[:, None, :] - points[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        n = len(points)
        mean_dist = dists[np.triu_indices(n, k=1)].mean()
        rows.append({"frame": frame, "team": team, "avg_distance_m": mean_dist})
    return pd.DataFrame(rows, columns=["frame", "team", "avg_distance_m"])
