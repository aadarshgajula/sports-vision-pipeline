#!/usr/bin/env python
"""Generate heatmaps, a passing network, and a tactical-metrics summary from
a position log produced by src/pipeline.py.

Heatmaps and tactical metrics need pitch_x/pitch_y (a homography loaded
during the pipeline run — see scripts/calibrate_pitch.py); without one, this
script skips them and says so, rather than producing a chart in meaningless
units. The passing network falls back to pixel coordinates when no
homography was used (see src/analytics/passing_network.py), so it always
runs, but `possession_radius` means pixels, not meters, in that case.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analytics import heatmaps, passing_network, tactical_metrics
from src.analytics.passing_network import DEFAULT_PIXEL_RADIUS_SCALE
from src.utils.position_log import load_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--position-log", default="outputs/position_log.csv")
    parser.add_argument("--heatmaps-dir", default="outputs/heatmaps")
    parser.add_argument("--passing-graphs-dir", default="outputs/passing_graphs")
    parser.add_argument("--possession-radius", type=float, default=3.0,
                         help="Possession radius in meters — only used if calibrated (pitch coordinates present)")
    parser.add_argument("--radius-scale", type=float, default=DEFAULT_PIXEL_RADIUS_SCALE,
                         help="Uncalibrated only: possession radius = this * median player bbox_height per frame")
    parser.add_argument("--min-possession-frames", type=int, default=3)
    args = parser.parse_args()

    df = load_csv(args.position_log)
    has_pitch_coords = df["pitch_x"].notna().any()

    if has_pitch_coords:
        for team in sorted(df["team"].dropna().unique()):
            team = int(team)
            out_path = f"{args.heatmaps_dir}/team_{team}_heatmap.png"
            try:
                heatmaps.generate_heatmap(df, out_path, team=team, title=f"Team {team} heatmap")
                print(f"Wrote {out_path}")
            except ValueError as e:
                print(f"Skipped team {team} heatmap: {e}")

        compactness = tactical_metrics.team_compactness(df)
        width_depth = tactical_metrics.team_width_depth(df)
        avg_dist = tactical_metrics.average_teammate_distance(df)
        print("\nTactical metrics (mean over all frames):")
        if not compactness.empty:
            print(compactness.groupby("team")["area_m2"].mean().rename("avg_hull_area_m2"))
        if not width_depth.empty:
            print(width_depth.groupby("team")[["width_m", "depth_m"]].mean())
        if not avg_dist.empty:
            print(avg_dist.groupby("team")["avg_distance_m"].mean())
    else:
        print("No pitch coordinates in position log (no homography was loaded during the "
              "pipeline run) — skipping heatmaps and tactical metrics. "
              "Run scripts/calibrate_pitch.py on a wide/static shot to enable them.")

    graph = passing_network.build_passing_network(
        df, possession_radius=args.possession_radius, min_possession_frames=args.min_possession_frames,
        radius_scale=args.radius_scale,
    )
    out_path = f"{args.passing_graphs_dir}/passing_network.png"
    try:
        passing_network.draw_passing_network(graph, out_path, title="Passing network")
        print(f"Wrote {out_path} ({graph.number_of_nodes()} players, {graph.number_of_edges()} passing links)")
    except ValueError as e:
        print(f"Skipped passing network: {e}")


if __name__ == "__main__":
    main()
