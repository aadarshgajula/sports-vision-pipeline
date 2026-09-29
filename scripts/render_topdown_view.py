#!/usr/bin/env python
"""Render an animated 2D top-down view: a schematic pitch with player dots
(colored by team) and the ball, moving frame-by-frame in sync with the real
match — the actual "top-down 2D player coordinates" visualization, as
opposed to generate_analytics.py's static heatmaps/passing-network summary.

Needs pitch_x/pitch_y in the position log (a homography must have been
loaded during the pipeline run — see scripts/calibrate_pitch.py). Frames
with no calibrated players/ball just render an empty pitch, so the output
stays in sync with the source video's timing.

Drawn with plain OpenCV rather than matplotlib — matplotlib's per-frame
figure rendering is too slow for a full video (hundreds to thousands of
frames); direct pixel drawing keeps this fast.
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.ball_track_cleaning import clean_ball_track
from src.utils.pitch_drawing import (
    BALL_COLOR,
    BALL_PREDICTED_COLOR,
    TEAM_COLORS_BGR,
    UNASSIGNED_COLOR,
    draw_pitch_base,
    m2px,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--position-log", required=True)
    parser.add_argument("--out", default="outputs/annotated_videos/topdown_view.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--dot-radius", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.position_log)
    if df["pitch_x"].notna().sum() == 0:
        raise SystemExit(
            "No pitch_x/pitch_y in this position log — the pipeline run that produced it "
            "didn't have a homography loaded. Calibrate first (scripts/calibrate_pitch.py), "
            "point configs/pipeline_config.yaml -> calibration.homography_path at the result, "
            "and re-run the pipeline before this script."
        )

    base = draw_pitch_base()
    h, w = base.shape[:2]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"avc1"), args.fps, (w, h))  # H.264, browser-playable

    total_frames = int(df["frame"].max()) + 1
    by_frame = {frame: g for frame, g in df.groupby("frame")}

    # The position log already carries the online Kalman tracker's estimate
    # for the ball (including short coasted gaps, flagged is_predicted), but
    # it has NO row at all for gaps beyond that tracker's patience — which
    # would otherwise render as the ball hard-cutting out and teleporting on
    # reappearance. clean_ball_track() is a second, offline pass over the
    # complete log that can interpolate straight to wherever the ball is
    # next actually observed, bridging those remaining gaps smoothly instead
    # (still leaving genuinely long gaps alone — seeing the full log doesn't
    # mean guessing across a real loss is honest).
    player_scale = (
        df[df["class_name"] == "person"].groupby("frame")["bbox_height"].median()
    )
    ball_track = clean_ball_track(df, "pitch_x", "pitch_y", player_scale)
    raw_is_predicted = (
        df[df["class_name"] == "sports ball"].set_index("frame")["is_predicted"]
    )

    for frame_idx in range(total_frames):
        canvas = base.copy()
        group = by_frame.get(frame_idx)
        if frame_idx in ball_track.index and pd.notna(ball_track.at[frame_idx, "pitch_x"]):
            pt = m2px(ball_track.at[frame_idx, "pitch_x"], ball_track.at[frame_idx, "pitch_y"])
            # Anything not a direct raw detection this frame — online
            # Kalman-coasted, or bridged by the offline pass above — gets
            # the same "inferred, not observed" hollow marker.
            is_inferred = bool(ball_track.at[frame_idx, "was_interpolated"]) or bool(
                raw_is_predicted.get(frame_idx, False)
            )
            color = BALL_PREDICTED_COLOR if is_inferred else BALL_COLOR
            cv2.circle(canvas, pt, 5, color, 2 if is_inferred else -1)
        if group is not None:
            for row in group.itertuples():
                if pd.isna(row.pitch_x) or pd.isna(row.pitch_y):
                    continue
                pt = m2px(row.pitch_x, row.pitch_y)
                if row.class_name == "person":
                    team = row.team if not pd.isna(row.team) else None
                    color = TEAM_COLORS_BGR.get(team, UNASSIGNED_COLOR)
                    cv2.circle(canvas, pt, args.dot_radius, color, -1)
                    cv2.circle(canvas, pt, args.dot_radius, (255, 255, 255), 1)
                    cv2.putText(canvas, str(int(row.track_id)), (pt[0] + 10, pt[1] + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                # other classes (e.g. "ref") intentionally not drawn as team-colored dots

        writer.write(canvas)
        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}/{total_frames}")

    writer.release()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
