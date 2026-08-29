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
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.field_config import CENTER_CIRCLE_RADIUS, KEYPOINTS, PITCH_LENGTH, PITCH_WIDTH

PIXELS_PER_METER = 10
MARGIN_M = 4
PITCH_COLOR = (50, 130, 40)      # BGR dark green
LINE_COLOR = (230, 230, 230)
TEAM_COLORS_BGR = {0: (70, 57, 230), 1: (157, 123, 69)}  # matches src/pipeline.py's palette
UNASSIGNED_COLOR = (160, 160, 160)
BALL_COLOR = (0, 255, 255)
BALL_PREDICTED_COLOR = (0, 200, 255)


def m2px(x_m: float, y_m: float) -> tuple[int, int]:
    return int((x_m + MARGIN_M) * PIXELS_PER_METER), int((y_m + MARGIN_M) * PIXELS_PER_METER)


def draw_pitch_base() -> np.ndarray:
    w = int((PITCH_LENGTH + 2 * MARGIN_M) * PIXELS_PER_METER)
    h = int((PITCH_WIDTH + 2 * MARGIN_M) * PIXELS_PER_METER)
    canvas = np.full((h, w, 3), PITCH_COLOR, dtype=np.uint8)

    def line(p1_m, p2_m):
        cv2.line(canvas, m2px(*p1_m), m2px(*p2_m), LINE_COLOR, 2)

    # Outer boundary
    line((0, 0), (PITCH_LENGTH, 0))
    line((PITCH_LENGTH, 0), (PITCH_LENGTH, PITCH_WIDTH))
    line((PITCH_LENGTH, PITCH_WIDTH), (0, PITCH_WIDTH))
    line((0, PITCH_WIDTH), (0, 0))

    # Halfway line
    line((PITCH_LENGTH / 2, 0), (PITCH_LENGTH / 2, PITCH_WIDTH))

    # Center circle
    cv2.circle(canvas, m2px(PITCH_LENGTH / 2, PITCH_WIDTH / 2),
               int(CENTER_CIRCLE_RADIUS * PIXELS_PER_METER), LINE_COLOR, 2)
    cv2.circle(canvas, m2px(PITCH_LENGTH / 2, PITCH_WIDTH / 2), 3, LINE_COLOR, -1)

    kp = {k.name: (k.x, k.y) for k in KEYPOINTS}
    for prefix in ("left", "right"):
        for box in ("penalty_area", "six_yard"):
            tl = kp[f"{prefix}_{box}_top_left"]
            tr = kp[f"{prefix}_{box}_top_right"]
            bl = kp[f"{prefix}_{box}_bottom_left"]
            br = kp[f"{prefix}_{box}_bottom_right"]
            line(tl, tr); line(tr, br); line(br, bl); line(bl, tl)
        spot = kp[f"{prefix}_penalty_spot"]
        cv2.circle(canvas, m2px(*spot), 3, LINE_COLOR, -1)

    return canvas


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

    for frame_idx in range(total_frames):
        canvas = base.copy()
        group = by_frame.get(frame_idx)
        if group is not None:
            for row in group.itertuples():
                if pd.isna(row.pitch_x) or pd.isna(row.pitch_y):
                    continue
                pt = m2px(row.pitch_x, row.pitch_y)
                if row.class_name == "sports ball":
                    is_pred = getattr(row, "is_predicted", False)
                    color = BALL_PREDICTED_COLOR if is_pred else BALL_COLOR
                    cv2.circle(canvas, pt, 5, color, -1 if not is_pred else 2)
                elif row.class_name == "person":
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
