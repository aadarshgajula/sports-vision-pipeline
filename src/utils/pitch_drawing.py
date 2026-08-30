"""Shared schematic-pitch drawing, used by both the 2D top-down video
renderer (scripts/render_topdown_view.py) and formation diagrams
(src/analytics/formation.py). Plain OpenCV, not matplotlib — fast enough to
call once per video frame, which matplotlib's per-frame figure rendering
is not.

Lives under src/utils/ rather than scripts/ so both scripts/ and src/
modules can import it without src/ reaching backward into scripts/.
"""

from __future__ import annotations

import cv2
import numpy as np

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
