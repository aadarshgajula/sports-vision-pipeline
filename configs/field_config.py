"""Standard FIFA/IFAB full-size pitch dimensions and landmark coordinates.

Coordinates are in meters, origin at the top-left corner flag, x along the
pitch length (0-105), y along the pitch width (0-68). This layout drives the
homography step: match these real-world points against pixel coordinates of
the same landmarks detected in a broadcast frame.

Dimensions per IFAB Laws of the Game (recommended full-size pitch):
  - Pitch: 105m x 68m
  - Penalty area: 16.5m deep, 40.32m wide
  - Six-yard box: 5.5m deep, 18.32m wide
  - Penalty spot: 11m from goal line
  - Center circle radius: 9.15m

If your footage is a smaller pitch (common outside top-flight stadiums),
override PITCH_LENGTH / PITCH_WIDTH and re-derive the dependent values below.
"""

from dataclasses import dataclass

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_WIDTH = 40.32
SIX_YARD_DEPTH = 5.5
SIX_YARD_WIDTH = 18.32
PENALTY_SPOT_DISTANCE = 11.0
CENTER_CIRCLE_RADIUS = 9.15

_pa_y0 = (PITCH_WIDTH - PENALTY_AREA_WIDTH) / 2
_pa_y1 = (PITCH_WIDTH + PENALTY_AREA_WIDTH) / 2
_sy_y0 = (PITCH_WIDTH - SIX_YARD_WIDTH) / 2
_sy_y1 = (PITCH_WIDTH + SIX_YARD_WIDTH) / 2
_mid_y = PITCH_WIDTH / 2
_mid_x = PITCH_LENGTH / 2


@dataclass(frozen=True)
class Keypoint:
    id: int
    name: str
    x: float
    y: float


# Ordered list — index == id, used as the class index when training the
# pitch-keypoint detector (src/calibration/keypoint_detector.py).
KEYPOINTS = [
    Keypoint(0, "corner_top_left", 0.0, 0.0),
    Keypoint(1, "corner_bottom_left", 0.0, PITCH_WIDTH),
    Keypoint(2, "corner_top_right", PITCH_LENGTH, 0.0),
    Keypoint(3, "corner_bottom_right", PITCH_LENGTH, PITCH_WIDTH),
    Keypoint(4, "halfway_top", _mid_x, 0.0),
    Keypoint(5, "halfway_bottom", _mid_x, PITCH_WIDTH),
    Keypoint(6, "center_spot", _mid_x, _mid_y),
    Keypoint(7, "center_circle_top", _mid_x, _mid_y - CENTER_CIRCLE_RADIUS),
    Keypoint(8, "center_circle_bottom", _mid_x, _mid_y + CENTER_CIRCLE_RADIUS),
    Keypoint(9, "center_circle_left", _mid_x - CENTER_CIRCLE_RADIUS, _mid_y),
    Keypoint(10, "center_circle_right", _mid_x + CENTER_CIRCLE_RADIUS, _mid_y),
    Keypoint(11, "left_penalty_area_top_left", 0.0, _pa_y0),
    Keypoint(12, "left_penalty_area_top_right", PENALTY_AREA_DEPTH, _pa_y0),
    Keypoint(13, "left_penalty_area_bottom_left", 0.0, _pa_y1),
    Keypoint(14, "left_penalty_area_bottom_right", PENALTY_AREA_DEPTH, _pa_y1),
    Keypoint(15, "left_six_yard_top_left", 0.0, _sy_y0),
    Keypoint(16, "left_six_yard_top_right", SIX_YARD_DEPTH, _sy_y0),
    Keypoint(17, "left_six_yard_bottom_left", 0.0, _sy_y1),
    Keypoint(18, "left_six_yard_bottom_right", SIX_YARD_DEPTH, _sy_y1),
    Keypoint(19, "left_penalty_spot", PENALTY_SPOT_DISTANCE, _mid_y),
    Keypoint(20, "right_penalty_area_top_left", PITCH_LENGTH - PENALTY_AREA_DEPTH, _pa_y0),
    Keypoint(21, "right_penalty_area_top_right", PITCH_LENGTH, _pa_y0),
    Keypoint(22, "right_penalty_area_bottom_left", PITCH_LENGTH - PENALTY_AREA_DEPTH, _pa_y1),
    Keypoint(23, "right_penalty_area_bottom_right", PITCH_LENGTH, _pa_y1),
    Keypoint(24, "right_six_yard_top_left", PITCH_LENGTH - SIX_YARD_DEPTH, _sy_y0),
    Keypoint(25, "right_six_yard_top_right", PITCH_LENGTH, _sy_y0),
    Keypoint(26, "right_six_yard_bottom_left", PITCH_LENGTH - SIX_YARD_DEPTH, _sy_y1),
    Keypoint(27, "right_six_yard_bottom_right", PITCH_LENGTH, _sy_y1),
    Keypoint(28, "right_penalty_spot", PITCH_LENGTH - PENALTY_SPOT_DISTANCE, _mid_y),
]

KEYPOINT_NAMES = [kp.name for kp in KEYPOINTS]
PITCH_COORDS = [(kp.x, kp.y) for kp in KEYPOINTS]
