"""Offline ball-track cleaning, shared by anything that consumes a position
log's ball rows: linear interpolation across short gaps, and rejection of
single-frame jumps too fast to be the real ball (a false positive latching
onto a crowd/ad-board object).

This is a second layer on top of src/tracking/ball_tracker.py's online
Kalman filter, not a replacement for it. The Kalman tracker runs during the
pipeline itself and can only coast forward on past velocity — it can't know
where the ball reappears, so it gives up (and logs nothing) past
max_coast_frames. This module runs afterward on the complete position log,
so it can interpolate straight to where the ball is actually next observed,
smoothly bridging gaps the online tracker couldn't. Gaps longer than
max_gap_frames are left alone rather than interpolated — bridging a scene
cut or a genuinely long loss (ball left frame) with a straight line would
show confident, invented motion instead of an honest "we lost it."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from configs.field_config import PITCH_LENGTH, PITCH_WIDTH

DEFAULT_MAX_BALL_GAP_FRAMES = 10  # interpolate ball position across gaps up to this long
DEFAULT_MAX_BALL_JUMP_SCALE = 5.0  # reject a ball detection that jumps > this * local player scale

# Generous slack beyond the touchline/goal line — enough for a throw-in or a
# ball that's rolled out of play, not enough to accept the kind of position
# the online Kalman tracker can produce mid-coast: with no correction for
# several frames it extrapolates in a straight line on whatever velocity it
# last had, and a slightly-off velocity estimate compounds every frame,
# easily drifting tens of meters past the pitch edge before max_coast_frames
# gives up. Those points must never anchor an interpolation (they'd draw a
# confident straight line through nonsense) — this check is pitch-space
# only, since pixel-space has no universal "off the pitch" definition.
PITCH_BOUNDS_MARGIN_M = 10


def _drop_implausible_pitch_positions(ball: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    if xcol != "pitch_x":
        return ball
    x_ok = ball[xcol].between(-PITCH_BOUNDS_MARGIN_M, PITCH_LENGTH + PITCH_BOUNDS_MARGIN_M)
    y_ok = ball[ycol].between(-PITCH_BOUNDS_MARGIN_M, PITCH_WIDTH + PITCH_BOUNDS_MARGIN_M)
    return ball[x_ok & y_ok]


def clean_ball_track(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    player_scale: pd.Series,
    max_gap_frames: int = DEFAULT_MAX_BALL_GAP_FRAMES,
    max_jump_scale: float = DEFAULT_MAX_BALL_JUMP_SCALE,
) -> pd.DataFrame:
    """Returns a frame-indexed ball position series (columns xcol, ycol,
    was_interpolated) with outlier jumps and implausible pitch positions
    dropped, and short gaps linearly interpolated. was_interpolated marks
    rows that were filled in rather than a real (or already Kalman-
    corrected) observation from the log."""
    ball = df[df["class_name"] == "sports ball"][["frame", xcol, ycol]].dropna()
    ball = _drop_implausible_pitch_positions(ball, xcol, ycol)
    if ball.empty:
        return ball.set_index("frame").assign(was_interpolated=pd.Series(dtype=bool))
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
        return pd.DataFrame(columns=[xcol, ycol, "was_interpolated"])

    cleaned = pd.DataFrame(cleaned_rows, columns=["frame", xcol, ycol]).set_index("frame")
    full_index = pd.RangeIndex(cleaned.index.min(), cleaned.index.max() + 1, name="frame")
    reindexed = cleaned.reindex(full_index)
    was_missing = reindexed[xcol].isna()
    interpolated = reindexed.interpolate(method="linear", limit=max_gap_frames, limit_area="inside")
    interpolated["was_interpolated"] = was_missing & interpolated[xcol].notna()
    return interpolated
