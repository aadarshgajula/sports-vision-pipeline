"""Per-frame position log: the shared data format between the pipeline and
the analytics modules (heatmaps, passing network, tactical metrics).

One row per tracked detection per frame. pitch_x/pitch_y are NaN when no
homography is loaded (analytics that need real-world coordinates, like
heatmaps, will need one — see scripts/calibrate_pitch.py). bbox_height is
the detection's pixel bounding-box height — a per-frame proxy for camera
zoom level, used to scale pixel-space distance thresholds (e.g. ball-
possession radius) so they mean roughly the same thing in a tight close-up
and a wide establishing shot, where a fixed pixel threshold wouldn't.
is_predicted is False for every real detection; the ball tracker
(src/tracking/ball_tracker.py) adds True rows for frames where the ball
wasn't actually detected but a Kalman filter bridged the gap from recent
motion — kept distinguishable from real observations rather than presented
as equally reliable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["frame", "track_id", "class_name", "team", "pixel_x", "pixel_y", "pitch_x", "pitch_y",
           "bbox_height", "is_predicted"]


class PositionLogger:
    def __init__(self):
        self._rows = []

    def add(
        self,
        frame: int,
        track_id: int,
        class_name: str,
        team: int | None,
        pixel_xy: tuple[float, float],
        bbox_height: float,
        pitch_xy: tuple[float, float] | None = None,
        is_predicted: bool = False,
    ):
        self._rows.append(
            {
                "frame": frame,
                "track_id": track_id,
                "class_name": class_name,
                "team": team,
                "pixel_x": pixel_xy[0],
                "pixel_y": pixel_xy[1],
                "pitch_x": pitch_xy[0] if pitch_xy is not None else float("nan"),
                "pitch_y": pitch_xy[1] if pitch_xy is not None else float("nan"),
                "bbox_height": bbox_height,
                "is_predicted": is_predicted,
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=COLUMNS)

    def save_csv(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
