"""ByteTrack wrapper: assigns stable track IDs to detections across frames."""

from __future__ import annotations

import supervision as sv


class Tracker:
    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self.tracker.update_with_detections(detections)

    def reset(self):
        self.tracker.reset()

    @classmethod
    def from_config(cls, config: dict) -> "Tracker":
        track_cfg = config["tracking"]
        return cls(
            track_activation_threshold=track_cfg.get("track_activation_threshold", 0.25),
            lost_track_buffer=track_cfg.get("lost_track_buffer", 30),
            minimum_matching_threshold=track_cfg.get("minimum_matching_threshold", 0.8),
            frame_rate=track_cfg.get("frame_rate", 30),
        )
