"""Pixel-to-pitch coordinate transform via homography.

Fit ViewTransformer from >=4 (pixel_xy, pitch_xy) correspondences — usually
produced by scripts/calibrate_pitch.py (manual click tool) or, once trained,
a pitch-keypoint detector (src/calibration/keypoint_detector.py). Then call
transform_points on player/ball pixel positions (typically bounding-box
bottom-center, i.e. where the feet touch the ground) to get real-world pitch
meters, using the layout in configs/field_config.py.

Caveat: a single homography only holds for a static camera. A broadcast feed
that pans/zooms/cuts needs a fresh homography per shot (or per frame, via a
keypoint detector run continuously) — this module doesn't attempt scene-cut
detection or re-calibration triggers.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class ViewTransformer:
    def __init__(self, source_points: np.ndarray, target_points: np.ndarray):
        source_points = source_points.astype(np.float32)
        target_points = target_points.astype(np.float32)
        if len(source_points) < 4 or len(target_points) < 4:
            raise ValueError("Need at least 4 point correspondences to fit a homography")
        self.homography, _ = cv2.findHomography(source_points, target_points, method=cv2.RANSAC)
        if self.homography is None:
            raise ValueError("Homography estimation failed — check for collinear/degenerate points")

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(points, self.homography)
        return transformed.reshape(-1, 2)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.homography)

    @classmethod
    def load(cls, path: str) -> "ViewTransformer":
        instance = cls.__new__(cls)
        instance.homography = np.load(path)
        return instance

    @classmethod
    def from_correspondence_file(cls, path: str) -> "ViewTransformer":
        with open(path) as f:
            data = json.load(f)
        source_points = np.array([c["pixel"] for c in data["correspondences"]])
        target_points = np.array([c["pitch"] for c in data["correspondences"]])
        return cls(source_points, target_points)


def bottom_center(xyxy: np.ndarray) -> np.ndarray:
    """Foot point for a bounding box: bottom-center, where a player touches the pitch."""
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    return np.stack([(x1 + x2) / 2, y2], axis=1)
