"""Thin OpenCV wrappers for reading/writing video frame streams."""

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoReader:
    def __init__(self, path: str):
        self.path = path
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            yield frame

    def release(self):
        self.capture.release()


class VideoWriter:
    def __init__(self, path: str, fps: float, width: int, height: int):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # avc1 (H.264), not mp4v (MPEG-4 Part 2) — browsers' native <video> tag
        # won't play mp4v at all; only found this out once these videos were
        # loaded into a real <video> element instead of Finder/QuickTime,
        # which use system codecs and play mp4v fine either way.
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    def write(self, frame: np.ndarray):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
