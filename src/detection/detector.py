"""YOLO-based detector wrapper, returning results as supervision.Detections.

Two modes, controlled by config:
  - Fallback mode (classes=null in config): uses stock YOLOv8n (COCO weights),
    filtered to "person" (class 0) and "sports ball" (class 32). Good enough
    to verify the pipeline end-to-end before any custom training — it won't
    distinguish player vs. referee vs. goalkeeper (COCO has no such classes),
    and ball detections will be noisier/patchier than a model actually
    fine-tuned for small, fast-moving footballs. Ball detections need a much
    lower confidence threshold than person detections (the ball is a tiny,
    motion-blurred object to a general-purpose COCO model), so fallback mode
    runs two passes and merges them.
  - Custom mode (classes=[...] in config): points at a checkpoint fine-tuned
    on a player/ball/referee/goalkeeper dataset (see scripts/train_detector.py),
    returning all configured classes untouched in a single pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import supervision as sv
from ultralytics import YOLO

COCO_PERSON_CLASS_ID = 0
COCO_SPORTS_BALL_CLASS_ID = 32
FALLBACK_BALL_CONFIDENCE_THRESHOLD = 0.1


class Detector:
    def __init__(
        self,
        model_path: str,
        device: str = "mps",
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        classes: list[int] | None = None,
        ball_confidence_threshold: float = FALLBACK_BALL_CONFIDENCE_THRESHOLD,
        class_name_overrides: dict[int, str] | None = None,
        singleton_class_ids: list[int] | None = None,
    ):
        self.model = YOLO(model_path)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.ball_confidence_threshold = ball_confidence_threshold
        self._is_fallback = classes is None
        # Classes that can only ever have exactly one true instance in frame
        # (there's one ball in play) — keep only the top-confidence detection
        # for these, dropping the rest as false positives rather than letting
        # duplicates corrupt anything downstream that assumes one row per
        # frame (e.g. src/analytics/passing_network.py's ball-position merge).
        self.singleton_class_ids = set(singleton_class_ids or [])
        if class_name_overrides:
            # The rest of the pipeline (src/pipeline.py, src/analytics/) hardcodes
            # checks against the COCO fallback's class names ("person", "sports
            # ball") rather than every possible custom dataset's own naming (e.g.
            # "player", "ball") — remap here once instead of threading a naming
            # convention through every downstream check.
            self.model.model.names = {**self.model.names, **class_name_overrides}

    def _infer(self, frame: np.ndarray, classes: list[int], conf: float) -> sv.Detections:
        result = self.model(
            frame,
            device=self.device,
            conf=conf,
            iou=self.iou_threshold,
            classes=classes,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)

    def _dedup_singletons(self, detections: sv.Detections) -> sv.Detections:
        if not self.singleton_class_ids or len(detections) == 0:
            return detections
        keep = np.ones(len(detections), dtype=bool)
        for class_id in self.singleton_class_ids:
            idx = np.where(detections.class_id == class_id)[0]
            if len(idx) > 1:
                drop = idx[np.argsort(-detections.confidence[idx])[1:]]
                keep[drop] = False
        return detections[keep]

    def detect(self, frame: np.ndarray) -> sv.Detections:
        if not self._is_fallback:
            return self._dedup_singletons(self._infer(frame, self.classes, self.confidence_threshold))

        people = self._infer(frame, [COCO_PERSON_CLASS_ID], self.confidence_threshold)
        ball = self._infer(frame, [COCO_SPORTS_BALL_CLASS_ID], self.ball_confidence_threshold)
        if len(ball) > 1:
            # Exactly one ball is ever in play — keep only the top candidate.
            ball = ball[[int(np.argmax(ball.confidence))]]
        return sv.Detections.merge([people, ball])

    @property
    def class_names(self) -> dict[int, str]:
        return self.model.names

    @classmethod
    def from_config(cls, config: dict) -> "Detector":
        det_cfg = config["detection"]
        return cls(
            model_path=det_cfg["model_path"],
            device=config.get("device", "mps"),
            confidence_threshold=det_cfg.get("confidence_threshold", 0.3),
            iou_threshold=det_cfg.get("iou_threshold", 0.5),
            classes=det_cfg.get("classes"),
            ball_confidence_threshold=det_cfg.get("ball_confidence_threshold", FALLBACK_BALL_CONFIDENCE_THRESHOLD),
            class_name_overrides=det_cfg.get("class_name_overrides"),
            singleton_class_ids=det_cfg.get("singleton_class_ids"),
        )
