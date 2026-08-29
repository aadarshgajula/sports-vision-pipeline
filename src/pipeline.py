"""End-to-end pipeline: detect -> track -> classify teams -> (optionally)
calibrate to pitch coordinates -> annotate video + write a position log for
the analytics stage (src/analytics/).

Team classification is fit once a diverse-enough buffer of distinct-track
torso crops is gathered (not just "the first N frames" — a broadcast feed
often opens on a tight 2-player close-up, and fitting the color clusters on
2 individuals means the classifier learns "person A vs person B," not
"kit color A vs kit color B," and won't generalize once more players enter
frame). After fitting, each track's team is decided once and cached — re-
running SigLIP on every detection every frame would be wasteful and would
risk a track flip-flopping between teams frame to frame.

Calibration (pixel -> pitch meters) is optional: without a homography file
(see scripts/calibrate_pitch.py), pitch_x/pitch_y in the position log stay
NaN, and analytics that need real-world coordinates (heatmaps, tactical
metrics) won't run — but tracking, team classification, the annotated video,
and the pixel-space passing network all still work.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import yaml

from src.calibration.homography import ViewTransformer, bottom_center
from src.detection.detector import Detector
from src.team_classification.classifier import TeamClassifier, crop_torso
from src.tracking.ball_tracker import BallTracker
from src.tracking.tracker import Tracker
from src.utils.position_log import PositionLogger
from src.utils.video_io import VideoReader, VideoWriter

TEAM_COLORS = {0: sv.Color(230, 57, 70), 1: sv.Color(69, 123, 157)}
UNASSIGNED_COLOR = sv.Color(153, 153, 153)
BALL_COLOR = sv.Color(255, 255, 0)
PREDICTED_BALL_COLOR_BGR = (0, 200, 255)  # BGR for direct cv2 drawing
BALL_TRACK_ID = -1  # ball's own track_id from ByteTrack is never used downstream; keep it constant


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class TeamAssigner:
    """Fits a TeamClassifier once enough distinct players have been seen,
    then caches a team id per track_id so later frames are a dict lookup,
    not a model call.

    Buffers ONE torso crop per distinct track_id (first sighting), not every
    crop from every frame — repeated crops of the same 1-2 players (e.g. a
    tight close-up shot at the start of a clip) wouldn't give the color
    clusters enough real diversity to generalize to players never seen
    during fitting.
    """

    MIN_FIT_CROPS = 20  # UMAP's spectral init needs enough samples relative to n_neighbors

    def __init__(self, device: str, min_distinct_tracks: int, max_wait_frames: int, max_fit_crops: int):
        self.classifier = TeamClassifier(device=device)
        self.min_distinct_tracks = min_distinct_tracks
        self.max_wait_frames = max_wait_frames
        self.max_fit_crops = max_fit_crops
        self._buffer_crop_by_track: dict[int, np.ndarray] = {}
        self._fitted = False
        self._gave_up = False
        self._team_cache: dict[int, int] = {}

    def _maybe_fit(self, frame_idx: int):
        if self._fitted or self._gave_up:
            return
        n = len(self._buffer_crop_by_track)
        # Require both: enough distinct players (the real diversity signal) AND
        # enough absolute samples for UMAP's spectral init to behave. If the
        # diversity target is hit early but sample count is still short, keep
        # buffering rather than fitting on too little (or giving up too soon).
        ready = n >= max(self.min_distinct_tracks, self.MIN_FIT_CROPS)
        deadline_hit = frame_idx >= self.max_wait_frames
        if not ready:
            if deadline_hit:
                self._gave_up = True  # never saw enough distinct players; skip team classification
                print(f"TeamAssigner: only {n} distinct tracks by frame {frame_idx} "
                      f"— giving up on team classification for this clip")
            return

        track_ids = list(self._buffer_crop_by_track.keys())
        crops = list(self._buffer_crop_by_track.values())
        self.classifier.fit(crops)
        self._fitted = True
        predictions = self.classifier.predict(crops)
        for track_id, team in zip(track_ids, predictions):
            self._team_cache[track_id] = int(team)

    def assign(self, frame_idx: int, crops: list[np.ndarray], track_ids: list[int]) -> list[int | None]:
        if not self._fitted and not self._gave_up and len(self._buffer_crop_by_track) < self.max_fit_crops:
            for crop, track_id in zip(crops, track_ids):
                self._buffer_crop_by_track.setdefault(track_id, crop)
        self._maybe_fit(frame_idx)

        teams: list[int | None] = []
        uncached_crops, uncached_track_ids = [], []
        for crop, track_id in zip(crops, track_ids):
            if track_id in self._team_cache:
                teams.append(self._team_cache[track_id])
            elif self._fitted:
                uncached_crops.append(crop)
                uncached_track_ids.append(track_id)
                teams.append(None)  # placeholder, filled in below
            else:
                teams.append(None)

        if uncached_crops:
            predictions = self.classifier.predict(uncached_crops)
            pred_by_track = dict(zip(uncached_track_ids, (int(p) for p in predictions)))
            for i, track_id in enumerate(track_ids):
                if track_id in pred_by_track:
                    self._team_cache[track_id] = pred_by_track[track_id]
                    teams[i] = pred_by_track[track_id]

        return teams


def run(config_path: str):
    config = load_config(config_path)
    paths = config["paths"]

    detector = Detector.from_config(config)
    tracker = Tracker.from_config(config)

    team_cfg = config.get("team_classification", {})
    team_assigner = (
        TeamAssigner(
            device=config.get("device", "mps"),
            min_distinct_tracks=team_cfg.get("min_distinct_tracks", 10),
            max_wait_frames=team_cfg.get("max_wait_frames", 500),
            max_fit_crops=team_cfg.get("max_fit_crops", 60),
        )
        if team_cfg.get("enabled", True)
        else None
    )

    ball_track_cfg = config.get("ball_tracking", {})
    ball_tracker = (
        BallTracker(
            max_coast_frames=ball_track_cfg.get("max_coast_frames", 15),
            process_noise=ball_track_cfg.get("process_noise", 5.0),
            measurement_noise=ball_track_cfg.get("measurement_noise", 10.0),
        )
        if ball_track_cfg.get("enabled", True)
        else None
    )

    homography_path = config.get("calibration", {}).get("homography_path")
    view_transformer = None
    if homography_path and Path(homography_path).exists():
        view_transformer = ViewTransformer.load(homography_path)
        print(f"Loaded homography from {homography_path}")
    else:
        print("No homography loaded — position log will be pixel-space only "
              "(heatmaps/tactical metrics need calibration; see scripts/calibrate_pitch.py)")

    logger = PositionLogger()

    reader = VideoReader(paths["raw_video"])
    writer = VideoWriter(paths["output_video"], reader.fps, reader.width, reader.height)

    print(f"Processing {paths['raw_video']} ({reader.frame_count} frames @ {reader.fps:.1f} fps)")

    for frame_idx, frame in enumerate(reader.frames()):
        detections = detector.detect(frame)
        detections = tracker.update(detections)

        is_person = np.array([detector.class_names[c] == "person" for c in detections.class_id])
        foot_points = bottom_center(detections.xyxy)

        teams: list[int | None] = [None] * len(detections)
        if team_assigner is not None and is_person.any():
            person_idx = np.where(is_person)[0]
            crops = [crop_torso(frame, detections.xyxy[i]) for i in person_idx]
            track_ids = [int(detections.tracker_id[i]) for i in person_idx]
            person_teams = team_assigner.assign(frame_idx, crops, track_ids)
            for i, team in zip(person_idx, person_teams):
                teams[i] = team

        pitch_points = (
            view_transformer.transform_points(foot_points) if view_transformer is not None else None
        )

        ball_idx = next((i for i in range(len(detections))
                          if detector.class_names[detections.class_id[i]] == "sports ball"), None)
        ball_tracker_result = None
        if ball_tracker is not None:
            observation = tuple(foot_points[ball_idx]) if ball_idx is not None else None
            ball_tracker_result = ball_tracker.update(observation)

        labels = []
        colors = []
        for i in range(len(detections)):
            class_name = detector.class_names[detections.class_id[i]]
            team = teams[i]
            pixel_xy = tuple(foot_points[i])
            pitch_xy = tuple(pitch_points[i]) if pitch_points is not None else None
            track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1

            if i == ball_idx and ball_tracker_result is not None:
                # Log the Kalman-corrected estimate, not the raw detection —
                # smooths jitter even on frames where the ball WAS detected.
                track_id = BALL_TRACK_ID
                bx, by, _ = ball_tracker_result
                pixel_xy = (bx, by)
                if view_transformer is not None:
                    pitch_xy = tuple(view_transformer.transform_points(np.array([[bx, by]]))[0])

            logger.add(
                frame=frame_idx,
                track_id=track_id,
                class_name=class_name,
                team=team,
                pixel_xy=pixel_xy,
                bbox_height=float(detections.xyxy[i][3] - detections.xyxy[i][1]),
                pitch_xy=pitch_xy,
            )

            if class_name == "sports ball":
                labels.append("ball")
                colors.append(BALL_COLOR)
            else:
                labels.append(f"#{track_id}" + (f" T{team}" if team is not None else ""))
                colors.append(TEAM_COLORS.get(team, UNASSIGNED_COLOR))

        palette = sv.ColorPalette(colors) if colors else sv.ColorPalette.DEFAULT
        color_lookup = np.arange(len(detections))
        annotated = sv.BoxAnnotator(color=palette).annotate(frame.copy(), detections, custom_color_lookup=color_lookup)
        annotated = sv.LabelAnnotator(color=palette).annotate(
            annotated, detections, labels=labels, custom_color_lookup=color_lookup
        )

        if ball_idx is None and ball_tracker_result is not None:
            # No raw ball detection this frame, but the Kalman filter is
            # still coasting on recent motion — log and draw it distinctly
            # (hollow marker) rather than silently passing it off as a real
            # detection.
            bx, by, _ = ball_tracker_result
            pitch_xy = (
                tuple(view_transformer.transform_points(np.array([[bx, by]]))[0])
                if view_transformer is not None else None
            )
            logger.add(
                frame=frame_idx, track_id=BALL_TRACK_ID, class_name="sports ball", team=None,
                pixel_xy=(bx, by), bbox_height=float("nan"), pitch_xy=pitch_xy, is_predicted=True,
            )
            cv2.circle(annotated, (int(bx), int(by)), 10, PREDICTED_BALL_COLOR_BGR, 2)
            cv2.putText(annotated, "ball?", (int(bx) + 12, int(by)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, PREDICTED_BALL_COLOR_BGR, 1)

        writer.write(annotated)

        if frame_idx % 50 == 0:
            print(f"  frame {frame_idx}: {len(detections)} detections")

    reader.release()
    writer.release()
    logger.save_csv(paths["position_log"])
    print(f"Wrote annotated video to {paths['output_video']}")
    print(f"Wrote position log to {paths['position_log']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    args = parser.parse_args()
    run(args.config)
