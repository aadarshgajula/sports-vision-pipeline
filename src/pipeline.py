"""End-to-end pipeline: detect -> track -> classify teams -> (optionally)
calibrate to pitch coordinates -> annotate video + write a position log for
the analytics stage (src/analytics/).

Team classification is nearest-anchor color matching, not unsupervised
clustering: the user provides one example player crop per team (and
optionally the referee) during the video's one-time manual calibration step
(webapp/jobs.py), and each track is classified by which anchor its jersey
color is closest to. Each track is reclassified on its first few sightings
and locked in by majority vote rather than trusting a single crop — one
blurry or backlit frame shouldn't permanently decide a track's team. Once
locked in, later sightings are a cached dict lookup, not a model call.

Calibration (pixel -> pitch meters) is optional: without a homography file
(see scripts/calibrate_pitch.py), pitch_x/pitch_y in the position log stay
NaN, and analytics that need real-world coordinates (heatmaps, tactical
metrics) won't run — but tracking, team classification, the annotated video,
and the pixel-space passing network all still work.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
    """Classifies each track's team by nearest-anchor color distance (see
    TeamClassifier), using one example crop per team — and optionally the
    referee — provided once during the video's manual calibration step
    (webapp/jobs.py). Locks in each track's team by majority vote over its
    first few sightings rather than trusting a single crop — one blurry or
    backlit frame shouldn't permanently decide a track's team. Once locked
    in, later frames are a dict lookup, not a model call.
    """

    SMOOTHING_MIN_VOTES = 5  # classifications needed before a track's team locks in

    def __init__(self, anchor_crops: dict[int | None, np.ndarray]):
        self.classifier = TeamClassifier()
        self.classifier.fit_from_anchors(anchor_crops)
        self._votes: dict[int, Counter] = defaultdict(Counter)
        self._team_cache: dict[int, int | None] = {}

    def _record_vote(self, track_id: int, team: int | None) -> int | None:
        """Adds one classification to a track's running tally, locking in
        the majority result once enough independent crops have weighed in.
        Returns the current best guess (locked or still-running) so callers
        always have a usable value, not a long stretch of None while votes
        accumulate."""
        if track_id in self._team_cache:
            return self._team_cache[track_id]
        votes = self._votes[track_id]
        votes[team] += 1
        if sum(votes.values()) >= self.SMOOTHING_MIN_VOTES:
            self._team_cache[track_id] = votes.most_common(1)[0][0]
        return votes.most_common(1)[0][0]

    def assign(self, crops: list[np.ndarray], track_ids: list[int]) -> list[int | None]:
        teams: list[int | None] = []
        pending_crops, pending_track_ids, pending_idx = [], [], []
        for i, (crop, track_id) in enumerate(zip(crops, track_ids)):
            if track_id in self._team_cache:
                teams.append(self._team_cache[track_id])
            else:
                teams.append(None)  # placeholder, filled in below
                pending_crops.append(crop)
                pending_track_ids.append(track_id)
                pending_idx.append(i)

        if pending_crops:
            predictions = self.classifier.predict(pending_crops)
            for i, track_id, team in zip(pending_idx, pending_track_ids, predictions):
                teams[i] = self._record_vote(track_id, team)

        return teams


def run(config_path: str):
    config = load_config(config_path)
    paths = config["paths"]

    detector = Detector.from_config(config)
    tracker = Tracker.from_config(config)

    team_cfg = config.get("team_classification", {})
    team_assigner = None
    if team_cfg.get("enabled", True):
        anchor_paths = team_cfg.get("anchors", {})
        if 0 not in anchor_paths or 1 not in anchor_paths:
            raise ValueError(
                "team_classification.anchors needs at least a '0' and '1' entry "
                "(one example player crop per team, from the calibration step)"
            )
        anchor_crops = {label: cv2.imread(path) for label, path in anchor_paths.items()}
        # YAML can't have a None key; the referee/other anchor is keyed "referee" in
        # config but None everywhere else (matching the position log's team=None).
        if "referee" in anchor_crops:
            anchor_crops[None] = anchor_crops.pop("referee")
        team_assigner = TeamAssigner(anchor_crops)

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
            person_teams = team_assigner.assign(crops, track_ids)
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
