#!/usr/bin/env python
"""Overlay live passing structure onto the annotated video: a ring around
whoever currently has the ball, and a fading arrow + label at the moment
each pass is detected.

This is a different view from generate_analytics.py's static passing_network
graph, which collapses the whole match into one summary image (nodes at each
player's average position, edges = total pass counts). This script instead
draws each possession/pass as it happens in time, on top of the already-
annotated video.

Reads outputs/annotated_videos/sample_annotated.mp4 (already has detection
boxes/team colors) rather than re-running detection, since all the position
data needed (per frame, per track_id: pixel_x/pixel_y) is already in the
position log.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analytics.passing_network import DEFAULT_PIXEL_RADIUS_SCALE, compute_pass_events, infer_possessor_per_frame
from src.utils.position_log import load_csv

TEAM_COLORS_BGR = {0: (70, 57, 230), 1: (157, 123, 69)}  # BGR, matches src/pipeline.py's team palette
PASS_COLOR_BGR = (0, 255, 255)
FADE_FRAMES = 40  # how long a pass arrow stays on screen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="outputs/annotated_videos/sample_annotated.mp4")
    parser.add_argument("--position-log", default="outputs/position_log.csv")
    parser.add_argument("--out", default="outputs/annotated_videos/sample_with_passes.mp4")
    parser.add_argument("--possession-radius", type=float, default=3.0,
                         help="Possession radius in meters — only used if calibrated (pitch coordinates present)")
    parser.add_argument("--radius-scale", type=float, default=DEFAULT_PIXEL_RADIUS_SCALE,
                         help="Uncalibrated only: possession radius = this * median player bbox_height per frame")
    parser.add_argument("--min-possession-frames", type=int, default=3)
    args = parser.parse_args()

    df = load_csv(args.position_log)

    possessor = infer_possessor_per_frame(df, args.possession_radius, radius_scale=args.radius_scale)
    possessor_by_frame = {row.frame: (int(row.track_id), row.team) for row in possessor.itertuples()}

    events = compute_pass_events(df, args.possession_radius, args.min_possession_frames, radius_scale=args.radius_scale)
    events_by_frame = {int(row.event_frame): row for row in events.itertuples()}
    print(f"{len(events)} pass events detected (radius_scale={args.radius_scale})")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))  # H.264, browser-playable

    active_events = []  # (event_row, age)
    pass_count = 0
    pos_by_frame_track = {
        (int(r.frame), int(r.track_id)): (r.pixel_x, r.pixel_y)
        for r in df.itertuples()
    }

    for frame_idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx in possessor_by_frame:
            track_id, team = possessor_by_frame[frame_idx]
            pos = pos_by_frame_track.get((frame_idx, track_id))
            if pos is not None:
                color = TEAM_COLORS_BGR.get(team, (200, 200, 200))
                cv2.circle(frame, (int(pos[0]), int(pos[1])), 28, color, 3)

        if frame_idx in events_by_frame:
            active_events.append([events_by_frame[frame_idx], 0])
            pass_count += 1

        for entry in active_events:
            event, age = entry
            alpha = max(0.0, 1.0 - age / FADE_FRAMES)
            overlay = frame.copy()
            from_pt = (int(event.from_x), int(event.from_y))
            to_pt = (int(event.to_x), int(event.to_y))
            cv2.arrowedLine(overlay, from_pt, to_pt, PASS_COLOR_BGR, 3, tipLength=0.08)
            mid = ((from_pt[0] + to_pt[0]) // 2, (from_pt[1] + to_pt[1]) // 2)
            cv2.putText(overlay, f"PASS #{event.from_track}->#{event.to_track}", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, PASS_COLOR_BGR, 2)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
            entry[1] += 1
        active_events = [e for e in active_events if e[1] < FADE_FRAMES]

        cv2.putText(frame, f"Passes detected: {pass_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        writer.write(frame)
        if frame_idx % 200 == 0:
            print(f"  frame {frame_idx}/{total_frames}")

    cap.release()
    writer.release()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
