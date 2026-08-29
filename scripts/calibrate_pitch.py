#!/usr/bin/env python
"""Manual pitch calibration: click pixel positions of known field landmarks
on a single frame, save the correspondences (and resulting homography) for
src/calibration/homography.py to consume.

Run this yourself — it opens an interactive OpenCV window, which needs a
human clicking, so it can't be scripted end-to-end.

Requires a frame where enough pitch landmarks are actually visible (corner
flags, penalty box corners, center circle, etc. — see configs/field_config.py
for the full list). A single homography is only valid for the camera
position/zoom in that exact frame: if the broadcast camera pans, zooms, or
cuts, you need a new calibration for each distinct shot. This won't work
well on a clip that's mostly tight player-following shots with few pitch
lines in view — you need a wide, mostly-static establishing shot.

Usage:
    python scripts/calibrate_pitch.py --video data/raw_videos/sample.webm \
        --frame 0 --out data/models/homography_sample.json

Controls:
    Left-click a landmark, then type its keypoint index (shown in the
    terminal list) and press Enter in the terminal. Press 'q' in the image
    window when done (need >= 4 points, ideally 6-8 spread across the frame
    for a stable fit).
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.field_config import KEYPOINTS
from src.calibration.homography import ViewTransformer

_clicks = []


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicks.append((x, y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", default="data/models/homography_sample.json")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {args.frame} from {args.video}")

    print("Available keypoints (pick the ones actually visible in the frame):")
    for kp in KEYPOINTS:
        print(f"  {kp.id:2d}  {kp.name}  ({kp.x:.2f}, {kp.y:.2f}) m")

    window = "calibrate — click a landmark, then answer the prompt below"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_mouse)

    correspondences = []
    last_click_count = 0
    while True:
        display = frame.copy()
        for (px, py), c in zip([c["pixel"] for c in correspondences], correspondences):
            cv2.circle(display, (int(px), int(py)), 6, (0, 255, 0), -1)
            cv2.putText(display, c["name"], (int(px) + 8, int(py)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)
        cv2.imshow(window, display)
        key = cv2.waitKey(50) & 0xFF

        if len(_clicks) > last_click_count:
            x, y = _clicks[-1]
            last_click_count = len(_clicks)
            keypoint_id = input(f"Clicked ({x},{y}) — keypoint id: ").strip()
            try:
                kp = KEYPOINTS[int(keypoint_id)]
            except (ValueError, IndexError):
                print("  invalid id, discarding this click")
                continue
            correspondences.append({"pixel": [x, y], "pitch": [kp.x, kp.y], "name": kp.name})
            print(f"  added {kp.name} -> pitch ({kp.x:.2f}, {kp.y:.2f})")

        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    if len(correspondences) < 4:
        print(f"Only {len(correspondences)} correspondences — need at least 4. Not saving.")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"video": args.video, "frame": args.frame, "correspondences": correspondences}, f, indent=2)
    print(f"Saved {len(correspondences)} correspondences to {args.out}")

    vt = ViewTransformer.from_correspondence_file(args.out)
    homography_path = str(Path(args.out).with_suffix(".npy"))
    vt.save(homography_path)
    print(f"Saved homography matrix to {homography_path}")


if __name__ == "__main__":
    main()
