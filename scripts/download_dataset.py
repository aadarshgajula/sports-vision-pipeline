#!/usr/bin/env python
"""Download a labeled object-detection dataset from Roboflow Universe in
YOLOv8 format, for fine-tuning src/detection/detector.py past the COCO
fallback (see requirements.txt's roboflow dependency).

Needs a (free) Roboflow account + API key: roboflow.com -> Settings -> API
Keys. Pass it via the ROBOFLOW_API_KEY env var, not a CLI flag (avoids it
ending up in shell history / process listings).

Find a dataset on https://universe.roboflow.com by searching e.g. "football
player detection" — you want one with ball/player/referee/goalkeeper
classes. Its workspace/project/version show up in the URL and on the
dataset's page (also in the auto-generated download code snippet Roboflow
shows you, under the "YOLOv8" export format).

Usage:
    export ROBOFLOW_API_KEY=your_key_here
    python scripts/download_dataset.py --workspace <workspace-slug> \
        --project <project-slug> --version <version-number>
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_OUT_DIR = "data/datasets/detection"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, help="Roboflow workspace slug")
    parser.add_argument("--project", required=True, help="Roboflow project slug")
    parser.add_argument("--version", type=int, required=True, help="Dataset version number")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY in your environment first (see this script's docstring)")

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    # overwrite=True: the SDK silently no-ops (no error, no files written) if
    # the target directory already exists with anything in it (even just a
    # placeholder .gitkeep) — worth knowing since that failure is silent.
    dataset = version.download("yolov8", location=args.out_dir, overwrite=True)

    print(f"\nDownloaded to {dataset.location}")
    print(f"data.yaml at {dataset.location}/data.yaml — check its `names:` list matches "
          f"what you expect (ball/player/referee/goalkeeper or similar) before training.")
    print(f"Next: python scripts/train_detector.py --data {dataset.location}/data.yaml")


if __name__ == "__main__":
    main()
