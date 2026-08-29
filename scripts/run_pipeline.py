#!/usr/bin/env python
"""End-to-end CLI entrypoint. Currently just runs Stage 1 (detection-only);
grows to call tracking/team-classification/calibration/analytics as those
stages land."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    args = parser.parse_args()
    run(args.config)
