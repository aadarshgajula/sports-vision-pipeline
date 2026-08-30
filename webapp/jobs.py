"""In-memory job state + background execution for the local web app.

Each uploaded video becomes a "job": a directory under data/webapp_jobs/{id}/
holding the video, its calibration, and every generated output. Jobs run as
subprocesses (not direct function calls) — reusing scripts/*.py and
src/pipeline.py exactly as already proven throughout this project, and
isolating a crash in one video's processing from taking down the whole
server.

No database: JOBS is a plain in-memory dict. Fine for a local, single-user
app that isn't expected to survive a server restart mid-job.
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import cv2
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "data" / "webapp_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

DFL_MODEL_PATH = "data/models/dfl_ball_player_ref_v3_best.pt"

# Quality-check thresholds — the automated replacement for a human (or an AI
# assistant) eyeballing the output. See README/conversation history: both of
# these are real failure modes that were caught manually before this existed.
MIN_MINORITY_TEAM_FRACTION = 0.15  # below this, team split looks collapsed into one cluster
MIN_REAL_BALL_FRACTION = 0.15      # below this, ball tracking is mostly guesswork

JOBS: dict[str, dict] = {}


def _python() -> str:
    return sys.executable


def create_job(video_bytes: bytes, filename: str) -> dict:
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    suffix = Path(filename).suffix or ".mp4"
    video_path = job_dir / f"video{suffix}"
    video_path.write_bytes(video_bytes)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open uploaded file as a video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    JOBS[job_id] = {
        "id": job_id,
        "dir": str(job_dir),
        "video_path": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "stage": "uploaded",
        "progress": 0.0,
        "error": None,
        "homography_path": None,
        "calibration_quality": None,
        "results": None,
        "warnings": [],
        "formations": [],
    }
    return JOBS[job_id]


def get_job(job_id: str) -> dict:
    if job_id not in JOBS:
        raise KeyError(job_id)
    return JOBS[job_id]


def get_frame_jpeg(job_id: str, index: int) -> bytes:
    job = get_job(job_id)
    cap = cv2.VideoCapture(job["video_path"])
    index = max(0, min(index, job["frame_count"] - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {index}")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("JPEG encode failed")
    return buf.tobytes()


def save_calibration(job_id: str, correspondences: list[dict]) -> dict:
    """correspondences: [{"pixel": [x,y], "pitch": [x,y], "name": str}, ...]"""
    import numpy as np

    from src.calibration.homography import ViewTransformer

    job = get_job(job_id)
    if len(correspondences) < 4:
        raise ValueError(f"Need at least 4 points, got {len(correspondences)}")

    source = np.array([c["pixel"] for c in correspondences], dtype=float)
    target = np.array([c["pitch"] for c in correspondences], dtype=float)
    vt = ViewTransformer(source, target)

    homography_path = Path(job["dir"]) / "homography.npy"
    vt.save(str(homography_path))

    # Reprojection error — the same manual sanity check done earlier in this
    # project, now automatic so a coach gets it without needing anyone to
    # eyeball the numbers for them.
    reprojected = vt.transform_points(source)
    errors = np.linalg.norm(reprojected - target, axis=1)

    quality = {
        "mean_error_m": float(errors.mean()),
        "max_error_m": float(errors.max()),
        "num_points": len(correspondences),
    }

    job["homography_path"] = str(homography_path)
    job["calibration_quality"] = quality
    job["stage"] = "calibrated"
    return quality


def _write_pipeline_config(job: dict) -> Path:
    config = {
        "device": "mps",
        "detection": {
            "model_path": DFL_MODEL_PATH,
            "confidence_threshold": 0.25,
            "iou_threshold": 0.5,
            "classes": [0, 1, 2],
            "class_name_overrides": {0: "sports ball", 1: "person"},
            "singleton_class_ids": [0],
        },
        "tracking": {
            "track_activation_threshold": 0.25,
            "lost_track_buffer": 30,
            "minimum_matching_threshold": 0.8,
            "frame_rate": int(round(job["fps"])),
        },
        "team_classification": {
            "enabled": True,
            "min_distinct_tracks": 10,
            "max_wait_frames": 500,
            "max_fit_crops": 60,
        },
        "ball_tracking": {
            "enabled": True,
            "max_coast_frames": 15,
            "process_noise": 5.0,
            "measurement_noise": 10.0,
        },
        "calibration": {"homography_path": job["homography_path"]},
        "paths": {
            "raw_video": job["video_path"],
            "output_video": str(Path(job["dir"]) / "annotated.mp4"),
            "position_log": str(Path(job["dir"]) / "position_log.csv"),
            "models_dir": "data/models",
        },
    }
    config_path = Path(job["dir"]) / "pipeline_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


_FRAME_LOG_RE = re.compile(r"^\s*frame (\d+):")


def _run_subprocess(cmd: list[str], log_path: Path, job: dict | None = None,
                     progress_range: tuple[float, float] | None = None) -> int:
    """Runs a script as a subprocess, streaming its output to a log file. If
    job + progress_range are given, parses src/pipeline.py's periodic
    "frame N: ..." lines to update job progress live — without this, a
    multi-minute pipeline run would leave the UI's progress bar frozen,
    which reads as hung rather than working."""
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            log_file.write(line)
            if job is not None and progress_range is not None:
                match = _FRAME_LOG_RE.match(line)
                if match and job.get("frame_count"):
                    frac = min(int(match.group(1)) / job["frame_count"], 1.0)
                    lo, hi = progress_range
                    job["progress"] = lo + frac * (hi - lo)
        proc.wait()
    return proc.returncode


def _run_quality_checks(job: dict):
    position_log = Path(job["dir"]) / "position_log.csv"
    if not position_log.exists():
        return
    df = pd.read_csv(position_log)
    warnings = []

    person = df[df["class_name"] == "person"]
    team_counts = person["team"].value_counts(dropna=True)
    if len(team_counts) < 2:
        warnings.append("Team classification did not separate two teams — results may be unreliable.")
    else:
        minority_fraction = team_counts.min() / team_counts.sum()
        if minority_fraction < MIN_MINORITY_TEAM_FRACTION:
            warnings.append(
                f"Team split looks lopsided ({minority_fraction:.0%} vs "
                f"{1 - minority_fraction:.0%}) — team classification may have failed "
                f"to separate the two sides correctly."
            )

    ball = df[df["class_name"] == "sports ball"]
    if len(ball) == 0:
        warnings.append("No ball detected in this video at all — ball tracking and passing network will be empty.")
    else:
        real_fraction = (~ball["is_predicted"]).sum() / job["frame_count"]
        if real_fraction < MIN_REAL_BALL_FRACTION:
            warnings.append(
                f"Ball was only actually detected in {real_fraction:.0%} of frames — "
                f"passing network and ball-position stats may be unreliable."
            )

    if job.get("calibration_quality", {}).get("mean_error_m", 0) > 1.0:
        warnings.append(
            f"Calibration reprojection error was {job['calibration_quality']['mean_error_m']:.2f}m "
            f"— consider re-calibrating with more spread-out, precisely-clicked points."
        )

    job["warnings"] = warnings


_FORMATION_LINE_RE = re.compile(r"^\s*Team (\d+): (.+?) \(confidence=(\w+)\)")
_CAVEAT_LINE_RE = re.compile(r"^\s*caveat: (.+)$")


def _parse_formation_summary(log_path: Path) -> list[dict]:
    """Reads generate_analytics.py's printed formation summary back out of
    its log, so the UI can show the label/confidence/caveats as text next
    to the diagram image rather than making a viewer read tiny text baked
    into the PNG."""
    if not log_path.exists():
        return []
    formations = []
    current = None
    for line in log_path.read_text().splitlines():
        match = _FORMATION_LINE_RE.match(line)
        if match:
            if current:
                formations.append(current)
            current = {"team": int(match.group(1)), "label": match.group(2),
                       "confidence": match.group(3), "caveats": []}
            continue
        caveat_match = _CAVEAT_LINE_RE.match(line)
        if caveat_match and current:
            current["caveats"].append(caveat_match.group(1))
    if current:
        formations.append(current)
    return formations


def _run_job_thread(job_id: str):
    job = JOBS[job_id]
    try:
        job["stage"] = "running_pipeline"
        job["progress"] = 0.05
        config_path = _write_pipeline_config(job)
        log_path = Path(job["dir"]) / "pipeline.log"
        rc = _run_subprocess(
            [_python(), "-m", "src.pipeline", "--config", str(config_path)], log_path,
            job=job, progress_range=(0.05, 0.7),
        )
        if rc != 0:
            raise RuntimeError(f"Pipeline failed — see {log_path.name} for details")

        job["stage"] = "running_analytics"
        job["progress"] = 0.7
        heatmaps_dir = Path(job["dir"]) / "heatmaps"
        passing_dir = Path(job["dir"]) / "passing_graphs"
        formations_dir = Path(job["dir"]) / "formations"
        analytics_log = Path(job["dir"]) / "analytics.log"
        _run_subprocess(
            [_python(), "scripts/generate_analytics.py",
             "--position-log", str(Path(job["dir"]) / "position_log.csv"),
             "--heatmaps-dir", str(heatmaps_dir),
             "--passing-graphs-dir", str(passing_dir),
             "--formations-dir", str(formations_dir)],
            analytics_log,
        )
        job["formations"] = _parse_formation_summary(analytics_log)

        job["stage"] = "rendering_topdown"
        job["progress"] = 0.85
        _run_subprocess(
            [_python(), "scripts/render_topdown_view.py",
             "--position-log", str(Path(job["dir"]) / "position_log.csv"),
             "--out", str(Path(job["dir"]) / "topdown.mp4"),
             "--fps", str(job["fps"])],
            Path(job["dir"]) / "topdown.log",
        )

        _run_quality_checks(job)

        job["results"] = {
            "annotated_video": "annotated.mp4",
            "topdown_video": "topdown.mp4",
            "position_log": "position_log.csv",
            "heatmap_team_0": "heatmaps/team_0_heatmap.png",
            "heatmap_team_1": "heatmaps/team_1_heatmap.png",
            "passing_network": "passing_graphs/passing_network.png",
            "formation_team_0": "formations/team_0_formation.png",
            "formation_team_1": "formations/team_1_formation.png",
        }
        job["stage"] = "done"
        job["progress"] = 1.0
    except Exception as e:  # noqa: BLE001 — surface any failure to the UI, don't crash the server
        job["stage"] = "error"
        job["error"] = str(e)


def start_job(job_id: str):
    thread = threading.Thread(target=_run_job_thread, args=(job_id,), daemon=True)
    thread.start()
