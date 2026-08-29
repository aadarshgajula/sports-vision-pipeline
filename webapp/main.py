"""Local web app: upload a tactical-cam video, calibrate it once (the one
allowed human step), and get an annotated video, 2D top-down view, heatmaps,
and a passing network — no further manual steps, config editing, or script
invocations needed.

Run with:
    uvicorn webapp.main:app --reload
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from configs.field_config import KEYPOINTS
from webapp import jobs

app = FastAPI(title="Sports Vision Pipeline")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/keypoints")
def get_keypoints():
    return [{"id": kp.id, "name": kp.name, "x": kp.x, "y": kp.y} for kp in KEYPOINTS]


@app.post("/api/upload")
async def upload_video(file: UploadFile):
    data = await file.read()
    try:
        job = jobs.create_job(data, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "job_id": job["id"],
        "fps": job["fps"],
        "frame_count": job["frame_count"],
        "width": job["width"],
        "height": job["height"],
    }


@app.get("/api/jobs/{job_id}/frame/{index}")
def get_frame(job_id: str, index: int):
    try:
        jpeg = jobs.get_frame_jpeg(job_id, index)
    except KeyError:
        raise HTTPException(404, "Unknown job")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return Response(content=jpeg, media_type="image/jpeg")


@app.post("/api/jobs/{job_id}/calibrate")
async def calibrate(job_id: str, correspondences: list[dict]):
    try:
        quality = jobs.save_calibration(job_id, correspondences)
    except KeyError:
        raise HTTPException(404, "Unknown job")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return quality


@app.post("/api/jobs/{job_id}/run")
def run_job(job_id: str):
    try:
        job = jobs.get_job(job_id)
    except KeyError:
        raise HTTPException(404, "Unknown job")
    if not job["homography_path"]:
        raise HTTPException(400, "Calibrate this video before running analysis")
    jobs.start_job(job_id)
    return {"status": "started"}


@app.get("/api/jobs/{job_id}/status")
def job_status(job_id: str):
    try:
        job = jobs.get_job(job_id)
    except KeyError:
        raise HTTPException(404, "Unknown job")
    return {
        "stage": job["stage"],
        "progress": job["progress"],
        "error": job["error"],
        "warnings": job["warnings"],
        "results": job["results"],
        "calibration_quality": job["calibration_quality"],
    }


@app.get("/api/jobs/{job_id}/files/{path:path}")
def job_file(job_id: str, path: str):
    try:
        job = jobs.get_job(job_id)
    except KeyError:
        raise HTTPException(404, "Unknown job")
    file_path = (Path(job["dir"]) / path).resolve()
    if not str(file_path).startswith(str(Path(job["dir"]).resolve())):
        raise HTTPException(403, "Invalid path")
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(file_path))
