# Sports Vision Pipeline

A computer vision pipeline for football broadcast/tactical-camera footage:
player + ball detection, multi-object tracking, team classification, ball-
tracking through detection gaps, camera calibration to real-world pitch
coordinates, and analytics — a 2D top-down tactical view, heatmaps, a
passing network, and team shape metrics — built and tested end-to-end
against real match footage rather than only synthetic data.

Ships as a local web app: upload a video, calibrate it once (the only manual
step), and get results back with no further scripting needed.

## Quick start — the web app

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn webapp.main:app --reload
```

Open `http://127.0.0.1:8000`, upload a video, calibrate it (see below), and
wait — it runs detection, tracking, team classification, ball tracking, and
all analytics automatically, then shows you the results with automated
quality warnings (see [Known limitations](#known-limitations)) if something
looks off.

**This is tuned for wide, fixed tactical-camera footage** (a high, mostly-
static camera showing most of the pitch) — see
[Two different detectors](#two-different-detectors-not-one-universal-model)
below for why that matters.

### Calibration (the one manual step)

Formation/heatmap/tactical-metrics output needs real-world pitch
coordinates, which needs a homography — and nothing in this pipeline can
infer pitch landmarks automatically yet (that would need its own trained
keypoint-detection model; see [Roadmap](#roadmap)). So: pick a frame where
pitch markings are clearly visible using the frame slider, choose a
landmark from the dropdown, click it on the image, repeat for as many
landmarks as you can identify (4 minimum, 6-8 spread across the frame gives
a meaningfully more accurate result), then save. You'll get an immediate
reprojection-error readout — under ~0.5m is good, several meters means
re-click with more care or better-spread points.

A single calibration only holds for one fixed camera position — if your
footage cuts to a different angle partway through, only the portion
matching the calibrated angle will project correctly.

## How it works

| Stage | Code |
|---|---|
| Detection | `src/detection/detector.py` |
| Tracking | `src/tracking/tracker.py` (ByteTrack) |
| Ball tracking (bridges detection gaps) | `src/tracking/ball_tracker.py` (Kalman filter) |
| Team classification | `src/team_classification/classifier.py` (nearest-anchor jersey color match) |
| Calibration | `src/calibration/homography.py`, `scripts/calibrate_pitch.py` |
| Analytics | `src/analytics/` — `heatmaps.py`, `passing_network.py`, `tactical_metrics.py` |
| 2D top-down view | `scripts/render_topdown_view.py` |
| Web app | `webapp/main.py` (FastAPI) + `webapp/jobs.py` (background job execution) |

The web app runs each stage as a subprocess of the existing CLI scripts
rather than reimplementing anything — `python -m src.pipeline`,
`scripts/generate_analytics.py`, and `scripts/render_topdown_view.py` all
work standalone too, for anyone who wants to script this directly instead
of using the browser UI. See each module's docstring for the real detail;
this table is just a map.

## Two different detectors, not one universal model

- **COCO fallback** (`data/models/yolov8n.pt`, auto-downloaded) — generic
  person/ball detection, works on *any* footage but can't tell player from
  referee from goalkeeper (COCO has no such categories), and ball recall is
  patchy (COCO wasn't trained for a small, fast football).
- **Custom-trained tactical-cam model**
  (`data/models/dfl_ball_player_ref_v1_best.pt`, not included in this repo —
  see [Training your own detector](#training-your-own-detector)) — trained
  on ~3,500 images of wide tactical-camera football footage (`ball`,
  `player`, `ref` classes). Excellent on matching footage (in testing: ~82%
  ball-frame coverage vs. ~30% for the fallback on the same clip), but
  **worse than the fallback on standard broadcast TV footage** (varying
  zoom, quick cuts) — it's a specialist, not a strict upgrade. The web app
  always uses this one, on the assumption you're feeding it tactical-cam
  footage; swap `webapp/jobs.py`'s `DFL_MODEL_PATH` back to the fallback if
  you're pointing it at broadcast TV clips instead.

## Known limitations

Stated plainly rather than glossed over — these are real, found by testing
against actual match footage during development, not hypothetical edge
cases:

- **Ball tracking has two layers of gap-bridging, still not gap-free.** The
  online Kalman filter (`src/tracking/ball_tracker.py`) coasts through
  short gaps during the pipeline run; a second offline pass
  (`src/utils/ball_track_cleaning.py`) interpolates further gaps in the
  completed position log, shared by both the 2D top-down view and the
  passing network, with implausible off-pitch positions rejected before
  they can anchor an interpolation. Gaps longer than both layers' limits —
  a real, extended loss (ball left frame, scene cut) — still show as a
  genuine cut rather than an invented trajectory, which is deliberate, not
  a bug.
- **Team classification requires one manual step: an example crop per
  team.** After unsupervised color clustering (SigLIP+KMeans, then several
  hand-engineered color-feature attempts) kept breaking on a new kit
  combination every time one failure mode was patched, it was replaced with
  nearest-anchor matching — the user clicks one example player per team
  (and optionally the referee) during the same one-time calibration step
  already required for pitch homography, and classification becomes
  distance-to-that-example rather than guessed cluster structure. More
  reliable, but it means team classification is only as good as the anchor
  crops picked — a blurry or occluded click will teach it the wrong color.
- **No dedicated goalkeeper anchor** — a goalkeeper's distinct kit color
  gets matched to whichever anchor (team or referee) is closest, since the
  calibration step only asks for one example per team plus the referee.
  Matters for anything describing outfield-only shape (e.g. formation
  recognition, which already excludes the goalkeeper via a positional
  heuristic instead, independent of this).
- **Calibration is entirely manual, one shot per camera angle** — no
  automatic pitch-keypoint detection exists (see Roadmap).
- **MPS (Apple Silicon GPU) training is unreliable.** Fine-tuning a
  detector on `device=mps` hit real, reproducible crashes (a shape-mismatch
  bug deep in ultralytics' loss computation, corroborated by multiple
  upstream GitHub issues) — `scripts/train_detector.py` defaults to
  `--device cpu` for this reason, which is slower but actually finishes.

## Training your own detector

`data/models/*.pt` isn't committed to this repo (see `.gitignore` — trained
checkpoints are regeneratable, not source). To train your own:

```bash
export ROBOFLOW_API_KEY=...   # roboflow.com → Settings → API Keys
python scripts/download_dataset.py --workspace <slug> --project <slug> --version <n>
```

**Check the dataset before trusting it** — Roboflow Universe datasets vary
wildly in quality; one tried during development turned out to be an
inconsistent merge of unrelated labeling projects with contradictory class
indices. If yours has similar issues, `scripts/curate_dataset.py` can
filter to a verified-clean subset by filename prefix + class remapping.

```bash
python scripts/train_detector.py --data <path-to-data.yaml> --epochs 30 --device cpu
```

Expect real wall-clock time on CPU (no dedicated GPU): roughly 5-13
minutes/epoch depending on dataset size and image resolution. Do a quick
`--epochs 2` run first to sanity-check the whole loop before committing to
a long one. Once trained, point `webapp/jobs.py`'s `DFL_MODEL_PATH` (or
`configs/pipeline_config.yaml`'s `detection.model_path` for the CLI path)
at the resulting `best.pt`, with `detection.classes`/`class_name_overrides`
matching your dataset's class order.

## Roadmap

- Formation recognition (team shape as e.g. "4-4-2") from tracked positions
  — in progress.
- Fix the ball-tracking flash/jump issue in the top-down view (likely via
  non-causal interpolation at render time, since that step isn't
  real-time-constrained the way live tracking is).
- Automatic pitch-keypoint detection, to remove the manual calibration step.

## Sample footage

The sample clips referenced during development (`sample.webm`, `sample2.webm`,
`Sample3.mp4`) are excluded from this repo (see `.gitignore`) and are not
included here — if you add your own, note their source/license before
sharing this repo further.
