#!/usr/bin/env python
"""Fine-tune a YOLOv8 checkpoint on a labeled football dataset (see
scripts/download_dataset.py), to replace the COCO fallback in
src/detection/detector.py with a model that actually knows about the ball,
referee, and goalkeeper as their own classes.

On Apple Silicon (MPS, no dedicated GPU) this is genuinely slow — expect
somewhere from 1-4+ hours for a few thousand images at 50-100 epochs,
not minutes. Start with a small --epochs run (e.g. 10) to sanity-check the
whole loop works before committing to a long run.

Usage:
    python scripts/train_detector.py --data data/datasets/detection/data.yaml

Then point configs/pipeline_config.yaml at the result:
    detection:
      model_path: <printed best.pt path>
      classes: [0, 1, 2, 3]  # match your data.yaml's class order
This switches src/detection/detector.py out of COCO fallback mode (single
pass, your model's own classes, no more ball/person merge-and-top-1 hack).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the dataset's data.yaml")
    parser.add_argument("--base-model", default="yolov8n.pt",
                         help="Starting checkpoint — yolov8n.pt (fastest) or yolov8s.pt (more accurate, slower)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960,
                         help="Larger than YOLO's usual 640 default — the ball is a small object, "
                              "more resolution helps it survive downsampling")
    parser.add_argument("--batch", type=int, default=8, help="Lower this if MPS runs out of memory")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--project", default="data/models/training_runs")
    parser.add_argument("--name", default="football_detector")
    parser.add_argument("--amp", action="store_true",
                         help="Enable mixed precision. Off by default on MPS — PyTorch's MPS autocast path is "
                              "less mature than CUDA's and can throw shape-mismatch errors deep in loss "
                              "computation (hit this firsthand: 'value tensor ... cannot be broadcast to "
                              "indexing result' inside ultralytics' TaskAlignedAssigner).")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.base_model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        amp=args.amp,
    )

    best_weights = model.trainer.best
    print(f"\nTraining done. Best weights: {best_weights}")
    print("Validation metrics:")
    metrics = model.val(data=args.data, device=args.device, project=args.project, name=f"{args.name}_val")
    print(f"  mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")
    print(f"\nUpdate configs/pipeline_config.yaml: detection.model_path -> {best_weights}, "
          f"detection.classes -> [list matching {args.data}'s class order]")


if __name__ == "__main__":
    main()
