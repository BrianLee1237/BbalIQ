#!/usr/bin/env python3
"""Train a court-keypoint model from the Roboflow dataset referenced by
github.com/HanaFEKI/AI_BasketBall_Analysis_v1 (workspace fyp-3bwmg,
project reloc2-den7l), since that repo's own trained weights are not
publicly accessible (confirmed dead Google Drive link, unanswered GitHub
issue asking the same question since September).

UPDATED to match that repo's exact training config (yolov8x-pose.pt,
500 epochs, batch 16) after a first pass with a scaled-down config
(yolov8n-pose, 100 epochs) produced a homography where only 10/18
keypoints were usable as RANSAC inliers -- degraded accuracy on the
far/smaller keypoints, consistent with that scale-down trade-off. This
exact config has NOT been scaled down for hardware without a dedicated
NVIDIA GPU: on Apple Silicon (PyTorch's MPS backend, not CUDA) this can
realistically take multiple days. Keep the laptop plugged in and
uninterrupted for the whole run.

Usage:
    export ROBOFLOW_API_KEY="your key here"   # never hardcode it in this file
    python3 train_court_keypoints.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MODEL = "yolov8x-pose.pt"  # extra-large -- matches the source repo's exact config
EPOCHS = 500                # matches the source repo's exact config
IMGSZ = 640
BATCH = 16                  # matches the source repo's exact config
# WARNING: this is the repo's literal training config, not scaled down for
# hardware without a dedicated NVIDIA GPU. On a MacBook Air (Apple
# Silicon MPS backend, not CUDA) this can realistically take MULTIPLE
# DAYS, not hours -- the nano/100-epoch run that produced the first
# court_keypoints.pt (10/18 keypoints usable as RANSAC inliers, degraded
# far-keypoint accuracy) was far cheaper than this. Keep the laptop
# plugged in and uninterrupted for the whole run. If this isn't
# practical, yolov8s-pose.pt with EPOCHS=250 is a meaningfully cheaper
# middle ground.

ROOT = Path(__file__).resolve().parent
OUT_MODEL = ROOT / "models" / "court_keypoints.pt"


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Set ROBOFLOW_API_KEY as an environment variable first:")
        print('  export ROBOFLOW_API_KEY="your key here"')
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("This needs the 'roboflow' package. Install it with: pip install roboflow")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("This needs 'ultralytics'. Install it with: pip install ultralytics")
        sys.exit(1)

    print("Downloading the court-keypoint dataset from Roboflow (fyp-3bwmg/reloc2-den7l v1)...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("fyp-3bwmg").project("reloc2-den7l")
    dataset = project.version(1).download("yolov8")
    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        print(f"Expected data.yaml at {data_yaml} but it's not there -- dataset download may have failed.")
        sys.exit(1)
    print(f"Dataset downloaded to {dataset.location}")

    print(f"Training {MODEL} for {EPOCHS} epochs (this will take a while -- keep the laptop plugged in)...")
    model = YOLO(MODEL)
    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        task="pose",
        plots=True,
    )

    # Ultralytics saves the best checkpoint under runs/pose/train*/weights/best.pt
    run_dir = Path(results.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"Training finished but couldn't find {best_pt} -- check {run_dir} manually.")
        sys.exit(1)

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_MODEL.write_bytes(best_pt.read_bytes())
    print(f"Done. Copied trained weights to {OUT_MODEL}")
    print("The Roboflow API key is no longer needed from this point forward --")
    print(f"courtiq_core.py loads {OUT_MODEL} as a plain local file.")
    print()
    print("IMPORTANT: this model has NOT been validated against your real footage yet.")
    print("Run this next and check the results actually look right:")
    print(f"  python3 -c \"from ultralytics import YOLO; m = YOLO('{OUT_MODEL}'); print(m.task, m.names)\"")
    print(f"  python3 courtiq_core.py <your_video> --keypoint-model {OUT_MODEL}")


if __name__ == "__main__":
    main()
