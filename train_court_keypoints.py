#!/usr/bin/env python3
"""Train a court-keypoint model from the Roboflow dataset referenced by
github.com/HanaFEKI/AI_BasketBall_Analysis_v1 (workspace fyp-3bwmg,
project reloc2-den7l), since that repo's own trained weights are not
publicly accessible (confirmed dead Google Drive link, unanswered GitHub
issue asking the same question since September).

Round 3 of this training: round 1 (yolov8n-pose, 100 epochs, imgsz=640)
produced a homography where only 10/18 keypoints were usable as RANSAC
inliers -- degraded accuracy on the far/smaller keypoints (visually
confirmed: every detected keypoint's confidence topped out around 0.45,
none reached "confident" territory, worst in the crowded area near the
hoop). Round 2 (the source repo's exact config, yolov8x-pose/500 epochs)
was stopped by the user as impractically slow on Apple Silicon without a
dedicated GPU.

This round is a middle ground: yolov8s-pose.pt (small -- meaningfully
more capacity than nano, far cheaper than extra-large), 200 epochs, and
higher training resolution (imgsz=960 instead of 640) specifically to
help the model see small/far keypoints (like the ones near the hoop that
struggled most) more precisely. Also disables mosaic augmentation
(mosaic=0.0) -- for precise single-instance landmark localization, mosaic
(which stitches 4 training images together) can hurt more than it helps,
a known failure mode for exactly this "right general area, low
confidence" symptom. Batch size dropped to 8 (from 16) since higher
resolution uses meaningfully more memory per image.

Still a real time cost, just not a multi-day one -- keep the laptop
plugged in and let it run uninterrupted.

Usage:
    export ROBOFLOW_API_KEY="your key here"   # never hardcode it in this file
    python3 train_court_keypoints.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MODEL = "yolov8s-pose.pt"  # small -- middle ground between nano (round 1) and extra-large (round 2, too slow)
EPOCHS = 200
IMGSZ = 960                 # higher than round 1's 640, to help localize small/far keypoints more precisely
BATCH = 8                   # lower than round 2's 16 -- higher resolution needs more memory per image
MOSAIC = 0.0                # disabled -- can hurt precise single-instance landmark localization

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
        mosaic=MOSAIC,
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
