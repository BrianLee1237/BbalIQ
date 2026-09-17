#!/usr/bin/env python3
"""Download a basketball-trained ball detection model for CourtIQ.

COCO's generic "sports ball" class (the default in courtiq_core.py) was
never trained on basketball footage specifically and performs poorly on
it. This downloads a basketball-specific YOLO model instead:

  Source: github.com/abdullahtarek/basketball_analysis (MIT licensed)
  Weights host: Google Drive (linked from that repo's README)

This script could not be run/validated against CourtIQ's own test footage
from the environment it was written in -- there was no network path to
Google Drive and no test video available there. Run it yourself, then
validate the actual detection rate on your own footage with:

    python3 courtiq_core.py <your_video> --ball-model models/basketball_ball.pt

and check the printed "Ball detection rate" line before trusting any
possession/decision output built on it (courtiq_core.py also refuses to
grade ball-dependent decisions below a 15% detection rate -- see
MIN_RELIABLE_BALL_DETECTION_RATE).
"""
from __future__ import annotations

import sys
from pathlib import Path

GOOGLE_DRIVE_FILE_ID = "1KejdrcEnto2AKjdgdo1U1syr5gODp6EL"
OUT_PATH = Path(__file__).resolve().parent / "models" / "basketball_ball.pt"


def main():
    try:
        import gdown
    except ImportError:
        print("This needs the 'gdown' package to fetch the weights from Google Drive.")
        print("Install it with: pip install gdown")
        sys.exit(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    print(f"Downloading basketball ball-detection weights to {OUT_PATH} ...")
    gdown.download(url, str(OUT_PATH), quiet=False)

    if not OUT_PATH.exists():
        print("Download did not produce a file. Check your network connection and try again,")
        print(f"or download it manually from https://drive.google.com/file/d/{GOOGLE_DRIVE_FILE_ID}/view")
        print(f"and save it to {OUT_PATH}")
        sys.exit(1)

    print(f"Done. Set COURTIQ_BALL_MODEL={OUT_PATH} (or pass --ball-model {OUT_PATH})")
    print("Then validate on your own footage before trusting it:")
    print(f"  python3 courtiq_core.py <your_video> --ball-model {OUT_PATH}")


if __name__ == "__main__":
    main()
