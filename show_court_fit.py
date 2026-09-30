"""Draws the fitted court model over the footage, so the calibration can be
judged by eye rather than by numbers.

Overlays the model's markings -- boundary, lane, free-throw line, arc,
centre circle -- through the computed homography. If those land on the real
painted lines, the calibration is right; if they don't, it isn't. Unlike a
plain court outline, which traces whatever region was detected and therefore
looks plausible even when the mapping is wrong, this can't hide an error.

Usage: python3 show_court_fit.py <video>
Output: court_fit_overlay.png
"""
import sys

import cv2
import numpy as np

import court_fit
import courtiq_core as c


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "game.mov"

    median = court_fit.median_frame(video_path)
    if median is None:
        raise SystemExit(f"Could not read frames from {video_path}")
    cv2.imwrite("court_fit_median.png", median)
    print("Wrote court_fit_median.png (the clip with players removed)")

    floor_mask = c.court_quad_debug(median).get("morphed_color_mask")
    if floor_mask is None:
        raise SystemExit("No floor detected.")
    floor_region = c.floor_hull_region(floor_mask)
    if floor_region is None:
        raise SystemExit("No floor region.")

    line_pixels = court_fit.detect_line_pixels(median, floor_region)
    print(f"Court line pixels found: {cv2.countNonZero(line_pixels)}")
    cv2.imwrite("court_fit_lines.png", line_pixels)
    print("Wrote court_fit_lines.png (the markings the fit works from)")

    fitted = court_fit.fit_court(median, floor_region, floor_mask)
    if fitted is None:
        raise SystemExit("No court fit -- see the message above for why.")
    homography, info = fitted
    model = info["model"]
    court_to_image = np.linalg.inv(homography)

    capture = cv2.VideoCapture(video_path)
    ok, frame = capture.read()
    capture.release()
    overlay = frame if ok else median.copy()

    def to_px(pt):
        src = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        return tuple(map(int, cv2.perspectiveTransform(src, court_to_image)[0][0]))

    for polyline in model.polylines():
        pts = [to_px(p) for p in polyline]
        for a, b in zip(pts, pts[1:]):
            cv2.line(overlay, a, b, (0, 255, 255), 3)

    hoop_px = to_px(model.hoop)
    cv2.circle(overlay, hoop_px, 12, (0, 0, 255), 3)
    cv2.putText(overlay, "HOOP", (hoop_px[0] + 16, hoop_px[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(overlay, f"{model.name} court, {info['cost']:.1f}px marking error",
                (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.imwrite("court_fit_overlay.png", overlay)
    print(f"\nFitted {model.name} dimensions, {info['cost']:.1f}px mean marking error.")
    print("Wrote court_fit_overlay.png -- the yellow markings should sit on the real "
          "painted lines, and the red circle on the rim.")


if __name__ == "__main__":
    main()
