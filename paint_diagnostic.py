"""Shows what detect_painted_regions() actually finds, and which region gets
chosen as the lane.

The calibration log can say a "key" was found with good rectangularity while
that region is really something else entirely -- the lane merged with a
same-coloured out-of-bounds apron, say. This draws every candidate region so
that's visible instead of inferred.

Usage: python3 paint_diagnostic.py <video> [frame_number]
Output: paint_regions.png
"""
import sys

import cv2
import numpy as np

import courtiq_core as c

video_path = sys.argv[1] if len(sys.argv) > 1 else "game.mov"
frame_number = int(sys.argv[2]) if len(sys.argv) > 2 else 0

capture = cv2.VideoCapture(video_path)
if frame_number > 0:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ok, frame = capture.read()
capture.release()
if not ok:
    raise SystemExit(f"Could not read frame {frame_number}")

height, width = frame.shape[:2]
debug = c.court_quad_debug(frame)
floor_mask = debug.get("morphed_color_mask")
print(f"floor mask: {'present' if floor_mask is not None else 'MISSING'} "
      f"(reason: {debug.get('reason')})")
print(f"floor coverage: {(floor_mask > 0).mean():.1%} of frame" if floor_mask is not None else "")

overlay = frame.copy()

if floor_mask is not None:
    floor_contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(cv2.contourArea(fc) for fc in floor_contours)
    significant = [fc for fc in floor_contours
                   if cv2.contourArea(fc) >= largest * c.FLOOR_PIECE_MIN_FRACTION]
    hull = cv2.convexHull(np.vstack(significant))
    hull_area = cv2.contourArea(hull)
    print(f"floor pieces kept: {len(significant)}, hull area: {hull_area:.0f} px "
          f"({hull_area / (height * width):.1%} of frame)")
    cv2.polylines(overlay, [hull], True, (255, 255, 0), 3)
    cv2.putText(overlay, "FLOOR HULL", tuple(hull[0][0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
else:
    hull_area = None

regions = c.detect_painted_regions(frame, floor_mask)
print(f"\npainted regions found: {len(regions)}")
for i, contour in enumerate(regions):
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        continue
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    area = cv2.contourArea(contour)
    quad, rectangularity = c._key_quad_from_contour(contour)
    share = f", {area / hull_area:.0%} of floor" if hull_area else ""
    print(f"  region {i}: area {area:8.0f} px{share}, centroid ({cx:.0f},{cy:.0f}), "
          f"rectangularity {rectangularity:.2f}")
    cv2.drawContours(overlay, [contour], -1, (0, 0, 255), 3)
    cv2.putText(overlay, f"#{i} {area / hull_area:.0%} of floor" if hull_area else f"#{i}",
                (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

hoop_candidates = c.detect_hoop_candidates(video_path)
if hoop_candidates:
    hoop_px = (hoop_candidates[0][0], hoop_candidates[0][1])
    print(f"\ntop hoop candidate: ({hoop_px[0]:.0f}, {hoop_px[1]:.0f})")
    for rank, (hx, hy, score) in enumerate(hoop_candidates[:4]):
        chosen = c.detect_key_with_score(frame, (hx, hy))
        if chosen is None:
            print(f"  candidate {rank} ({hx:.0f},{hy:.0f}): no region chosen")
            continue
        quad, rect = chosen
        qcx = sum(p[0] for p in quad) / 4
        qcy = sum(p[1] for p in quad) / 4
        print(f"  candidate {rank} ({hx:.0f},{hy:.0f}) -> region centred "
              f"({qcx:.0f},{qcy:.0f}), rectangularity {rect:.2f}")
        if rank == 0:
            pts = np.array(quad, dtype=np.int32)
            cv2.polylines(overlay, [pts], True, (0, 255, 0), 4)
            cv2.putText(overlay, "CHOSEN AS LANE", (int(qcx), int(qcy) + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imwrite("paint_regions.png", overlay)
print("\nWrote paint_regions.png -- yellow = floor hull, red = painted regions, "
      "green = what got chosen as the lane")
