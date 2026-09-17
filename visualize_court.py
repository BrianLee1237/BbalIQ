import sys

import cv2

import courtiq_core as c

video_path = sys.argv[1] if len(sys.argv) > 1 else "game.mov"

capture = cv2.VideoCapture(video_path)
ok, frame = capture.read()
capture.release()
if not ok:
    raise SystemExit(f"Could not read a frame from {video_path}")

debug = c.court_quad_debug(frame)

print(f"color mask coverage:  {debug['color_coverage']:.1%} of frame")
print(f"court_top_row:        {debug['court_top_row']} (of {frame.shape[0]} total rows)")
print(f"morphed color mask:   {debug['morphed_coverage']:.1%} of frame")
print(f"accepted region:      {debug['largest_contour_fraction']:.1%} of frame")
print(f"line-refined:          {debug.get('line_refined')}")
if debug["reason"]:
    print(f"FAILED: {debug['reason']}")
else:
    print("Found a quad.")

for name in ("color_mask", "morphed_color_mask"):
    if debug[name] is not None:
        cv2.imwrite(f"debug_{name}.png", debug[name])
        print(f"Wrote debug_{name}.png")

out = frame.copy()
top_row = debug["court_top_row"]
if top_row is not None:
    cv2.line(out, (0, top_row), (frame.shape[1], top_row), (255, 0, 0), 3)
quad = debug["quad"]
if quad is not None:
    pts = [tuple(map(int, p)) for p in quad]
    for i in range(4):
        cv2.line(out, pts[i], pts[(i + 1) % 4], (0, 0, 255), 4)
    for label, p in zip(["TL", "TR", "BR", "BL"], pts):
        cv2.putText(out, label, p, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
cv2.imwrite("court_detection_debug.png", out)
print("Wrote court_detection_debug.png (blue line = detected crowd/floor row boundary)")
