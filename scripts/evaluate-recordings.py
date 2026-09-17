"""Offline ONNX detector audit. Outputs local traces; never uploads recordings.

This uses OpenCV/ONNX Runtime, not the browser decoder. Replay the observations
through replay-recordings.ts to test the production TypeScript shot tracker.
Dependencies for this optional audit: opencv-python-headless, onnxruntime.
"""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
parser.add_argument("--output", type=Path, default=Path("work/recording-audit"))
parser.add_argument("--indices", default="", help="Optional comma-separated clip indices")
args = parser.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
args.output.mkdir(parents=True, exist_ok=True)
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
session = ort.InferenceSession("public/models/attalla-basketball-yolov8n.onnx", opts,
                               providers=["CPUExecutionProvider"])

def detect(image):
    h, w = image.shape[:2]
    gain = min(640 / w, 640 / h)
    rw, rh = round(w * gain), round(h * gain)
    px, py = (640-rw)//2, (640-rh)//2
    padded = np.full((640, 640, 3), 114, np.uint8)
    padded[py:py+rh, px:px+rw] = cv2.resize(image, (rw, rh))
    tensor = padded[:, :, ::-1].transpose(2, 0, 1).copy()[None].astype(np.float32) / 255
    result = session.run(None, {session.get_inputs()[0].name: tensor})[0][0].T
    objects = []
    for cls in [0, 1]:
        rows = result[(result[:, 4:].argmax(axis=1) == cls) &
                      (result[:, 4+cls] >= (0.12 if cls == 0 else 0.2))]
        boxes, scores = [], []
        for row in rows:
            x, y, bw, bh = row[:4]
            left = max(0., float((x-bw/2-px)/gain))
            top = max(0., float((y-bh/2-py)/gain))
            right = min(float(w), float((x+bw/2-px)/gain))
            bottom = min(float(h), float((y+bh/2-py)/gain))
            if right-left < 2 or bottom-top < 2:
                continue
            boxes.append([left, top, right-left, bottom-top])
            scores.append(float(row[4+cls]))
        keep = cv2.dnn.NMSBoxes(boxes, scores, 0., 0.7)
        for i in np.array(keep).flatten():
            x, y, bw, bh = boxes[i]
            objects.append(dict(classId=cls, label="ball" if cls == 0 else "hoop",
                                left=x, top=y, right=x+bw, bottom=y+bh, confidence=scores[i]))
    return objects

def calibration(image, hoop):
    h, w = image.shape[:2]
    hw, hh = hoop['right']-hoop['left'], hoop['bottom']-hoop['top']
    rw = max(w*.035, hw*.64)
    rh = max(h*.012, min(rw/4.2, hh*.3))
    cx, cy = (hoop['left']+hoop['right'])/2, hoop['top']+hh*.36
    best = (0, 0, 0, -1)
    for y in range(max(0, int(hoop['top'])), min(h-1, int(np.ceil(hoop['top']+hh*.72)))+1):
        xs = []
        for x in range(max(0, int(hoop['left'])), min(w-1, int(np.ceil(hoop['right'])))+1):
            b, g, r = map(int, image[y, x])
            if r >= 85 and r-min(r,g,b) >= 34 and r >= g*1.08 and r >= b*1.22 and max(r,g,b)-min(r,g,b) >= 38:
                xs.append(x)
        if xs:
            score = len(xs)+(xs[-1]-xs[0])*.45
            if score > best[0]:
                best = (score, xs[0], xs[-1], y)
    score, left, right, row = best
    if row >= 0 and right-left >= hw*.28 and score >= 8:
        rw = max(w*.035, right-left+2*max(1, (right-left)*.05))
        rh = max(h*.012, rw/4.2)
        cx, cy = (left+right)/2, row
    return dict(x=max(0,min(1-rw/w,(cx-rw/2)/w)),
                y=max(0,min(1-rh/h,(cy-rh/2)/h)), width=rw/w, height=rh/h)

paths = [p for p in args.directory.glob('*.mp4')
         if any(s in p.name.lower() for s in ['basketball', 'free throw', 'hoopers', 'threes'])]
for index, path in enumerate(paths):
    if args.indices and index not in [int(i) for i in args.indices.split(',')]:
        continue
    out = args.output / f"clip-{index}.json"
    if out.exists():
        print(f"cached {index}: {path.name}", flush=True)
        continue
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sw, sh = int(cap.get(3)), int(cap.get(4))
    scale = min(1,640/sw,640/sh)
    aw, ah = round(sw*scale), round(sh*scale)
    print(f"START {index}: {path.name} {sw}x{sh} {count/fps:.2f}s", flush=True)
    rim = None
    for t in [min(1, count/fps*.08), .25, 2., 3.]:
        cap.set(cv2.CAP_PROP_POS_MSEC, t*1000)
        ok, source = cap.read()
        if not ok: continue
        small = cv2.resize(source,(aw,ah))
        hoops = [o for o in detect(small) if o['label']=='hoop']
        if hoops:
            hoop = max(hoops, key=lambda o:o['confidence'])
            rim = calibration(small,hoop)
            preview = source.copy()
            cv2.rectangle(preview,(int(rim['x']*sw),int(rim['y']*sh)),
                          (int((rim['x']+rim['width'])*sw),int((rim['y']+rim['height'])*sh)),(0,255,255),3)
            cv2.imwrite(str(args.output/f"clip-{index}-rim.jpg"), cv2.resize(preview,(960,round(sh*960/sw))))
            break
    if rim is None:
        print(f"NO HOOP {index}", flush=True)
        cap.release()
        continue
    cx, cy = (rim['x']+rim['width']/2)*aw, (rim['y']+rim['height']/2)*ah
    halfw = max(aw*.13,min(rim['width']*aw*4,aw*.2))
    halfh = max(ah*.2,min(rim['width']*aw*4,ah*.25))
    left, top = max(0,int(cx-halfw)), max(0,int(cy-halfh))
    right, bottom = min(aw,int(np.ceil(cx+halfw))),min(ah,int(np.ceil(cy+halfh)))
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    frames = []
    for n in range(min(count,int(fps*300))):
        ok, source = cap.read()
        if not ok: break
        if fps > 31 and n % max(1,round(fps/30)): continue
        small = cv2.resize(source,(aw,ah))
        full = detect(small)
        sx, sy = sw/aw, sh/ah
        native = source[round(top*sy):round(bottom*sy),round(left*sx):round(right*sx)]
        nh,nw = native.shape[:2]
        gain = min(1,640/max(nw,nh))
        native = cv2.resize(native,(max(1,round(nw*gain)),max(1,round(nh*gain))))
        focused = detect(native)
        for o in focused:
            for field in ['left','right']: o[field] = left+o[field]*(right-left)/native.shape[1]
            for field in ['top','bottom']: o[field] = top+o[field]*(bottom-top)/native.shape[0]
        frames.append(dict(atMs=round(n/fps*1000),full=full,focused=focused))
        if n % max(1,int(fps*5)) == 0: print(f"  {index} {n/fps:.1f}s",flush=True)
    cap.release()
    out.write_text(json.dumps(dict(source=str(path), fps=fps, width=aw,height=ah,rim=rim,frames=frames)),encoding='utf-8')
    print(f"DONE {index}: {len(frames)} frames",flush=True)
