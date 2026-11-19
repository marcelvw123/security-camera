#!/usr/bin/env python3
"""Evaluate a YOLO vehicle model on a crop of sampled frames in a local video."""
import argparse
import json
import math
from pathlib import Path
import cv2
from ultralytics import YOLO

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('video', type=Path)
p.add_argument('--model', required=True)
p.add_argument('--crop', nargs=4, type=float, metavar=('X', 'Y', 'W', 'H'), required=True,
               help='normalized crop coordinates')
p.add_argument('--interval', type=float, default=1)
p.add_argument('--confidence', type=float, default=.1)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
x, y, w, h = a.crop
if not all(math.isfinite(v) for v in a.crop) or x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > 1 or y+h > 1 or a.interval <= 0 or not 0 < a.confidence < 1:
    p.error('invalid crop, interval, or confidence')
cap = cv2.VideoCapture(str(a.video))
if not cap.isOpened(): p.error('cannot open video')
fps = cap.get(cv2.CAP_PROP_FPS)
if not math.isfinite(fps) or fps <= 0: p.error('invalid video frame rate')
model = YOLO(a.model)
results, index, next_time = [], 0, 0
try:
    while True:
        ok, frame = cap.read()
        if not ok: break
        timestamp = index / fps
        index += 1
        if timestamp < next_time: continue
        next_time = timestamp + a.interval
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = int(x*width), int(y*height), math.ceil((x+w)*width), math.ceil((y+h)*height)
        pred = model.predict(frame[y1:y2, x1:x2], classes=[2, 3, 5, 7], conf=a.confidence,
                             imgsz=640, device='cpu', verbose=False)[0]
        boxes = [{'label': model.names[int(cls)], 'confidence': round(float(conf), 3),
                  'box': [round(float(box[0])+x1, 1), round(float(box[1])+y1, 1),
                          round(float(box[2])+x1, 1), round(float(box[3])+y1, 1)]}
                 for box, conf, cls in zip(pred.boxes.xyxy, pred.boxes.conf, pred.boxes.cls)]
        results.append({'seconds': round(timestamp, 3), 'boxes': boxes})
finally:
    cap.release()
a.output.write_text(json.dumps({'video': a.video.name, 'crop': a.crop, 'interval': a.interval,
                                'confidence': a.confidence, 'samples': results}, indent=2) + '\n')
print(f"{sum(bool(s['boxes']) for s in results)}/{len(results)} sampled frames have vehicle detections")
