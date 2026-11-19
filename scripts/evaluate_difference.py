#!/usr/bin/env python3
"""Replay candidate frame-difference motion on a continuous video."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from motion_difference import FrameDifferenceMotion
from scripts.evaluate_motion import score_events


def evaluate(path, interval=.5, min_area=500, warmup=5, pixel_threshold=25, exclusions=None):
    if not math.isfinite(interval) or interval <= 0 or not math.isfinite(warmup) or warmup < 0:
        raise ValueError('interval must be positive and warmup nonnegative')
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError('cannot open video')
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError('video does not have a valid frame rate')
    detector = FrameDifferenceMotion(min_area=min_area, pixel_threshold=pixel_threshold, exclusions=exclusions)
    samples, next_time, index = [], 0.0, 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = index / fps
            index += 1
            if timestamp + 1e-9 < next_time:
                continue
            next_time = timestamp + interval
            boxes = detector.observe(frame, timestamp)
            samples.append({'seconds':round(timestamp, 3), 'warmup':timestamp < warmup, 'boxes':boxes})
    finally:
        capture.release()
    return {'video':Path(path).name, 'fps':fps, 'duration_seconds':round(index / fps, 3),
            'interval_seconds':interval, 'min_area':min_area, 'pixel_threshold':pixel_threshold, 'warmup_seconds':warmup,
            'exclusions': exclusions or [], 'elapsed_seconds':round(time.perf_counter()-started, 3), 'samples':samples}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path)
    parser.add_argument('--interval',type=float,default=.5)
    parser.add_argument('--min-area',type=int,default=500)
    parser.add_argument('--pixel-threshold',type=int,default=25)
    parser.add_argument('--warmup',type=float,default=5)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--annotations',type=Path)
    parser.add_argument('--exclusions-file', type=Path, help='saved per-camera motion_exclusions.json')
    parser.add_argument('--camera', help='camera source ID, for example manual:701')
    args=parser.parse_args()
    try:
        if bool(args.exclusions_file) != bool(args.camera):
            raise ValueError('--exclusions-file and --camera must be provided together')
        exclusions = []
        if args.exclusions_file:
            saved = json.loads(args.exclusions_file.read_text())
            if not isinstance(saved, dict) or args.camera not in saved:
                raise ValueError('camera not present in exclusions file')
            exclusions = saved[args.camera]
            FrameDifferenceMotion.validate_exclusions(exclusions)
        result=evaluate(args.video,args.interval,args.min_area,args.warmup,args.pixel_threshold,exclusions)
        if args.annotations:
            result['event_score']=score_events(result,json.loads(args.annotations.read_text()))
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    except (ValueError,OSError) as error:
        parser.error(str(error))
    eligible=[s for s in result['samples'] if not s['warmup']]
    print(f"{sum(bool(s['boxes']) for s in eligible)}/{len(eligible)} samples with candidate motion; saved {args.output}")


if __name__=='__main__':
    main()
