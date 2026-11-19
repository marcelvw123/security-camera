#!/usr/bin/env python3
"""Replay an independent recording through the application's motion detector."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from security_camera_core import detect_motion, motion_sensitivity_config


def evaluate(path, interval, sensitivity, warmup):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise ValueError('Video does not have a valid frame rate')
    detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    settings = motion_sensitivity_config(sensitivity)
    samples, next_sample, index, cpu_seconds = [], 0.0, 0, 0.0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = index / fps
            index += 1
            if timestamp + 1e-9 < next_sample:
                continue
            next_sample = timestamp + interval
            cpu_started = time.process_time()
            boxes = detect_motion(frame, detector, settings)
            cpu_seconds += time.process_time() - cpu_started
            samples.append({'seconds': round(timestamp, 3), 'warmup': timestamp < warmup,
                            'boxes': [list(map(int, box)) for box in boxes]})
    finally:
        capture.release()
    return {'video': Path(path).name, 'fps': fps, 'duration_seconds': round(index / fps, 3),
            'interval_seconds': interval, 'sensitivity': settings.level, 'warmup_seconds': warmup,
            'detector_cpu_seconds': round(cpu_seconds, 3),
            'elapsed_seconds': round(time.perf_counter() - started, 3), 'samples': samples}



def score_events(result, annotations):
    """Score labelled events using temporal and rectangular spatial overlap.

    A hit is only a possible match: another moving object in the same region
    can produce it. Unsampled events are reported separately from misses.
    """
    if not isinstance(annotations, dict) or not isinstance(annotations.get('events'), list):
        raise ValueError('annotations must contain an events list')
    duration = result['duration_seconds']
    samples = result['samples']
    scored = []
    for index, event in enumerate(annotations['events']):
        if not isinstance(event, dict):
            raise ValueError(f'event {index + 1} must be an object')
        start, end, roi = event.get('start'), event.get('end'), event.get('roi')
        numeric = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        if not numeric(start) or not numeric(end) or not 0 <= start < end <= duration:
            raise ValueError(f'event {index + 1} needs 0 <= start < end <= video duration')
        if (not isinstance(roi, list) or len(roi) != 4 or not all(numeric(v) for v in roi)
                or roi[0] < 0 or roi[1] < 0 or roi[2] <= 0 or roi[3] <= 0):
            raise ValueError(f'event {index + 1} needs roi [x, y, width, height]')
        eligible = [sample for sample in samples if not sample['warmup'] and start <= sample['seconds'] < end]
        def overlaps(box):
            return (box[0] < roi[0] + roi[2] and roi[0] < box[0] + box[2]
                    and box[1] < roi[1] + roi[3] and roi[1] < box[1] + box[3])
        hits = [sample['seconds'] for sample in eligible if any(overlaps(box) for box in sample['boxes'])]
        scored.append({'event': index + 1, 'label': event.get('label', ''), 'start': start, 'end': end,
                       'status': 'hit' if hits else 'miss' if eligible else 'unsampled',
                       'eligible_samples': len(eligible),
                       'first_hit_seconds': hits[0] if hits else None,
                       'latency_seconds': round(hits[0] - start, 3) if hits else None})
    counts = {status: sum(event['status'] == status for event in scored)
              for status in ('hit', 'miss', 'unsampled')}
    return {'events': scored, 'counts': counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--interval', type=float, default=1.0, help='Sampling interval in seconds')
    parser.add_argument('--sensitivity', type=int, default=5, choices=range(1, 11))
    parser.add_argument('--warmup', type=float, default=5.0, help='Mark initial background learning period')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, help='JSON file containing reviewed event intervals and regions')
    args = parser.parse_args()
    if args.interval <= 0 or args.warmup < 0:
        parser.error('interval must be positive and warmup must not be negative')
    try:
        result = evaluate(args.video, args.interval, args.sensitivity, args.warmup)
        if args.annotations:
            result['event_score'] = score_events(result, json.loads(args.annotations.read_text()))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    active = sum(bool(sample['boxes']) and not sample['warmup'] for sample in result['samples'])
    print(f"{len(result['samples'])} samples, {active} with motion after warmup; saved {args.output}")


if __name__ == '__main__':
    main()
