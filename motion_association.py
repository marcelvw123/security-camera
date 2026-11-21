"""Associate sampled motion regions with object detections for diagnostics only."""
import math
from collections import deque


class MotionAssociation:
    """Keep recent per-camera motion samples in source-frame pixel coordinates."""
    def __init__(self, history_seconds=12, window_seconds=2, minimum_samples=2,
                 min_object_overlap=.05, max_samples=256):
        self.history_seconds = history_seconds
        self.window_seconds = window_seconds
        self.minimum_samples = minimum_samples
        self.min_object_overlap = min_object_overlap
        self.samples = deque(maxlen=max_samples)

    def clear(self):
        self.samples.clear()

    def add(self, timestamp, boxes, width, height):
        if not math.isfinite(timestamp) or width <= 0 or height <= 0:
            return
        if self.samples and (timestamp <= self.samples[-1][0] or (width, height) != self.samples[-1][2:]):
            self.clear()
        valid = []
        for box in boxes:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            try:
                x, y, w, h = (float(v) for v in box)
            except (ValueError, TypeError):
                continue
            if all(math.isfinite(v) for v in (x,y,w,h)) and w > 0 and h > 0:
                valid.append((max(0, x), max(0, y), min(width, x+w), min(height, y+h)))
        self.samples.append((timestamp, valid, width, height))
        while self.samples and timestamp-self.samples[0][0] > self.history_seconds:
            self.samples.popleft()

    def check(self, detections, timestamp, width, height):
        """Use only samples at or before the object frame; never borrow future motion."""
        samples = [(t, boxes) for t, boxes, w, h in self.samples
                   if (w,h) == (width,height) and 0 <= timestamp-t <= self.window_seconds]
        results = []
        for det in detections:
            box = det.get('box')
            if not isinstance(box, (tuple,list)) or len(box) != 4:
                continue
            try:
                x1,y1,x2,y2 = [float(v) for v in box]
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in (x1,y1,x2,y2)) or x2 <= x1 or y2 <= y1:
                continue
            object_area = (x2-x1)*(y2-y1)
            matches = 0
            for _t, regions in samples:
                if any(max(0,min(x2,mx2)-max(x1,mx1))*max(0,min(y2,my2)-max(y1,my1))/object_area >= self.min_object_overlap
                       for mx1,my1,mx2,my2 in regions):
                    matches += 1
            status = ('unknown' if not samples else
                      'associated' if matches >= self.minimum_samples else
                      'possible' if matches else 'no overlap')
            results.append({'label': str(det.get('label') or 'object'), 'status': status,
                            'motion_samples': len(samples), 'matching_samples': matches})
        return results
