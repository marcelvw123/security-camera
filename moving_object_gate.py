"""Experimental motion-backed object tracking; never sends alerts itself."""
import math

VEHICLES = {'car', 'truck', 'bus', 'motorcycle'}


def _intersection(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


class MovingObjectGate:
    """Require motion overlap, repeated observations, and vehicle displacement.

    Boxes are source-frame xyxy; motion regions are source-frame xywh. Tracks
    are per-camera and expire after a short gap; no future motion is borrowed.
    """
    def __init__(self, min_overlap=.05, min_displacement=.015, max_gap=4):
        self.min_overlap = min_overlap
        self.min_displacement = min_displacement
        self.max_gap = max_gap
        self.clear()

    def clear(self):
        self.tracks = []
        self.last_timestamp = None
        self.dimensions = None

    def observe(self, detections, regions, timestamp, width, height, flow_evidence=None):
        if width <= 0 or height <= 0 or not math.isfinite(timestamp):
            return []
        if self.last_timestamp is not None and (timestamp <= self.last_timestamp or self.dimensions != (width, height)):
            self.clear()
        self.last_timestamp, self.dimensions = timestamp, (width, height)
        self.tracks = [t for t in self.tracks if timestamp-t['time'] <= self.max_gap]
        motion = []
        for region in regions or []:
            try:
                x, y, w, h = map(float, region)
                if all(math.isfinite(v) for v in (x, y, w, h)) and w > 0 and h > 0:
                    motion.append((x, y, x+w, y+h))
            except (TypeError, ValueError):
                continue
        output, used = [], set()
        for detection_index, det in enumerate(detections):
            label = det.get('label')
            if label not in VEHICLES | {'person'}:
                continue
            try:
                box = tuple(map(float, det['box']))
                if len(box) != 4 or not all(math.isfinite(v) for v in box):
                    continue
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            area = (x2-x1)*(y2-y1)
            overlap = max((_intersection(box, region)/area for region in motion), default=0)
            category = 'vehicle' if label in VEHICLES else 'person'
            center = ((x1+x2)/2, (y1+y2)/2)
            choices = []
            for index, track in enumerate(self.tracks):
                if index in used or track['category'] != category:
                    continue
                old = track['box']
                old_area = (old[2]-old[0])*(old[3]-old[1])
                inter = _intersection(box, old)
                iou = inter/(area+old_area-inter) if area+old_area-inter else 0
                distance = math.hypot((center[0]-track['center'][0])/width, (center[1]-track['center'][1])/height)
                if iou >= .08 or distance <= .08:
                    choices.append((distance, index))
            if choices:
                index = min(choices)[1]
                track = self.tracks[index]
                track.update(box=box, center=center, time=timestamp)
            else:
                track = {'category': category, 'box': box, 'center': center, 'origin': center,
                         'time': timestamp, 'overlap_samples': 0}
                self.tracks.append(track)
                index = len(self.tracks)-1
            used.add(index)
            if overlap >= self.min_overlap:
                track['overlap_samples'] += 1
            travel = math.hypot((center[0]-track['origin'][0])/width, (center[1]-track['origin'][1])/height)
            flow = (flow_evidence[detection_index] if flow_evidence is not None and detection_index < len(flow_evidence) else None)
            flow_confirmed = flow is not None and math.isfinite(flow) and flow >= 1.0
            if overlap < self.min_overlap:
                reason = 'insufficient motion overlap'
            elif flow_confirmed:
                reason = 'moving object confirmed by image motion'
            elif track['overlap_samples'] < 2:
                reason = 'awaiting second motion-backed observation'
            elif category == 'vehicle' and travel < self.min_displacement:
                reason = 'vehicle displacement below threshold'
            else:
                reason = 'moving object confirmed'
            output.append({'label': label, 'box': list(box), 'qualified': reason.startswith('moving object confirmed'),
                           'reason': reason, 'motion_overlap': round(overlap, 3),
                           'motion_samples': track['overlap_samples'], 'displacement': round(travel, 4),
                           'flow_pixels_per_half_second': flow})
        return output
