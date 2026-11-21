"""Per-camera, in-memory alert decisions; recordings remain independent."""
import math

VEHICLES = {'car', 'truck', 'bus', 'motorcycle'}


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _overlap(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0


class AlertPolicy:
    """Track detections per model; conservatively suppress vehicles in the initial baseline."""
    def __init__(self, baseline_seconds=12, min_samples=2, max_gap=30,
                 person_quiet=20, vehicle_quiet=25, max_box_area=.32):
        self.baseline_seconds, self.min_samples, self.max_gap = baseline_seconds, min_samples, max_gap
        self.person_quiet, self.vehicle_quiet, self.max_box_area = person_quiet, vehicle_quiet, max_box_area
        self.sources = {}
        self.last_person_alert = None
        self.recent_vehicle_alerts = []

    def observe(self, source, detections, timestamp, width, height, vehicle_area=None):
        if width <= 0 or height <= 0:
            return [], ['invalid frame dimensions']
        state = self.sources.setdefault(source, {'start': timestamp, 'last': None, 'samples': 0, 'tracks': []})
        if state['last'] is not None and timestamp <= state['last']:
            return [], ['stale sample ignored']
        # A long gap means we cannot establish whether a stationary vehicle arrived meanwhile.
        if state['last'] is not None and timestamp-state['last'] > self.max_gap:
            state.update(start=timestamp, samples=0, tracks=[])
        state['last'] = timestamp
        state['samples'] += 1
        baseline = timestamp-state['start'] < self.baseline_seconds or state['samples'] <= self.min_samples
        state['tracks'] = [t for t in state['tracks'] if timestamp-t['last'] <= self.max_gap]
        self.recent_vehicle_alerts = [v for v in self.recent_vehicle_alerts if timestamp-v['time'] < self.vehicle_quiet]
        events, decisions, used = [], [], set()
        for det in detections:
            label, box = det.get('label'), det.get('box')
            if label not in VEHICLES | {'person'} or not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            try:
                x1,y1,x2,y2 = [float(v) for v in box]
                confidence = float(det.get('confidence', 0))
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in (x1,y1,x2,y2,confidence)) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                continue
            category = 'person' if label == 'person' else 'vehicle'
            if category == 'vehicle' and (x2-x1)*(y2-y1)/(width*height) > self.max_box_area:
                decisions.append('vehicle rejected: oversized box')
                continue
            if category == 'vehicle' and confidence < .25:
                decisions.append('vehicle rejected: low confidence')
                continue
            center = _center((x1,y1,x2,y2))
            if category == 'vehicle' and vehicle_area is not None:
                ax,ay,aw,ah = vehicle_area
                if not (ax <= center[0]/width <= ax+aw and ay <= center[1]/height <= ay+ah):
                    decisions.append('vehicle outside configured vehicle area')
                    continue
            choices = []
            for i, track in enumerate(state['tracks']):
                if i in used or track['category'] != category:
                    continue
                old = track['box']; oldcenter = _center(old)
                distance = math.hypot((center[0]-oldcenter[0])/width,(center[1]-oldcenter[1])/height)
                if _overlap((x1,y1,x2,y2), old) >= .08 or distance <= .08:
                    choices.append((distance,i))
            if choices:
                index = min(choices)[1]; used.add(index); track = state['tracks'][index]
                track.update(box=(x1,y1,x2,y2), last=timestamp, count=track['count']+1)
            else:
                track={'category':category,'box':(x1,y1,x2,y2),'origin':center,'last':timestamp,
                       'count':1,'baseline':baseline,'alerted':False}
                state['tracks'].append(track); used.add(len(state['tracks'])-1)
            if category == 'person':
                if track['count'] >= 2 and not track['alerted'] and (self.last_person_alert is None or timestamp-self.last_person_alert >= self.person_quiet):
                    events.append({'label':'person','reason':'person confirmed in two samples'})
                    self.last_person_alert=timestamp
                    track['alerted'] = True
                else:
                    decisions.append('person pending confirmation or in cooldown')
                continue
            travel = math.hypot((center[0]-track['origin'][0])/width,(center[1]-track['origin'][1])/height)
            if track['alerted']:
                decisions.append('vehicle already alerted')
            elif track['count'] < 2:
                decisions.append('vehicle pending confirmation')
            elif track['baseline'] and travel < .035:
                decisions.append('stationary vehicle present during baseline')
            else:
                duplicate = any(math.hypot((center[0]-v['center'][0])/width,(center[1]-v['center'][1])/height) < .15 for v in self.recent_vehicle_alerts)
                if duplicate:
                    decisions.append('vehicle grouped with recent alert')
                    track['alerted'] = True
                else:
                    reason = 'vehicle moved' if travel >= .035 else 'new vehicle confirmed'
                    events.append({'label':label,'reason':reason})
                    self.recent_vehicle_alerts.append({'center':center,'time':timestamp})
                    track['alerted'] = True
        if baseline:
            decisions.append('vehicle baseline in progress')
        return events, decisions
