"""Experimental confirmation of weak vehicle detections using motion tracks."""
import math
from alert_policy import VEHICLES


def _intersection(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


class WeakMotionVehicle:
    """Two independently sampled, spatially matched motion-backed candidates.

    This is an experimental diagnostic, not calibrated for production alerts.
    """
    def __init__(self, max_gap=2.0, min_flow=.5, min_travel=.01, min_overlap=.05):
        self.max_gap, self.min_flow = max_gap, min_flow
        self.min_travel, self.min_overlap = min_travel, min_overlap
        self.clear()

    def clear(self):
        self.tracks = []
        self.last_time = None
        self.dimensions = None

    def observe(self, detections, regions, flows, timestamp, width, height):
        if width <= 0 or height <= 0 or not math.isfinite(timestamp):
            return []
        if self.last_time is not None and (timestamp <= self.last_time or (width, height) != self.dimensions):
            self.clear()
        self.last_time, self.dimensions = timestamp, (width, height)
        self.tracks = [track for track in self.tracks if timestamp-track['time'] <= self.max_gap]
        motion = []
        for region in regions or []:
            try:
                x,y,w,h = map(float,region)
                if all(math.isfinite(v) for v in (x,y,w,h)) and w > 0 and h > 0:
                    motion.append((x,y,x+w,y+h))
            except (ValueError, TypeError):
                continue
        candidates = []
        for det,flow in zip(detections, flows or []):
            try:
                box = tuple(map(float,det['box']))
                confidence = float(det.get('confidence',0))
                if len(box)!=4 or not all(math.isfinite(v) for v in (*box,confidence,float(flow))):
                    continue
                x1,y1,x2,y2 = box
                if det.get('label') not in VEHICLES or confidence < .05 or float(flow) < self.min_flow or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    continue
            except (TypeError, ValueError, KeyError):
                continue
            area=(x2-x1)*(y2-y1)
            if area/(width*height) > .32 or max((_intersection(box,r)/area for r in motion),default=0) < self.min_overlap:
                continue
            candidates.append((confidence,box,det['label'],float(flow)))
        candidates.sort(reverse=True)
        # Avoid counting alternate vehicle labels on the same image as separate tracks.
        unique=[]
        for candidate in candidates:
            box=candidate[1];area=(box[2]-box[0])*(box[3]-box[1])
            if any(_intersection(box,old[1])/max(1,min(area,(old[1][2]-old[1][0])*(old[1][3]-old[1][1]))) >= .5 for old in unique):
                continue
            unique.append(candidate)
        used=set();result=[]
        for confidence,box,label,flow in unique:
            center=((box[0]+box[2])/2,(box[1]+box[3])/2)
            choices=[]
            for i,track in enumerate(self.tracks):
                if i in used:continue
                distance=math.hypot((center[0]-track['center'][0])/width,(center[1]-track['center'][1])/height)
                if distance <= .08:
                    choices.append((distance,i))
            if choices:
                index=min(choices)[1];track=self.tracks[index]
                track.update(center=center,time=timestamp,count=track['count']+1)
            else:
                track={'center':center,'origin':center,'time':timestamp,'count':1}
                self.tracks.append(track);index=len(self.tracks)-1
            used.add(index)
            travel=math.hypot((center[0]-track['origin'][0])/width,(center[1]-track['origin'][1])/height)
            result.append({'label':label,'box':list(box),'confidence':confidence,'flow':round(flow,3),
                           'samples':track['count'],'travel':round(travel,4),
                           'qualified':track['count']>=2 and travel>=self.min_travel})
        return result
