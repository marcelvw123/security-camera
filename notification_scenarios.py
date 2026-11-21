"""Final notification rules for people observed in a motion-qualified scene."""
import math
from datetime import datetime
from zoneinfo import ZoneInfo

LOCAL_TIMEZONE = ZoneInfo('Africa/Johannesburg')


def count_people(detections):
    """Count same-frame person boxes, suppressing substantially overlapping duplicates."""
    candidates = []
    for detection in detections:
        if detection.get('label') != 'person':
            continue
        try:
            box = tuple(float(v) for v in detection['box'])
            if len(box) != 4 or not all(math.isfinite(v) for v in box):
                continue
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            confidence = float(detection.get('confidence', 0))
            if not math.isfinite(confidence):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        candidates.append((confidence, box))
    candidates.sort(key=lambda item: item[0], reverse=True)
    kept = []
    for _, box in candidates:
        area = (box[2]-box[0]) * (box[3]-box[1])
        duplicate = False
        for other in kept:
            other_area = (other[2]-other[0]) * (other[3]-other[1])
            overlap = max(0, min(box[2],other[2])-max(box[0],other[0])) * max(0, min(box[3],other[3])-max(box[1],other[1]))
            union = area + other_area - overlap
            if (union and overlap/union >= .5) or overlap/min(area,other_area) >= .8:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return len(kept)


def evaluate_person_scenario(detections, captured_at):
    """Return a matching event; captured_at must include its time-zone offset."""
    if isinstance(captured_at, str):
        try:
            captured_at = datetime.fromisoformat(captured_at)
        except ValueError:
            return None
    if not isinstance(captured_at, datetime) or captured_at.tzinfo is None or captured_at.utcoffset() is None:
        return None
    local_time = captured_at.astimezone(LOCAL_TIMEZONE)
    count = count_people(detections)
    if count >= 2:
        scenario, reason = 'two_or_more_people', f'{count} people detected'
    elif count == 1 and (local_time.hour >= 21 or local_time.hour < 5):
        scenario, reason = 'one_person_overnight', '1 person detected between 21:00 and 05:00'
    else:
        return None
    return {'label': 'person', 'scenario': scenario, 'reason': reason, 'person_count': count,
            'captured_at': local_time.isoformat()}
