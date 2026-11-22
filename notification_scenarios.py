"""Final notification rules for people observed in a motion-qualified scene."""
import math
from datetime import datetime
from zoneinfo import ZoneInfo

LOCAL_TIMEZONE = ZoneInfo('Africa/Johannesburg')


def crop_person_supported(person, full_detections, frame_height, minimum_height_ratio=.06):
    """Small crop-only person guesses need a matching full-frame person box."""
    box = person.get('box')
    if person.get('label') != 'person':
        return True
    if not isinstance(box, (list, tuple)) or len(box) != 4 or frame_height <= 0:
        return False
    x1, y1, x2, y2 = box
    if (y2-y1) / frame_height >= minimum_height_ratio:
        return True
    area = max(0, x2-x1) * max(0, y2-y1)
    if not area:
        return False
    for item in full_detections:
        other = item.get('box')
        if item.get('label') != 'person' or not isinstance(other, (list, tuple)) or len(other) != 4:
            continue
        overlap = max(0, min(x2,other[2])-max(x1,other[0])) * max(0, min(y2,other[3])-max(y1,other[1]))
        other_area = max(0,other[2]-other[0]) * max(0,other[3]-other[1])
        if other_area and overlap / min(area, other_area) >= .5:
            return True
    return False


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


def scenario_people(full_detections, combined_detections, verified_people=None):
    """Require verified people for a group; allow one supported person overnight."""
    full_people = [item for item in full_detections if item.get('label') == 'person']
    if count_people(verified_people or []) >= 2:
        return verified_people
    return (full_people or [item for item in combined_detections if item.get('label') == 'person'])[:1]


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
