"""Bounded second-model verification for full-frame person detections."""

from notification_scenarios import count_people


def verify_scene_people(model, frame, detections, min_confidence=.35, max_candidates=4):
    """Confirm at most four person boxes using padded crops of the same frame.

    Return confirmed source-frame boxes and scores for diagnostics. Stop after
    two confirmations because the current scenarios need only the group count.
    """
    height, width = frame.shape[:2]
    candidates = sorted((d for d in detections if d.get('label') == 'person'),
                        key=lambda d: d.get('confidence', 0), reverse=True)[:max_candidates]
    checked = []
    for candidate in candidates:
        x1,y1,x2,y2 = candidate['box']
        px,py = (x2-x1)*.5, (y2-y1)*.5
        left,top = max(0,int(x1-px)),max(0,int(y1-py))
        right,bottom = min(width,int(x2+px)),min(height,int(y2+py))
        if right <= left or bottom <= top:
            continue
        result = model.predict(frame[top:bottom,left:right], classes=[0], conf=min_confidence,
                               imgsz=640, device='cpu', verbose=False)[0]
        area=(x2-x1)*(y2-y1)
        best=0.0
        best_box=None
        for box,confidence in zip(result.boxes.xyxy,result.boxes.conf):
            bx1,by1,bx2,by2 = map(float,box)
            bx1+=left;bx2+=left;by1+=top;by2+=top
            overlap=max(0,min(x2,bx2)-max(x1,bx1))*max(0,min(y2,by2)-max(y1,by1))
            other_area=max(0,bx2-bx1)*max(0,by2-by1)
            if area > 0 and other_area > 0 and overlap/min(area,other_area) >= .5:
                if float(confidence) > best:
                    best=float(confidence)
                    best_box=[bx1,by1,bx2,by2]
        checked.append({**candidate,'verification_confidence':round(best,3),
                        'verification_box':best_box,'verified':best >= min_confidence})
        confirmed=[{**item,'box':item['verification_box']} for item in checked if item['verified']]
        if count_people(confirmed) >= 2:
            break
    return checked
