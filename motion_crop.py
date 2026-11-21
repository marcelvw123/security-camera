"""Bounded motion-region detection for comparison with a full-frame pass."""
import math


def select_motion_crop(regions, width, height, padding=.5, min_side=160, max_area=.7):
    """Return one padded xyxy crop in source pixels, or None for unusable regions."""
    candidates=[]
    for region in regions or []:
        if not isinstance(region,(list,tuple)) or len(region)!=4:
            continue
        try:x,y,w,h=map(float,region)
        except (TypeError,ValueError):continue
        if not all(math.isfinite(v) for v in (x,y,w,h)) or w<=0 or h<=0:
            continue
        cx,cy=x+w/2,y+h/2
        cw,ch=max(min_side,w*(1+2*padding)),max(min_side,h*(1+2*padding))
        x1,y1=max(0,math.floor(cx-cw/2)),max(0,math.floor(cy-ch/2))
        x2,y2=min(width,math.ceil(cx+cw/2)),min(height,math.ceil(cy+ch/2))
        if x2<=x1 or y2<=y1 or (x2-x1)*(y2-y1)>width*height*max_area:
            continue
        candidates.append((w*h,(x1,y1,x2,y2)))
    return max(candidates,key=lambda candidate:candidate[0])[1] if candidates else None


def detect_motion_crop(model, frame, crop):
    """Run the same detector on the crop and map its boxes back to source pixels."""
    x1,y1,x2,y2=crop
    result=model.predict(frame[y1:y2,x1:x2],classes=[0,2,3,5,7],conf=.35,
                         imgsz=640,device='cpu',verbose=False)[0]
    detections=[]
    for box,confidence,cls in zip(result.boxes.xyxy,result.boxes.conf,result.boxes.cls):
        b=[float(v) for v in box]
        detections.append({'label':model.names[int(cls)],'confidence':float(confidence),
                           'box':[max(x1,min(x2,b[0]+x1)),max(y1,min(y2,b[1]+y1)),
                                  max(x1,min(x2,b[2]+x1)),max(y1,min(y2,b[3]+y1))]})
    return detections


def merge_scene_detections(full_detections, crop_detections, overlap_threshold=.5):
    """Combine same-frame results while suppressing overlapping duplicate boxes."""
    vehicles={'car','bus','truck','motorcycle'}
    def category(item):
        return 'vehicle' if item['label'] in vehicles else item['label']
    def iou(a,b):
        x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3])
        inter=max(0,x2-x1)*max(0,y2-y1)
        union=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
        return inter/union if union else 0
    candidates=sorted(full_detections+crop_detections,key=lambda d:d['confidence'],reverse=True)
    merged=[]
    for detection in candidates:
        if not any(category(detection)==category(existing) and iou(detection['box'],existing['box'])>=overlap_threshold for existing in merged):
            merged.append(detection)
    return merged
