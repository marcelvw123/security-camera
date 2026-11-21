"""Estimate coherent image motion inside an object box for experimental gating."""
import math


def motion_flow_evidence(previous, current, seconds, boxes, frame_width, frame_height):
    """Return per-box motion at the motion detector's reduced image resolution.

    Optical flow is only evidence of image motion; camera movement, reflections,
    and objects passing over one another can produce false positives.
    """
    import cv2
    import numpy as np
    if (previous is None or current is None or previous.shape != current.shape
            or previous.ndim != 2 or not .1 <= seconds <= 2 or frame_width <= 0 or frame_height <= 0):
        return [None] * len(boxes)
    flow = cv2.calcOpticalFlowFarneback(previous, current, None, .5, 3, 15, 3, 5, 1.2, 0)
    height, width = previous.shape
    results = []
    for box in boxes:
        try:
            x1, y1, x2, y2 = map(float, box)
            if not all(math.isfinite(v) for v in (x1,y1,x2,y2)) or x2 <= x1 or y2 <= y1:
                raise ValueError
            x1=max(0,min(width,int(x1*width/frame_width))); x2=max(0,min(width,math.ceil(x2*width/frame_width)))
            y1=max(0,min(height,int(y1*height/frame_height))); y2=max(0,min(height,math.ceil(y2*height/frame_height)))
            roi=flow[y1:y2,x1:x2]
            if roi.shape[0] < 3 or roi.shape[1] < 3:
                raise ValueError
            vectors=roi.reshape(-1,2)
            median=np.median(vectors,axis=0)
            coherent=float(np.linalg.norm(median)) * .5 / seconds
            results.append(round(coherent,3))
        except (TypeError,ValueError):
            results.append(None)
    return results
