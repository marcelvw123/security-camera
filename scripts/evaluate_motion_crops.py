"""Offline motion-crop versus full-frame comparison; never sends notifications."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from motion_crop import detect_motion_crop, select_motion_crop, merge_scene_detections
from motion_difference import FrameDifferenceMotion
from moving_object_gate import MovingObjectGate
from motion_flow import motion_flow_evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=Path('models/yolo11n.pt'))
    parser.add_argument('--gate-full-frame', action='store_true',
                        help='Only run scene detection when the moving-object gate passes')
    parser.add_argument('--image-motion', action='store_true',
                        help='Use coherent adjacent-frame motion as additional gate evidence')
    args=parser.parse_args()
    cases=json.loads(args.manifest.read_text())['cases']
    model=YOLO(str(args.model));results=[]
    for case in cases:
        path=Path(case['path']);cap=cv2.VideoCapture(str(path))
        fps=cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or fps<=0:
            raise RuntimeError(f'Cannot open video: {path.name}')
        duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
        limit=min(duration,case.get('max_seconds',duration))
        sensitivity=case.get('sensitivity',3)
        motion=FrameDifferenceMotion(pixel_threshold=20 if sensitivity==1 else 15,
             min_area=max(1,int(200*5/sensitivity)),exclusions=case.get('exclusions',[]))
        samples=[];next_crop=5.0;gate=MovingObjectGate()
        for i in range(int(limit*2)):
            t=i*.5;cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read()
            if not ok:break
            h,w=frame.shape[:2];previous,previous_time=motion.previous,motion.previous_time
            regions=motion.observe(frame,t)
            if t<next_crop or not regions:continue
            crop=select_motion_crop(regions,w,h)
            if crop is None:continue
            next_crop=t+1
            cropped=detect_motion_crop(model,frame,crop)
            flow=(motion_flow_evidence(previous,motion.previous,t-previous_time,[d['box'] for d in cropped],w,h)
                  if args.image_motion and previous_time is not None and cropped else None)
            gate_results=gate.observe(cropped,regions,t,w,h,flow)
            gate_passed=any(d['qualified'] for d in gate_results)
            full_detections=[]
            if gate_passed or not args.gate_full_frame:
                full=model.predict(frame,classes=[0,2,3,5,7],conf=.35,imgsz=640,device='cpu',verbose=False)[0]
                full_detections=[{'label':model.names[int(cl)],'confidence':float(conf),'box':[float(v) for v in box]}
                                 for box,conf,cl in zip(full.boxes.xyxy,full.boxes.conf,full.boxes.cls)]
            samples.append({'seconds':t,'motion_regions':regions,'crop_box':list(crop),
                            'crop_detections':cropped,'flow_evidence':flow,'moving_gate':gate_results,'gate_passed':gate_passed,
                            'full_frame_ran':gate_passed or not args.gate_full_frame, 'full_frame_detections':full_detections,
                            'combined_detections':merge_scene_detections(full_detections,cropped)})
        cap.release()
        with path.open('rb') as source:
            digest=hashlib.sha256()
            for chunk in iter(lambda: source.read(1024*1024),b''):digest.update(chunk)
        record={'name':case['name'],'expected':case.get('expected'),'sha256':digest.hexdigest(),
                'duration_seconds':duration,'replayed_seconds':limit,'samples':samples}
        results.append(record)
        print(json.dumps({'name':record['name'],'samples':len(samples),
              'crop_positive':sum(bool(s['crop_detections']) for s in samples),
              'full_positive':sum(bool(s['full_frame_detections']) for s in samples),
              'gate_positive':sum(any(d['qualified'] for d in s['moving_gate']) for s in samples)}),flush=True)
    args.output.write_text(json.dumps({'model':str(args.model),'motion_interval_seconds':.5,
        'crop_interval_seconds':1,'warmup_seconds':5,'gate_full_frame':args.gate_full_frame,
        'image_motion':args.image_motion,'results':results},indent=2)+'\n')


if __name__=='__main__':main()
