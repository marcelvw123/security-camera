"""Replay the actual motion-first inference function; never sends alerts."""
import argparse, json
from pathlib import Path
import cv2
from ultralytics import YOLO
from motion_crop import select_motion_crop
from motion_difference import FrameDifferenceMotion
from moving_object_gate import MovingObjectGate
from object_detection import process_motion_frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();model=YOLO('models/yolo11n.pt');results=[]
    for case in json.loads(args.manifest.read_text())['cases']:
        cap=cv2.VideoCapture(case['path']);fps=cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or fps<=0:raise RuntimeError('Cannot open '+case['path'])
        length=min(cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps,case.get('max_seconds',10**6))
        sens=case.get('sensitivity',3)
        motion=FrameDifferenceMotion(pixel_threshold=20 if sens==1 else 15,min_area=max(1,int(200*5/sens)),exclusions=case.get('exclusions',[]))
        gate=MovingObjectGate();samples=[]
        for i in range(int(length*2)):
            t=i*.5;cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read()
            if not ok:break
            height,width=frame.shape[:2];old,oldtime=motion.previous,motion.previous_time
            regions=motion.observe(frame,t)
            if t<5 or not regions:continue
            crop=select_motion_crop(regions,width,height)
            if crop is None:continue
            pair=(old,motion.previous,t-oldtime) if oldtime is not None else None
            detections,details=process_motion_frame(model,frame,crop,regions,pair,gate,t)
            qualified=[d for d in details['moving_gate'] if d['qualified']]
            if qualified:
                samples.append({'seconds':t,'moving':qualified,'scene_labels':[d[0] for d in detections]})
        cap.release()
        record={'name':case['name'],'replayed_seconds':length,'qualified_frames':samples}
        results.append(record)
        print(json.dumps({'name':case['name'],'qualified_frames':len(samples),
                          'moving_labels':sorted({d['label'] for sample in samples for d in sample['moving']})}),flush=True)
    args.output.write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
