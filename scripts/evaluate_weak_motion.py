import cv2,json,math
from ultralytics import YOLO
from motion_difference import FrameDifferenceMotion
from motion_flow import motion_flow_evidence
from weak_motion_vehicle import WeakMotionVehicle
cases=json.load(open('../dataset_work/motion_association_audit_20261120/crop_manifest.json'))['cases'];model=YOLO('models/yolo26n.pt');out=[]
for case in cases:
 c=cv2.VideoCapture(case['path']);fps=c.get(cv2.CAP_PROP_FPS);duration=c.get(cv2.CAP_PROP_FRAME_COUNT)/fps
 sens=case['sensitivity'];m=FrameDifferenceMotion(pixel_threshold=20 if sens==1 else 15,min_area=max(1,int(200*5/sens)),exclusions=case['exclusions']);g=WeakMotionVehicle();samples=[]
 for i in range(int(min(duration,case.get('max_seconds',duration))*2)):
  t=i*.5;c.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,f=c.read()
  if not ok:break
  h,w=f.shape[:2];prev=m.previous;pt=m.previous_time;b=m.observe(f,t)
  if t<5 or not b:continue
  pred=model.predict(f,imgsz=640,classes=[2,3,5,7],conf=.05,device='cpu',verbose=False)[0]
  det=[{'label':model.names[int(k)],'confidence':round(float(conf),3),'box':[float(v) for v in box]} for k,conf,box in zip(pred.boxes.cls,pred.boxes.conf,pred.boxes.xyxy)]
  flows=motion_flow_evidence(prev,m.previous,t-pt,[x['box'] for x in det],w,h) if det else []
  # Weak detections need two separate samples, each backed by coherent flow.
  checked=g.observe(det,b,flows,t,w,h)
  if checked:samples.append({'t':t,'checks':checked,'det':det})
 print(case['name'],json.dumps(samples),flush=True)
 out.append({'name':case['name'],'samples':samples})
open('../dataset_work/motion_association_audit_20261120/weak_motion_results.json','w').write(json.dumps(out,indent=2))
