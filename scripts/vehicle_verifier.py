#!/usr/bin/env python3
"""Local-only vehicle verifier. Use a secure tunnel to reach it from a camera device."""
import argparse
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import cv2
import numpy as np
from ultralytics import YOLO

CROPS = {'manual:701': (.46875, .13889, .390625, .33333),
         'manual:801': (0, .0868, .4375, .3125)}
MAX_BYTES = 4 * 1024 * 1024


def make_handler(model, token):
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_POST(self):
            if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token):
                return self.reply(401, {'error': 'unauthorized'})
            url = urlparse(self.path)
            camera = parse_qs(url.query).get('camera', [''])[0]
            if url.path != '/verify' or camera not in CROPS: return self.reply(400, {'error': 'unknown camera'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0 or length > MAX_BYTES: raise ValueError()
                frame = cv2.imdecode(np.frombuffer(self.rfile.read(length), np.uint8), cv2.IMREAD_COLOR)
                if frame is None or frame.shape[0] * frame.shape[1] > 12000000: raise ValueError()
                x,y,w,h = CROPS[camera]; height,width = frame.shape[:2]
                x1,y1,x2,y2 = int(x*width),int(y*height),int((x+w)*width),int((y+h)*height)
                if not lock.acquire(blocking=False): return self.reply(503, {'error':'busy'})
                try: result = model.predict(frame[y1:y2,x1:x2], classes=[2,3,5,7], conf=.1, imgsz=640, device='cpu', verbose=False)[0]
                finally: lock.release()
                boxes = [{'label':model.names[int(cls)],'confidence':round(float(conf),3),
                          'box':[round(float(box[0])+x1,1),round(float(box[1])+y1,1),round(float(box[2])+x1,1),round(float(box[3])+y1,1)]}
                         for box,conf,cls in zip(result.boxes.xyxy,result.boxes.conf,result.boxes.cls)]
                return self.reply(200, {'camera':camera,'detections':boxes})
            except (ValueError, TypeError): return self.reply(400, {'error':'invalid image'})
            except Exception: return self.reply(500, {'error':'verification failed'})
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True); parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8766)
    args=parser.parse_args(); token=args.token_file.read_text().strip()
    if len(token)<32: parser.error('token must have at least 32 characters')
    server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(YOLO(args.model),token))
    print(f'Vehicle verifier listening on localhost:{args.port}', flush=True)
    server.serve_forever()

if __name__ == '__main__': main()
