"""Bounded, shared object inference for the motion-only camera application."""
import logging
import math
import threading
import time
import os
from collections import OrderedDict
from pathlib import Path

VEHICLES = {'car', 'motorcycle', 'bus', 'truck'}


class ObjectGate:
    """Confirm people across samples; require vehicle displacement to exclude parked cars."""
    def __init__(self, quiet_seconds=8):
        self.previous = []
        self.previous_time = None
        self.active = False
        self.last_seen = None
        self.quiet_seconds = quiet_seconds

    def update(self, detections, timestamp, width):
        matches = []
        if self.previous_time is not None and 0 < timestamp - self.previous_time <= 8:
            used = set()
            for label, box, confidence in detections:
                category = 'vehicle' if label in VEHICLES else label
                if category not in ('person', 'vehicle'):
                    continue
                x1, y1, x2, y2 = box
                cx, cy = (x1+x2)/2, (y1+y2)/2
                choices = []
                for i, (old_label, old_box, _) in enumerate(self.previous):
                    old_category = 'vehicle' if old_label in VEHICLES else old_label
                    if i in used or old_category != category:
                        continue
                    ox1, oy1, ox2, oy2 = old_box
                    distance = math.hypot(cx-(ox1+ox2)/2, cy-(oy1+oy2)/2)
                    if category == 'vehicle':
                        old_width, old_height = max(1, ox2-ox1), max(1, oy2-oy1)
                        new_width, new_height = max(1, x2-x1), max(1, y2-y1)
                        if (max(old_width, new_width) / min(old_width, new_width) > 1.8 or
                                max(old_height, new_height) / min(old_height, new_height) > 1.8):
                            continue
                    if distance <= max(width*.10, x2-x1, ox2-ox1):
                        choices.append((distance, i))
                if choices:
                    distance, i = min(choices)
                    used.add(i)
                    if category == 'person' or distance >= max(12, width*.015):
                        matches.append(category)
        self.previous, self.previous_time = detections, timestamp
        if matches:
            self.last_seen = timestamp
            if not self.active:
                self.active = True
                return sorted(set(matches))
        elif self.active and self.last_seen is not None and timestamp - self.last_seen >= self.quiet_seconds:
            self.active = False
        return []


class RoadEntryGate:
    """Trigger on a newly detected road vehicle after an observed empty road."""
    def __init__(self, cooldown=20):
        self.empty_at = None
        self.last_event = -float('inf')
        self.cooldown = cooldown

    def update(self, timestamp, detections):
        if not detections:
            self.empty_at = timestamp
            return False
        if (self.empty_at is not None and 0 < timestamp - self.empty_at <= 8
                and timestamp - self.last_event >= self.cooldown):
            self.empty_at = None
            self.last_event = timestamp
            return True
        self.empty_at = None
        return False


class SharedObjectDetector:
    """At most one pending frame per camera; inference never blocks capture."""
    def __init__(self, model_path, interval=1.0):
        self.model_path = Path(model_path)
        self.interval = interval
        self.condition = threading.Condition()
        self.pending = OrderedDict()
        self.results = {}
        self.last_submitted = {}
        self.generations = {}
        self.status = 'starting'
        token_file = os.getenv('REMOTE_VEHICLE_TOKEN_FILE', '').strip()
        self.remote_shadow = RemoteVehicleShadow(token_file) if token_file else None
        self.thread = threading.Thread(target=self._run, daemon=True, name='object-inference')
        self.thread.start()

    def submit(self, camera, frame, timestamp):
        if self.remote_shadow:
            self.remote_shadow.submit(camera, frame, timestamp)
        with self.condition:
            if self.status == 'error' or timestamp - self.last_submitted.get(camera, -float('inf')) < self.interval:
                return
            self.last_submitted[camera] = timestamp
            self.pending[camera] = (frame.copy(), timestamp, self.generations.get(camera, 0))
            self.condition.notify()

    def take(self, camera):
        with self.condition:
            return self.results.pop(camera, None)

    def remove(self, camera):
        if self.remote_shadow:
            self.remote_shadow.remove(camera)
        with self.condition:
            self.pending.pop(camera, None)
            self.results.pop(camera, None)
            self.last_submitted.pop(camera, None)
            self.generations[camera] = self.generations.get(camera, 0) + 1

    def take_remote(self, camera):
        return self.remote_shadow.take(camera) if self.remote_shadow else None

    def _run(self):
        try:
            import torch
            from ultralytics import YOLO
            torch.set_num_threads(1)
            if not self.model_path.is_file():
                raise FileNotFoundError('Object model file not found')
            model = YOLO(str(self.model_path))
            self.status = 'ready'
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: bool(self.pending))
                    camera, (frame, timestamp, generation) = self.pending.popitem(last=False)
                result = model.predict(frame, classes=[0, 2, 3, 5, 7], conf=.35,
                                       imgsz=640, device='cpu', verbose=False)[0]
                detections = [(model.names[int(cls)], [float(v) for v in box], float(conf))
                              for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)]
                with self.condition:
                    if generation == self.generations.get(camera, 0):
                        self.results[camera] = (timestamp, frame.shape[1], detections)
        except Exception:
            logging.exception('Object detection stopped')
            with self.condition:
                self.status = 'error'
                self.pending.clear()

class RemoteVehicleShadow:
    """Send only the latest sampled frame per camera through a local SSH tunnel."""
    def __init__(self, token_file):
        self.token_file = Path(token_file)
        self.condition = threading.Condition()
        self.pending = OrderedDict()
        self.last_submitted = {}
        self.last_error = 0
        self.results = {}
        self.generations = {}
        threading.Thread(target=self._run, daemon=True, name='remote-vehicle-shadow').start()

    @staticmethod
    def verifier_camera(camera):
        if camera in ('manual:701', 'manual:801'):
            return camera
        if camera.startswith('probe:/Streaming/Channels/{channel}:'):
            channel = camera.rsplit(':', 1)[-1]
            if channel in ('701', '801'):
                return 'manual:' + channel
        return None

    def submit(self, camera, frame, timestamp):
        if self.verifier_camera(camera) is None:
            return
        with self.condition:
            if timestamp - self.last_submitted.get(camera, -float('inf')) < 1:
                return
            self.last_submitted[camera] = timestamp
            self.pending[camera] = (frame.copy(), timestamp, self.generations.get(camera, 0))
            self.condition.notify()

    def remove(self, camera):
        with self.condition:
            self.pending.pop(camera, None)
            self.last_submitted.pop(camera, None)
            self.results.pop(camera, None)
            self.generations[camera] = self.generations.get(camera, 0) + 1

    def take(self, camera):
        with self.condition:
            return self.results.pop(camera, None)

    def _run(self):
        import cv2
        import json
        from urllib.parse import urlencode
        from urllib.request import Request, urlopen
        while True:
            with self.condition:
                self.condition.wait_for(lambda: bool(self.pending))
                camera, (frame, timestamp, generation) = self.pending.popitem(last=False)
            try:
                ok, image = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if not ok:
                    raise ValueError('JPEG encoding failed')
                token = self.token_file.read_text().strip()
                verifier_camera = self.verifier_camera(camera)
                url = 'http://127.0.0.1:18766/verify?' + urlencode({'camera': verifier_camera})
                req = Request(url, data=image.tobytes(), headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'image/jpeg'})
                with urlopen(req, timeout=5) as response:
                    data = json.load(response)
                if data.get('camera') != verifier_camera or not isinstance(data.get('detections'), list):
                    raise ValueError('invalid verifier response')
                with self.condition:
                    if generation == self.generations.get(camera, 0):
                        self.results[camera] = (timestamp, data['detections'])
                if data.get('detections'):
                    logging.info('REMOTE_VEHICLE_SHADOW camera=%s detections=%s', camera, data['detections'])
            except Exception as exc:
                now = time.monotonic()
                if now - self.last_error >= 60:
                    logging.warning('Remote vehicle verifier unavailable: %s', type(exc).__name__)
                    self.last_error = now


class ObjectClipRecorder:
    """Create one separate short clip for a confirmed event using capture frames."""
    def __init__(self, detector, camera, output_dir, writer_factory, status_callback, preroll,
                 min_free_bytes, fps=10, seconds=20):
        self.detector, self.camera = detector, camera
        self.output_dir, self.writer_factory = Path(output_dir), writer_factory
        self.status_callback, self.min_free_bytes = status_callback, min_free_bytes
        self.fps, self.seconds = fps, seconds
        self.preroll = preroll
        self.gate = ObjectGate()
        self.road_gate = RoadEntryGate()
        self.writer = None
        self.end_time = self.next_frame = 0
        self.next_disk_check = 0

    def observe(self, frame, now, motion_active):
        import shutil
        from datetime import datetime
        from motion_preroll import AsyncPreRollWriter
        if motion_active or self.writer:
            self.detector.submit(self.camera, frame, now)
        result = self.detector.take(self.camera)
        found = []
        if result is not None and 0 <= now - result[0] <= 8:
            found = self.gate.update(result[2], result[0], result[1])
        remote = self.detector.take_remote(self.camera)
        if remote is not None and 0 <= now - remote[0] <= 8 and self.road_gate.update(remote[0], remote[1]):
            found = sorted(set(found + ['vehicle']))
        if found and self.writer is None:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                if shutil.disk_usage(self.output_dir).free >= self.min_free_bytes:
                    name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in self.camera)
                    path = self.output_dir / f"object_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4"
                    height, width = frame.shape[:2]
                    raw = self.writer_factory(path, self.fps, (width, height))
                    self.writer = AsyncPreRollWriter(raw, self.preroll, now, self.fps)
                    self.end_time, self.next_frame = now + self.seconds, now
                    self.next_disk_check = now + 5
                    self.status_callback(True)
                    logging.info('Object clip started: camera=%s objects=%s file=%s', self.camera, found, path.name)
                else:
                    logging.warning('Object clip skipped: disk reserve reached')
        if self.writer:
            if now >= self.next_disk_check:
                self.next_disk_check = now + 5
                if shutil.disk_usage(self.output_dir).free < self.min_free_bytes:
                    logging.warning('Object clip stopped: disk reserve reached')
                    self.end_time = min(now, self.end_time)
            while self.next_frame <= min(now, self.end_time):
                self.writer.write(frame)
                self.next_frame += 1/self.fps
            if now >= self.end_time:
                self.close()

    def close(self):
        if self.writer:
            writer, self.writer = self.writer, None
            try:
                writer.release()
                logging.info('Object clip saved: %s', writer.path)
            finally:
                self.status_callback(False)

    def stop(self):
        try:
            self.close()
        finally:
            self.detector.remove(self.camera)

    def reset(self):
        self.stop()
        self.gate = ObjectGate()
        self.road_gate = RoadEntryGate()
