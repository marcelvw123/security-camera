"""Bounded, shared object inference for the motion-only camera application."""
import logging
import math
import threading
import time
import os
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

VEHICLES = {'car', 'motorcycle', 'bus', 'truck'}


class ObjectGate:
    """Confirm people across samples."""
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
                category = label
                if category != 'person':
                    continue
                x1, y1, x2, y2 = box
                cx, cy = (x1+x2)/2, (y1+y2)/2
                choices = []
                for i, (old_label, old_box, _) in enumerate(self.previous):
                    old_category = old_label
                    if i in used or old_category != category:
                        continue
                    ox1, oy1, ox2, oy2 = old_box
                    distance = math.hypot(cx-(ox1+ox2)/2, cy-(oy1+oy2)/2)
                    if distance <= max(width*.10, x2-x1, ox2-ox1):
                        choices.append((distance, i))
                if choices:
                    distance, i = min(choices)
                    used.add(i)
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


class VehicleDetectionGate:
    """Start immediately on a detection, and rearm after a quiet sampled period."""
    def __init__(self, quiet_seconds=8):
        self.active = False
        self.last_seen = None
        self.last_sample = None
        self.quiet_seconds = quiet_seconds

    def update(self, timestamp, labels):
        detected = any(label in VEHICLES for label in labels)
        if self.last_sample is not None and timestamp < self.last_sample and not detected:
            return False
        self.last_sample = max(timestamp, self.last_sample or timestamp)
        if detected:
            self.last_seen = max(timestamp, self.last_seen or timestamp)
            if not self.active:
                self.active = True
                return True
        elif self.active and self.last_seen is not None and timestamp - self.last_seen >= self.quiet_seconds:
            self.active = False
        return False


class SharedObjectDetector:
    """At most one pending frame per camera; inference never blocks capture."""
    def __init__(self, model_path, interval=1.0):
        self.model_path = Path(model_path)
        self.weak_model_path = os.getenv('WEAK_MOTION_MODEL', '').strip()
        self.interval = interval
        self.condition = threading.Condition()
        self.pending = OrderedDict()
        self.results = {}
        self.last_submitted = {}
        self.last_crop_submitted = {}
        self.generations = {}
        self.processing_camera = None
        self.processing_started = None
        self.status = 'starting'
        # All inference belongs to this app and follows the motion-crop gate.
        self.remote_shadow = None
        self.thread = threading.Thread(target=self._run, daemon=True, name='object-inference')
        self.thread.start()

    def submit(self, camera, frame, timestamp, motion_boxes=None, motion_pair=None):
        from motion_crop import select_motion_crop
        crop = select_motion_crop(motion_boxes, frame.shape[1], frame.shape[0])
        if crop is None:
            return
        with self.condition:
            if self.status == 'error' or timestamp - self.last_submitted.get(camera, -float('inf')) < self.interval:
                return
            self.last_submitted[camera] = timestamp
            self.pending[camera] = (frame.copy(), timestamp, self.generations.get(camera, 0), crop,
                                    list(motion_boxes), motion_pair, datetime.now(timezone.utc).isoformat())
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
            self.last_crop_submitted.pop(camera, None)
            self.generations[camera] = self.generations.get(camera, 0) + 1

    def take_remote(self, camera):
        return self.remote_shadow.take(camera) if self.remote_shadow else None

    def vehicle_crop_revision(self, camera):
        verifier = self.remote_shadow
        return verifier.crop_revision(camera) if isinstance(verifier, LocalVehicleVerifier) else None

    def is_processing(self, camera):
        with self.condition:
            processing = self.processing_camera == camera
        return processing or bool(self.remote_shadow and self.remote_shadow.is_processing(camera))

    def health(self):
        with self.condition:
            stalled = self.processing_started is not None and time.monotonic() - self.processing_started > 60
            return 'error' if not self.thread.is_alive() else ('stalled' if stalled else self.status)

    def _run(self):
        try:
            import torch
            from ultralytics import YOLO
            from moving_object_gate import MovingObjectGate
            torch.set_num_threads(2)
            if not self.model_path.is_file():
                raise FileNotFoundError('Object model file not found')
            model = YOLO(str(self.model_path))
            verifier_path = self.model_path.parent / 'yolo11s.pt'
            person_verifier = None
            if verifier_path.is_file():
                try:
                    person_verifier = YOLO(str(verifier_path))
                except Exception:
                    logging.exception('Person verifier could not load; group notifications disabled')
            else:
                logging.warning('Person verifier missing; group notifications disabled: %s', verifier_path)
            gates = {}
            self.status = 'ready' if person_verifier is not None else 'degraded'
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: bool(self.pending))
                    camera, (frame, timestamp, generation, crop, regions, motion_pair, captured_at) = self.pending.popitem(last=False)
                    self.processing_camera = camera
                    self.processing_started = time.monotonic()
                if camera not in gates or gates[camera][0] != generation:
                    gates[camera] = (generation, MovingObjectGate())
                try:
                    detections, crop_result = process_motion_frame(
                        model, frame, crop, regions, motion_pair, gates[camera][1], timestamp, person_verifier)
                    crop_result['captured_at'] = captured_at
                    if crop_result.get('person_verification_error'):
                        self.status = 'degraded'
                    elif 'person_verification' in crop_result:
                        self.status = 'ready'
                except Exception:
                    logging.exception('Motion-gated detection failed for %s', camera)
                    detections, crop_result = [], {'crop_box': list(crop), 'motion_regions': regions,
                                                  'error': 'motion-gated inference failed'}
                with self.condition:
                    self.processing_camera = None
                    self.processing_started = None
                    if generation == self.generations.get(camera, 0):
                        self.results[camera] = (timestamp, frame.shape[1], detections,
                                                frame if detections else None, crop_result)
        except Exception:
            logging.exception('Object detection stopped')
            with self.condition:
                self.status = 'error'
                self.processing_camera = None
                self.processing_started = None
                self.pending.clear()


def process_motion_frame(model, frame, crop, regions, motion_pair, gate, timestamp, person_verifier=None):
    """Detect the motion crop first; run scene inference only after it qualifies."""
    from motion_crop import detect_motion_crop, merge_scene_detections
    from motion_flow import motion_flow_evidence
    height, width = frame.shape[:2]
    cropped = detect_motion_crop(model, frame, crop)
    flows = None
    if motion_pair is not None and cropped:
        try:
            flows = motion_flow_evidence(*motion_pair, [item['box'] for item in cropped], width, height)
        except Exception:
            logging.exception('Image motion calculation failed; using tracked observations')
    gated = gate.observe(cropped, regions, timestamp, width, height, flows)
    comparison = {'crop_box': list(crop), 'motion_regions': regions, 'detections': cropped,
                  'flow_evidence': flows, 'moving_gate': gated}
    qualified = [item for item in gated if item['qualified']]
    if not qualified:
        comparison['combined'] = []
        comparison['full_frame_ran'] = False
        return [], comparison
    scene = model.predict(frame, classes=[0, 2, 3, 5, 7], conf=.35,
                          imgsz=640, device='cpu', verbose=False)[0]
    full = [{'label': model.names[int(cls)], 'box': [float(v) for v in box], 'confidence': float(conf)}
            for box, conf, cls in zip(scene.boxes.xyxy, scene.boxes.conf, scene.boxes.cls)]
    from notification_scenarios import count_people, crop_person_supported
    for item in gated:
        if item['qualified'] and not crop_person_supported(item, full, height):
            item.update(qualified=False, reason='small person not confirmed in full frame')
    supported_crops = [item for item in cropped if crop_person_supported(item, full, height)]
    comparison['full_frame_detections'] = full
    comparison['combined'] = merge_scene_detections(full, supported_crops)
    comparison['full_frame_ran'] = True
    if not any(item['qualified'] for item in gated):
        return [], comparison
    comparison['verified_full_people'] = []
    if count_people(full) >= 2:
        if person_verifier is None:
            comparison['person_verification_error'] = 'person verifier unavailable'
        else:
            try:
                from person_verifier import verify_scene_people
                checked = verify_scene_people(person_verifier, frame, full)
                comparison['person_verification'] = checked
                comparison['verified_full_people'] = [
                    {**item, 'box': item['verification_box']} for item in checked if item['verified']]
            except Exception:
                logging.exception('Person verification failed')
                comparison['person_verification_error'] = 'person verification failed'
    return [(item['label'], item['box'], item['confidence']) for item in comparison['combined']], comparison

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
        self.processing_camera = None
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

    def is_processing(self, camera):
        with self.condition:
            return self.processing_camera == camera

    def _run(self):
        import cv2
        import json
        from urllib.parse import urlencode
        from urllib.request import Request, urlopen
        while True:
            with self.condition:
                self.condition.wait_for(lambda: bool(self.pending))
                camera, (frame, timestamp, generation) = self.pending.popitem(last=False)
                self.processing_camera = camera
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
                        self.results[camera] = (timestamp, data['detections'], frame if data['detections'] else None)
                if data.get('detections'):
                    logging.info('REMOTE_VEHICLE_SHADOW camera=%s detections=%s', camera, data['detections'])
            except Exception as exc:
                now = time.monotonic()
                if now - self.last_error >= 60:
                    logging.warning('Remote vehicle verifier unavailable: %s', type(exc).__name__)
                    self.last_error = now
            finally:
                with self.condition:
                    self.processing_camera = None


class LocalVehicleVerifier(RemoteVehicleShadow):
    """Analyze one latest road frame at a time on the Pi, without a network service."""
    CROPS = {'manual:701': (.46875, .13889, .390625, .33333),
             'manual:801': (0, .0868, .4375, .3125)}

    @classmethod
    def default_crop(cls, camera):
        canonical = RemoteVehicleShadow.verifier_camera(camera)
        return list(cls.CROPS.get(canonical, (0, 0, 1, 1)))

    @staticmethod
    def verifier_camera(camera):
        return camera if camera else None

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.condition = threading.Condition()
        self.pending, self.last_submitted, self.results, self.generations = OrderedDict(), {}, {}, {}
        self.processing_camera = None
        self.crops = {}
        self.last_error = 0
        self.status = 'starting'
        threading.Thread(target=self._run, daemon=True, name='local-vehicle-verifier').start()

    def set_crop(self, camera, crop):
        from motion_difference import FrameDifferenceMotion
        FrameDifferenceMotion.validate_exclusions([crop])
        with self.condition:
            self.crops[camera] = list(crop)
            self.pending.pop(camera, None)
            self.results.pop(camera, None)
            self.last_submitted.pop(camera, None)
            self.generations[camera] = self.generations.get(camera, 0) + 1

    def crop_revision(self, camera):
        with self.condition:
            return self.generations.get(camera, 0)

    def current_crop(self, camera):
        with self.condition:
            return list(self.crops.get(camera, self.default_crop(camera)))

    def _run(self):
        try:
            import torch
            from ultralytics import YOLO
            torch.set_num_threads(2)
            if not self.model_path.is_file():
                raise FileNotFoundError('Local vehicle model not found')
            model = YOLO(str(self.model_path))
            self.status = 'ready'
            logging.info('Local vehicle verifier ready')
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: bool(self.pending))
                    camera, (frame, timestamp, generation) = self.pending.popitem(last=False)
                    self.processing_camera = camera
                    x, y, w, h = self.crops.get(camera, self.default_crop(camera))
                height, width = frame.shape[:2]
                x1, y1 = int(x*width), int(y*height)
                x2, y2 = max(x1+1, min(width, math.ceil((x+w)*width))), max(y1+1, min(height, math.ceil((y+h)*height)))
                prediction = model.predict(frame[y1:y2, x1:x2], classes=[2, 3, 5, 7], conf=.1,
                                           imgsz=416, device='cpu', verbose=False)[0]
                detections = [{'label': model.names[int(cls)], 'confidence': round(float(conf), 3),
                               'box': [round(float(box[0])+x1, 1), round(float(box[1])+y1, 1),
                                       round(float(box[2])+x1, 1), round(float(box[3])+y1, 1)]}
                              for box, conf, cls in zip(prediction.boxes.xyxy, prediction.boxes.conf, prediction.boxes.cls)]
                with self.condition:
                    self.processing_camera = None
                    if generation == self.generations.get(camera, 0):
                        self.results[camera] = (timestamp, detections, frame if detections else None)
                if detections:
                    logging.info('LOCAL_VEHICLE camera=%s detections=%s', camera, detections)
        except Exception:
            self.status = 'error'
            with self.condition:
                self.processing_camera = None
            logging.exception('Local vehicle verifier stopped')


class ObjectClipRecorder:
    """Create one separate short clip for a confirmed event using capture frames."""
    def __init__(self, detector, camera, output_dir, writer_factory, status_callback, preroll,
                 min_free_bytes, fps=10, seconds=20, alert_camera=None):
        self.detector, self.camera = detector, camera
        self.alert_camera = alert_camera or camera
        self.output_dir, self.writer_factory = Path(output_dir), writer_factory
        self.status_callback, self.min_free_bytes = status_callback, min_free_bytes
        self.fps, self.seconds = fps, seconds
        self.preroll = preroll
        self.gate = ObjectGate()
        from alert_policy import AlertPolicy
        self.alert_policy = AlertPolicy()
        from motion_association import MotionAssociation
        self.motion_association = MotionAssociation()
        from moving_object_gate import MovingObjectGate
        self.moving_object_gate = MovingObjectGate()
        from weak_motion_vehicle import WeakMotionVehicle
        self.weak_motion_vehicle = WeakMotionVehicle()
        self.last_idle_submit = None
        self.last_motion_event = {}
        self.last_scenario_alert = {}
        self.vehicle_gate = VehicleDetectionGate()
        self.road_crop_revision = detector.vehicle_crop_revision(camera)
        self.writer = None
        self.end_time = self.next_frame = 0
        self.next_disk_check = 0

    def observe(self, frame, now, motion_active, motion_boxes=None, motion_pair=None):
        import shutil
        from datetime import datetime
        from motion_preroll import AsyncPreRollWriter
        # No object inference or baseline scan runs without a fresh motion crop.
        if motion_boxes:
            self.detector.submit(self.camera, frame, now, motion_boxes, motion_pair)
        result = self.detector.take(self.camera)
        found = []
        checked = []
        if result is not None and 0 <= now - result[0] <= 8:
            timestamp, _width, detections = result[:3]
            captured = result[3] if len(result) > 3 else None
            crop_result = result[4] if len(result) > 4 else None
            labels = [d[0] for d in detections]
            boxes = [{'label': d[0], 'box': d[1], 'confidence': d[2]} for d in detections]
            evidence = self.save_evidence(captured, boxes) if labels and (motion_active or self.writer) else None
            height, width = (captured if captured is not None else frame).shape[:2]
            associations = self.motion_association.check(boxes, timestamp, width, height)
            decisions, events = [], []
            qualified = [item for item in (crop_result or {}).get('moving_gate', []) if item.get('qualified')]
            for item in qualified:
                label = item.get('label')
                category = 'vehicle' if label in VEHICLES else label
                if category not in ('person', 'vehicle'):
                    continue
                quiet = 25 if category == 'vehicle' else 20
                previous = self.last_motion_event.get(category)
                if previous is not None and timestamp - previous < quiet:
                    continue
                self.last_motion_event[category] = timestamp
                found.append(category)
            # Scene rules are evaluated independently of the recording cooldown.
            # A moving vehicle may reveal two people in the same full frame.
            if qualified and (crop_result or {}).get('full_frame_ran'):
                from notification_scenarios import evaluate_person_scenario, scenario_people
                if crop_result.get('person_verification_error'):
                    decisions.append('person verification unavailable; group alert suppressed')
                people = scenario_people(crop_result.get('full_frame_detections', []), boxes,
                                         crop_result.get('verified_full_people', []))
                event = evaluate_person_scenario(people, crop_result.get('captured_at'))
                if event:
                    previous = self.last_scenario_alert.get(event['scenario'])
                    if previous is None or timestamp - previous >= 20:
                        self.last_scenario_alert[event['scenario']] = timestamp
                        events.append(event)
                    else:
                        decisions.append('person scenario in cooldown')
                else:
                    decisions.append('no person notification scenario matched')
            if crop_result and not qualified:
                decisions.append('motion crop did not confirm a moving object')
            for event in events:
                logging.info('Alert event camera=%s source=motion_crop object=%s reason=%s',
                             self.camera, event['label'], event['reason'])
            checked.append((timestamp, labels, evidence, events, decisions, 'full', associations, crop_result))
        if any(check[3] for check in checked) and not found:
            found.append('person')
        if found and self.writer is None and (motion_active or any(check[3] for check in checked)):
                self.output_dir.mkdir(parents=True, exist_ok=True)
                if shutil.disk_usage(self.output_dir).free >= self.min_free_bytes:
                    name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in self.camera)
                    path = self.output_dir / f"object_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4"
                    height, width = frame.shape[:2]
                    raw = self.writer_factory(path, self.fps, (width, height))
                    self.writer = AsyncPreRollWriter(raw, self.preroll, now, self.fps)
                    from motion_clip_metadata import start_metadata
                    start_metadata(self.writer.path)
                    self.end_time, self.next_frame = now + self.seconds, now
                    self.next_disk_check = now + 5
                    self.status_callback(True)
                    logging.info('Object clip started: camera=%s objects=%s file=%s', self.camera, found, path.name)
                else:
                    logging.warning('Object clip skipped: disk reserve reached')
        if self.writer:
            from motion_clip_metadata import update_metadata
            for _timestamp, labels, evidence, events, decisions, _source, _associations, _crop_result in checked:
                if labels or events:
                    update_metadata(self.writer.path, labels, evidence, alerts=events, decisions=decisions)
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
        return checked

    def save_evidence(self, frame, detections):
        """Store the actual full frame used by inference, annotated in source pixels."""
        import cv2
        import shutil
        import tempfile
        from datetime import datetime
        if frame is None:
            logging.warning('Detection frame unavailable for %s', self.camera)
            return None
        try:
            folder = self.output_dir / 'detection_frames'
            folder.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(folder).free < self.min_free_bytes:
                logging.warning('Detection frame not saved: disk reserve reached')
                return None
            annotated = frame.copy()
            height, width = annotated.shape[:2]
            for detection in detections:
                box = detection.get('box')
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                x1, y1, x2, y2 = [int(round(float(value))) for value in box]
                x1, x2 = max(0, min(width-1, x1)), max(0, min(width-1, x2))
                y1, y2 = max(0, min(height-1, y1)), max(0, min(height-1, y2))
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,220,0), 2)
                cv2.putText(annotated, str(detection.get('label', 'object')), (x1,max(15,y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0,220,0), 2)
            ok, encoded = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                raise OSError('JPEG encoding failed')
            safe_camera = ''.join(c if c.isalnum() or c in '_-' else '_' for c in self.camera)
            name = f"detection_{safe_camera}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}.jpg"
            path = folder / name
            temporary = None
            try:
                with tempfile.NamedTemporaryFile('wb', dir=folder, prefix='.detection_', delete=False) as out:
                    temporary = out.name
                    out.write(encoded.tobytes())
                    out.flush()
                os.replace(temporary, path)
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
            return name
        except (OSError, ValueError, TypeError, cv2.error):
            logging.exception('Could not save detection frame for %s', self.camera)
            return None

    def close(self):
        if self.writer:
            writer, self.writer = self.writer, None
            saved = False
            try:
                writer.release()
                saved = True
                logging.info('Object clip saved: %s', writer.path)
            finally:
                from motion_clip_metadata import finish_metadata, load_metadata
                finish_metadata(writer.path)
                self.status_callback(False)
            if saved and os.getenv('WHATSAPP_OBJECT_ALERTS_ENABLED', '').lower() in ('1', 'true', 'yes', 'on'):
                metadata = load_metadata(writer.path)
                if metadata and metadata.get('alert_objects'):
                    from object_alerts import queue_object_alert
                    queue_object_alert(writer.path, self.alert_camera, metadata['alert_objects'])

    def stop(self):
        try:
            self.close()
        finally:
            self.detector.remove(self.camera)

    def reset(self):
        self.stop()
        self.gate = ObjectGate()
        self.vehicle_gate = VehicleDetectionGate()
