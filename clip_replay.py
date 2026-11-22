"""Isolated, real-time replay through the same live camera pipeline."""
import json
import logging
import os
import tempfile
import threading
import uuid
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from object_detection import LocalVehicleVerifier
from security_camera_core import run_detection


class ClipReplay:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.job = None
        self.thread = None
        self.stop_event = threading.Event()
        candidates = sorted(self.root.glob('*/job.json'), key=lambda p: p.stat().st_mtime, reverse=True) if self.root.is_dir() else []
        for candidate in candidates:
            try:
                job = json.loads(candidate.read_text())
                if job.get('id') != candidate.parent.name:
                    continue
                if job.get('status') in ('uploading', 'running'):
                    job['status'] = 'error'
                    job['error'] = 'The app restarted before the test finished.'
                self.job = job
                break
            except (OSError, ValueError, AttributeError):
                continue

    def _persist(self):
        if not self.job:
            return
        folder = self.root / self.job['id']
        temporary = None
        try:
            with tempfile.NamedTemporaryFile('w', dir=folder, prefix='.job-', delete=False) as output:
                temporary = output.name
                os.fchmod(output.fileno(), 0o600)
                json.dump(self.job, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, folder / 'job.json')
        except (OSError, TypeError, ValueError) as exc:
            logging.warning('Could not persist clip test status (%s)', type(exc).__name__)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def reserve(self, worker, timestamp, suffix, stage='motion'):
        if suffix not in ('.mp4', '.mov', '.avi', '.mkv', '.m4v', '.jpg', '.jpeg', '.png'):
            raise ValueError('Choose a supported video or JPG/PNG image')
        if stage not in ('motion', 'object', 'scenario'):
            raise ValueError('Choose a valid starting step')
        if suffix in ('.jpg', '.jpeg', '.png') and stage == 'motion':
            raise ValueError('Motion detection needs a video with more than one frame')
        try:
            recorded_at = datetime.fromisoformat(timestamp)
            if recorded_at.tzinfo is None:
                recorded_at = recorded_at.replace(tzinfo=ZoneInfo('Africa/Johannesburg'))
        except (ValueError, TypeError):
            raise ValueError('Enter the date and time when the clip was recorded')
        with self.lock:
            if self.job and self.job['status'] in ('uploading', 'running'):
                raise ValueError('A clip test is already in progress')
            job_id = uuid.uuid4().hex
            folder = self.root / job_id
            folder.mkdir(parents=True, mode=0o700)
            self.stop_event.clear()
            self.job = {'id': job_id, 'status': 'uploading', 'camera': worker.label,
                        'position': 0, 'duration': None, 'stage': stage, 'motion_checks': 0,
                        'object_checks': 0, 'events': [], 'recorded_at': recorded_at.isoformat(),
                        'error': None}
            self._persist()
            return folder / ('source' + suffix), recorded_at

    def fail(self, message):
        with self.lock:
            if self.job:
                self.job['status'] = 'error'
                self.job['error'] = message
                self._persist()

    def snapshot(self):
        with self.lock:
            if not self.job:
                return None
            job = json.loads(json.dumps(self.job))
        folder = self.root / job['id']
        files = []
        for category, subdir, extensions in (
            ('source', '', {'.mp4','.mov','.avi','.mkv','.m4v','.jpg','.jpeg','.png'}),
            ('motion', 'motion', {'.mp4','.mov','.avi','.mkv','.m4v'}),
            ('object', 'object', {'.mp4','.mov','.avi','.mkv','.m4v'}),
            ('image', 'object/detection_frames', {'.jpg','.jpeg','.png'}),
        ):
            base = folder / subdir
            if base.is_dir():
                for path in sorted(base.iterdir()):
                    if path.is_file() and path.suffix.lower() in extensions:
                        files.append({'kind': category, 'name': path.name,
                                      'url': f'/replay-media/{job["id"]}/{subdir}/{path.name}'})
        job['files'] = files
        return job

    def start(self, path, recorded_at, worker, detector, algorithm):
        with self.lock:
            self.job['status'] = 'running'
            self._persist()
        source_id = 'replay:' + self.job['id']
        exclusions = worker.current_exclusions()
        vehicle_area = worker.current_vehicle_area()
        if isinstance(detector.remote_shadow, LocalVehicleVerifier):
            detector.remote_shadow.set_crop(source_id, vehicle_area)
        def progress(position, duration):
            with self.lock:
                self.job['position'], self.job['duration'] = round(position, 1), duration
                if int(position) % 3 == 0:
                    self._persist()
        def checks(entries):
            with self.lock:
                for entry in entries:
                    if isinstance(entry, dict):
                        self.job['motion_checks'] += 1
                    else:
                        _timestamp, labels, _evidence, events, decisions, _source, _associations, details = entry
                        self.job['object_checks'] += 1
                        self.job['events'].append({'captured_at': (details or {}).get('captured_at'),
                            'objects': labels, 'scenarios': events, 'decisions': decisions,
                            'details': details})
                        self.job['events'] = self.job['events'][-300:]
        def execute():
            try:
                if self.job.get('stage', 'motion') == 'motion':
                    run_detection(str(path), 'Clip test - ' + worker.label, set(), self.stop_event.is_set,
                        motion_sensitivity=worker.sensitivity, excluded_areas=exclusions,
                        motion_algorithm=algorithm, object_detector=detector, object_source_id=source_id,
                        replay={'path': path, 'recorded_at': recorded_at, 'motion_dir': path.parent/'motion',
                                'object_dir': path.parent/'object'}, replay_progress_callback=progress,
                        replay_check_callback=checks)
                else:
                    self._run_scene(path, recorded_at, detector, source_id, progress, checks)
                with self.lock:
                    self.job['status'] = 'cancelled' if self.stop_event.is_set() else 'complete'
                    self._persist()
            except Exception as exc:
                logging.warning('Clip replay failed (%s)', type(exc).__name__)
                self.fail('The clip could not be processed. Check its format and the application log.')
        self.thread = threading.Thread(target=execute, name='clip-replay', daemon=True)
        self.thread.start()

    def _run_scene(self, path, recorded_at, detector, source_id, progress, checks):
        import cv2
        last_events = {}
        image_dir = path.parent / 'object' / 'detection_frames'
        frame_number = 0
        images_saved = 0
        last_image_at = None

        def collect():
            nonlocal images_saved, last_image_at
            result = detector.take(source_id)
            if result is None:
                return
            timestamp, _width, detections, frame, details = result
            if details.get('error'):
                raise RuntimeError('Object inference failed')
            scenarios = []
            for event in details.get('scenarios', []):
                previous = last_events.get(event['scenario'])
                if previous is None or timestamp - previous >= 20:
                    last_events[event['scenario']] = timestamp
                    scenarios.append(event)
            labels = [item[0] for item in detections]
            decisions = []
            if details.get('person_verification_error'):
                decisions.append('person verification unavailable; group alert suppressed')
            if not scenarios:
                decisions.append('no person notification scenario matched')
            checks([(timestamp, labels, None, scenarios, decisions, 'full', [], details)])
            if (frame is not None and (detections or path.suffix.lower() in ('.jpg','.jpeg','.png'))
                    and images_saved < 100 and (last_image_at is None or timestamp - last_image_at >= 2 or scenarios)):
                import shutil
                image_dir.mkdir(parents=True, exist_ok=True)
                if shutil.disk_usage(image_dir).free < 2048 * 1024 * 1024:
                    raise OSError('Disk reserve reached')
                annotated = frame.copy()
                for label, box, confidence in detections:
                    x1,y1,x2,y2 = map(int, box)
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,220,0),2)
                    cv2.putText(annotated, f'{label} {confidence:.2f}', (x1,max(15,y1-5)),
                                cv2.FONT_HERSHEY_SIMPLEX,.55,(0,220,0),2)
                if not cv2.imwrite(str(image_dir / f'frame_{int(timestamp*1000)}.jpg'), annotated):
                    raise OSError('Could not save detection image')
                images_saved += 1
                last_image_at = timestamp

        def finish():
            deadline = time.monotonic() + 30
            while not self.stop_event.is_set() and detector.has_pending(source_id) and time.monotonic() < deadline:
                collect()
                time.sleep(.05)
            collect()
            if detector.has_pending(source_id) and not self.stop_event.is_set():
                raise RuntimeError('Object detection did not finish')

        try:
            if path.suffix.lower() in ('.jpg','.jpeg','.png'):
                frame = cv2.imread(str(path))
                if frame is None:
                    raise ValueError('Could not decode image')
                if frame.shape[0] * frame.shape[1] > 24000000:
                    raise ValueError('Image exceeds supported resolution')
                if not detector.submit_scene(source_id, frame, time.monotonic(), recorded_at.isoformat()):
                    raise RuntimeError('Object detector is unavailable')
                finish()
                return
            cap = cv2.VideoCapture(str(path))
            try:
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                if not cap.isOpened() or not 0 < fps <= 240:
                    raise ValueError('Could not decode video')
                count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = count / fps if count > 0 else None
                started = time.monotonic()
                last_progress = -1
                while not self.stop_event.is_set():
                    target = started + frame_number/fps
                    while not self.stop_event.is_set() and time.monotonic() < target:
                        time.sleep(min(.05, max(0, target-time.monotonic())))
                    if self.stop_event.is_set():
                        break
                    ok, frame = cap.read()
                    if not ok:
                        break
                    position = frame_number/fps
                    frame_number += 1
                    if int(position) != last_progress:
                        last_progress = int(position)
                        progress(position, duration)
                    collect()
                    captured_at = (recorded_at + timedelta(seconds=position)).isoformat()
                    detector.submit_scene(source_id, frame, time.monotonic(), captured_at)
                finish()
            finally:
                cap.release()
        finally:
            detector.remove(source_id)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=20)
