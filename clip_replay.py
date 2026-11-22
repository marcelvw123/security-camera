"""Isolated, real-time replay through the same live camera pipeline."""
import json
import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime
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

    def reserve(self, worker, timestamp, suffix):
        if suffix not in ('.mp4', '.mov', '.avi', '.mkv', '.m4v'):
            raise ValueError('Choose an MP4, MOV, AVI, MKV or M4V video')
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
                        'position': 0, 'duration': None, 'motion_checks': 0,
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
            ('source', '', {'.mp4','.mov','.avi','.mkv','.m4v'}),
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
                        if labels or events or (details and details.get('error')):
                            self.job['events'].append({'captured_at': (details or {}).get('captured_at'),
                                'objects': labels, 'scenarios': events, 'decisions': decisions,
                                'details': details})
                            self.job['events'] = self.job['events'][-300:]
        def execute():
            try:
                run_detection(str(path), 'Clip test - ' + worker.label, set(), self.stop_event.is_set,
                    motion_sensitivity=worker.sensitivity, excluded_areas=exclusions,
                    motion_algorithm=algorithm, object_detector=detector, object_source_id=source_id,
                    replay={'path': path, 'recorded_at': recorded_at, 'motion_dir': path.parent/'motion',
                            'object_dir': path.parent/'object'}, replay_progress_callback=progress,
                    replay_check_callback=checks)
                with self.lock:
                    self.job['status'] = 'cancelled' if self.stop_event.is_set() else 'complete'
                    self._persist()
            except Exception as exc:
                logging.warning('Clip replay failed (%s)', type(exc).__name__)
                self.fail('The clip could not be processed. Check its format and the application log.')
        self.thread = threading.Thread(target=execute, name='clip-replay', daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=20)
