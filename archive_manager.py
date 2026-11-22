"""Find completed media and archive it without blocking camera capture."""
import logging
import threading
import time
from pathlib import Path

from archive_outbox import ArchiveOutbox
from archive_transport import upload_archive
from motion_clip_metadata import load_metadata, metadata_path, write_metadata


class ArchiveManager:
    def __init__(self, database, folders, transport=upload_archive, alert_status=None, alerts_enabled=True):
        self.outbox = ArchiveOutbox(database)
        self.folders = {kind: Path(folder) for kind, folder in folders.items()}
        self.transport = transport
        self.alert_status = alert_status or (lambda _path: None)
        self.alerts_enabled = alerts_enabled
        self.stop_event = threading.Event()
        self.thread = None

    def recover_interrupted(self):
        # Called once before camera workers start; no current writer owns these files.
        for kind in ('motion', 'object'):
            folder = self.folders.get(kind)
            if not folder or not folder.is_dir():
                continue
            for sidecar in folder.glob('*.detections.json'):
                clip = Path(str(sidecar)[:-len('.detections.json')])
                data = load_metadata(clip)
                if clip.is_file() and data and data.get('recording'):
                    data['recording'] = False
                    data['interrupted'] = True
                    write_metadata(clip, data)

    def scan(self, now=None):
        now = time.time() if now is None else now
        for kind, folder in self.folders.items():
            if not folder.is_dir():
                continue
            for path in folder.iterdir():
                if not path.is_file() or path.suffix.lower() not in ({'.jpg', '.jpeg', '.png'} if kind == 'image' else {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}):
                    continue
                try:
                    data = load_metadata(path) if kind != 'image' else None
                    if data and data.get('recording'):
                        continue
                    # A newly created writer may not yet have written its sidecar.
                    if kind != 'image' and data is None and (metadata_path(path).exists() or now - path.stat().st_mtime < 120):
                        continue
                    metadata = {'detections': data} if data is not None else {}
                    # Existing alert transport needs this source file. Keep alert clips
                    # until durable alert delivery is integrated and resolved.
                    retain = bool(self.alerts_enabled and kind == 'object' and data and data.get('alert_objects') and self.alert_status(path) != 'accepted')
                    self.outbox.enqueue(path, kind, metadata, retain_local=retain)
                    self.outbox.retry_cleanup(path)
                    if self.alerts_enabled and kind == 'object' and data and data.get('alert_objects') and not retain:
                        self.outbox.release(path)
                except (OSError, ValueError) as exc:
                    logging.warning('Archive scan skipped a file (%s)', type(exc).__name__)

    def run(self):
        next_scan = 0
        while not self.stop_event.is_set():
            try:
                if time.monotonic() >= next_scan:
                    self.scan()
                    next_scan = time.monotonic() + 30
                result = self.outbox.process_one(self.transport)
                if result is None:
                    self.stop_event.wait(5)
            except Exception as exc:
                logging.warning('Archive worker failed (%s)', type(exc).__name__)
                self.stop_event.wait(10)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, name='media-archive', daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
