"""Send completed object-positive clips through the existing WhatsApp API."""
import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path

from alert_outbox import AlertOutbox
from cloud_upload import upload_clip
from video_compat import ffmpeg_executable

_lock = threading.Lock()
_wakeup = threading.Event()
_started = False
_outbox = None
_on_accepted = None


def get_outbox():
    global _outbox
    with _lock:
        if _outbox is None:
            _outbox = AlertOutbox(Path.home() / '.local/state/securitycamera/alerts.db')
        return _outbox


def start_alert_worker(on_accepted=None):
    global _started, _on_accepted
    outbox = get_outbox()
    with _lock:
        if on_accepted is not None:
            _on_accepted = on_accepted
        if not _started:
            threading.Thread(target=_run, args=(outbox,), name='whatsapp-object-alerts', daemon=True).start()
            _started = True
    _wakeup.set()


def queue_object_alert(clip_path, camera, labels):
    labels = sorted({str(label) for label in labels if label})
    if not labels:
        return False
    try:
        get_outbox().enqueue(clip_path, camera, labels)
        start_alert_worker()
        _wakeup.set()
        logging.info('WhatsApp alert saved to queue: camera=%s clip=%s', camera, clip_path)
        return True
    except Exception as exc:
        logging.error('Could not persist WhatsApp alert (%s)', type(exc).__name__)
        return False


def _run(outbox):
    while True:
        try:
            item = outbox.claim()
            if item is None:
                _wakeup.wait(5)
                _wakeup.clear()
                continue
            path, camera, labels = item
            try:
                sent = bool(_send(Path(path), camera, labels))
            except Exception as exc:
                logging.error('WhatsApp alert outcome uncertain (%s)', type(exc).__name__)
                sent = False
            outbox.finish(path, sent)
            if sent and _on_accepted:
                _on_accepted(path)
        except Exception as exc:
            logging.error('WhatsApp queue unavailable (%s)', type(exc).__name__)
            _wakeup.wait(10)
            _wakeup.clear()


def _send(clip_path, camera, labels):
    ffmpeg = ffmpeg_executable()
    if not ffmpeg:
        logging.error('WhatsApp alert not sent: ffmpeg is unavailable')
        return False
    metadata = {'camera': camera, 'source_name': camera, 'detected': labels,
                'detected_objects': labels, 'clip_type': 'object_detection',
                'scenario': 'object_detected', 'upload_on_motion': False}
    with tempfile.TemporaryDirectory(prefix='securitycamera-alert-') as folder:
        video = Path(folder) / clip_path.name
        command = [ffmpeg, '-y', '-loglevel', 'error', '-i', str(clip_path), '-an',
                   '-c:v', 'libx264', '-threads', '1', '-profile:v', 'baseline',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                   '-preset', 'ultrafast', '-crf', '27', str(video)]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=300)
        except (OSError, subprocess.SubprocessError):
            logging.exception('WhatsApp alert video preparation failed: %s', clip_path)
            return False
        if not upload_clip(str(video), metadata):
            logging.error('WhatsApp object alert upload failed: %s', clip_path)
            return False
        return True
