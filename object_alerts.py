"""Send completed object-positive clips through the existing WhatsApp API."""
import logging
import queue
import subprocess
import tempfile
import threading
from pathlib import Path

from cloud_upload import upload_clip
from video_compat import ffmpeg_executable

_queue = queue.Queue(maxsize=50)
_lock = threading.Lock()
_started = False


def queue_object_alert(clip_path, camera, labels):
    """Queue one alert for a completed clip with at least one detected object."""
    global _started
    labels = sorted({str(label) for label in labels if label})
    if not labels:
        return False
    with _lock:
        if not _started:
            threading.Thread(target=_run, name='whatsapp-object-alerts', daemon=True).start()
            _started = True
    try:
        _queue.put_nowait((Path(clip_path), camera, labels))
        logging.info('WhatsApp object alert queued: camera=%s objects=%s clip=%s', camera, labels, clip_path)
        return True
    except queue.Full:
        logging.error('WhatsApp object alert queue full: camera=%s clip=%s', camera, clip_path)
        return False


def _run():
    while True:
        item = _queue.get()
        try:
            _send(*item)
        except Exception:
            logging.exception('WhatsApp object alert failed: clip=%s', item[0])
        finally:
            _queue.task_done()


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
