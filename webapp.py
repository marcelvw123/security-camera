from __future__ import annotations

from runtime_warnings import configure_runtime_warnings

configure_runtime_warnings()

import argparse
import io
import re
import zipfile
import json
import logging
import mimetypes
import os
import signal
import sys
import threading
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from app_config import load_dotenv
from camera_discovery import DiscoveredStream, discover_nvr_streams
from camera_registry import CameraRegistry
from archive_client import archive_request, list_archive
from network_guess import likely_dvr_ip
from motion_difference import FrameDifferenceMotion
from object_detection import LocalVehicleVerifier
from motion_clip_metadata import load_metadata, title_for_metadata
from security_camera_core import (
    DEFAULT_RTSP_PATH_TEMPLATE,
    DEFAULT_RTSP_PORT,
    OUTPUT_DIR,
    OBJECT_OUTPUT_DIR,
    build_rtsp_url,
    can_open_rtsp_channel,
    run_detection,
)


LOG_FILE = "security_camera_web.log"
DEFAULT_MOTION_SENSITIVITY = 5
EXCLUSIONS_FILE = Path(__file__).resolve().parent / "motion_exclusions.json"
EXCLUSIONS_LOCK = threading.RLock()
VEHICLE_AREAS_FILE = Path(__file__).resolve().parent / "vehicle_detection_areas.json"
VEHICLE_AREAS_LOCK = threading.RLock()


def load_vehicle_areas():
    with VEHICLE_AREAS_LOCK:
        if not VEHICLE_AREAS_FILE.exists():
            return {}
        data = json.loads(VEHICLE_AREAS_FILE.read_text())
        if not isinstance(data, dict):
            raise ValueError('Invalid vehicle area settings')
        for crop in data.values():
            FrameDifferenceMotion.validate_exclusions([crop])
        return data


def save_vehicle_area(source_id, crop):
    FrameDifferenceMotion.validate_exclusions([crop])
    with VEHICLE_AREAS_LOCK:
        data = load_vehicle_areas()
        data[source_id] = crop
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=VEHICLE_AREAS_FILE.parent, prefix='.vehicle_areas_', delete=False) as out:
                temporary = out.name
                json.dump(data, out)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, VEHICLE_AREAS_FILE)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)


def load_exclusion_map():
    with EXCLUSIONS_LOCK:
        if not EXCLUSIONS_FILE.exists():
            return {}
        data = json.loads(EXCLUSIONS_FILE.read_text())
        if not isinstance(data, dict):
            raise ValueError("Invalid saved exclusion settings")
        for areas in data.values():
            FrameDifferenceMotion.validate_exclusions(areas)
        return data


def save_camera_exclusions(source_id, areas):
    FrameDifferenceMotion.validate_exclusions(areas)
    with EXCLUSIONS_LOCK:
        data = load_exclusion_map()
        data[source_id] = areas
        EXCLUSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=EXCLUSIONS_FILE.parent, prefix=".motion_exclusions_", delete=False) as out:
                temporary = out.name
                json.dump(data, out)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, EXCLUSIONS_FILE)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MEDIA_FOLDERS = {
    "clips": {
        "label": "Clips",
        "path": OUTPUT_DIR,
        "extensions": VIDEO_EXTENSIONS,
        "type": "video",
    },
    "object_clips": {
        "label": "Object clips",
        "path": OBJECT_OUTPUT_DIR,
        "extensions": VIDEO_EXTENSIONS,
        "type": "video",
    },
    "detection_frames": {
        "label": "Detection images",
        "path": OBJECT_OUTPUT_DIR / 'detection_frames',
        "extensions": IMAGE_EXTENSIONS,
        "type": "image",
    },
}

OBJECT_DETECTOR = None
if os.getenv("OBJECT_DETECTION_ENABLED", "false").lower() in ("1", "true", "yes", "on"):
    from object_detection import SharedObjectDetector
    OBJECT_DETECTOR = SharedObjectDetector(Path(__file__).resolve().parent / 'models' / 'yolo11n.pt', interval=.5)


@dataclass
class WorkerState:
    source_id: str
    label: str
    stop_event: threading.Event
    thread: threading.Thread
    kind: str
    sensitivity: int = 5
    motion_algorithm: str = "difference"
    status: str = "starting"
    object_recording: bool = False
    object_error: bool = False
    detected_objects: tuple = ()
    detected_at: float = 0.0
    preview_enabled: bool = True
    frame: bytes | None = None
    frame_version: int = 0
    last_frame_at: float = field(default_factory=time.monotonic)
    condition: threading.Condition = field(default_factory=threading.Condition)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    excluded_areas: list = field(default_factory=list)
    vehicle_area: list = field(default_factory=list)
    exclusions_lock: threading.Lock = field(default_factory=threading.Lock)

    def current_exclusions(self):
        with self.exclusions_lock:
            return [list(area) for area in self.excluded_areas]

    def recent_objects(self):
        with self.condition:
            return list(self.detected_objects) if time.monotonic() - self.detected_at < 8 else []

    def record_objects(self, labels):
        with self.condition:
            self.detected_objects = tuple(sorted({label for label in labels if isinstance(label, str) and label}))
            self.detected_at = time.monotonic()

    def set_exclusions(self, areas):
        with self.exclusions_lock:
            self.excluded_areas = [list(area) for area in areas]

    def current_vehicle_area(self):
        with self.exclusions_lock:
            return list(self.vehicle_area)

    def set_vehicle_area(self, area):
        with self.exclusions_lock:
            self.vehicle_area = list(area)


class WebAppState:
    def __init__(self):
        self.lock = threading.RLock()
        self.workers = {}
        self.scan_messages = []

    def add_worker(self, worker):
        with self.lock:
            if worker.source_id in self.workers:
                raise ValueError(f"{worker.label} is already running")
            self.workers[worker.source_id] = worker

    def get_worker(self, source_id):
        with self.lock:
            return self.workers.get(source_id)

    def remove_worker(self, source_id):
        with self.lock:
            return self.workers.pop(source_id, None)

    def worker_summaries(self):
        with self.lock:
            return [
                {
                    "source_id": worker.source_id,
                    "label": worker.label,
                    "kind": worker.kind,
                    "sensitivity": worker.sensitivity,
                    "motion_algorithm": worker.motion_algorithm,
                    "status": worker.status,
                    "frame_age_seconds": round(max(0, time.monotonic() - worker.last_frame_at), 1),
                    "recording": worker.status.startswith("Recording clip:"),
                    "object_recording": worker.object_recording,
                    "object_processing": bool(OBJECT_DETECTOR and OBJECT_DETECTOR.is_processing(worker.source_id)),
                    "detected_objects": worker.recent_objects(),
                    "object_detection_status": 'error' if worker.object_error else (OBJECT_DETECTOR.health() if OBJECT_DETECTOR else 'disabled'),
                    "vehicle_verifier_status": getattr(OBJECT_DETECTOR.remote_shadow, 'status', 'remote') if OBJECT_DETECTOR and OBJECT_DETECTOR.remote_shadow else 'disabled',
                    "preview_enabled": worker.preview_enabled,
                    "started_at": worker.started_at,
                    "excluded_areas": worker.current_exclusions(),
                    "vehicle_area": worker.current_vehicle_area(),
                }
                for worker in self.workers.values()
            ]


APP_STATE = WebAppState()
CAMERA_REGISTRY = CameraRegistry()
SUPERVISOR_STOP = threading.Event()
from clip_replay import ClipReplay
CLIP_REPLAY = ClipReplay(Path.home() / 'SecurityCamera' / 'clip_tests')
ARCHIVE_MANAGER = None
ARCHIVE_ENABLED = os.getenv("MEDIA_ARCHIVE_ENABLED", "false").lower() in ("1", "true", "yes", "on")


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def env_value(*names, default=""):
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def default_settings():
    load_dotenv()
    return {
        "username": env_value("CAMERA_USERNAME", "NVR_USERNAME"),
        "password": env_value("CAMERA_PASSWORD", "NVR_PASSWORD"),
        "ip_address": env_value("CAMERA_IP", "NVR_IP", default=likely_dvr_ip() or ""),
        "port": env_value("CAMERA_RTSP_PORT", "NVR_RTSP_PORT", default=DEFAULT_RTSP_PORT),
        "rtsp_path_template": env_value(
            "CAMERA_RTSP_PATH_TEMPLATE",
            "NVR_RTSP_PATH_TEMPLATE",
            default=DEFAULT_RTSP_PATH_TEMPLATE,
        ),
    }


def optional_text(value):
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None

    return text


def stream_from_payload(payload):
    return DiscoveredStream(
        label=str(payload.get("label", "Camera Stream")).strip() or "Camera Stream",
        stream_id=str(payload.get("stream_id", "")).strip(),
        rtsp_url=optional_text(payload.get("rtsp_url")),
        rtsp_path_template=optional_text(payload.get("rtsp_path_template")),
        channel=optional_text(payload.get("channel")),
        source=str(payload.get("source", "browser")).strip() or "browser",
    )


def stream_to_json(stream):
    return {
        "label": stream.label,
        "stream_id": stream.stream_id,
        "rtsp_url": stream.rtsp_url,
        "rtsp_path_template": stream.rtsp_path_template,
        "channel": stream.channel,
        "source": stream.source,
    }


def parse_sensitivity(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_MOTION_SENSITIVITY
    return max(1, min(parsed, 10))


def update_frame(worker, frame_bytes):
    with worker.condition:
        worker.frame = frame_bytes
        worker.frame_version += 1
        worker.condition.notify_all()


def update_status(worker, message):
    worker.status = message


def update_object_status(worker, value):
    worker.object_recording = value is True
    if value == 'error':
        worker.object_error = True


def preview_frame_callback(worker):
    def callback(frame_bytes):
        update_frame(worker, frame_bytes)

    callback.should_emit = lambda: worker.preview_enabled
    return callback


def run_camera_worker(worker, stream, settings, sensitivity, excluded_areas=None, motion_algorithm="difference"):
    try:
        if stream.rtsp_url:
            stream_url = stream.rtsp_url
            logging.info("Starting browser stream with direct RTSP URL: label=%s", worker.label)
        else:
            stream_settings = dict(settings)
            if stream.rtsp_path_template:
                stream_settings["rtsp_path_template"] = stream.rtsp_path_template
            stream_url = build_rtsp_url(stream_settings, channel=stream.channel)
            logging.info(
                "Starting browser stream from settings: label=%s template=%s channel=%s",
                worker.label,
                stream_settings.get("rtsp_path_template"),
                stream.channel,
            )

        worker.status = "running"
        run_detection(
            stream_url,
            worker.label,
            set(),
            worker.stop_event.is_set,
            preview_frame_callback(worker),
            sensitivity,
            lambda message: update_status(worker, message),
            upload_on_motion=True,
            excluded_areas=excluded_areas,
            motion_algorithm=motion_algorithm,
            exclusions_provider=worker.current_exclusions,
            object_detector=OBJECT_DETECTOR,
            object_status_callback=lambda value: update_object_status(worker, value),
            object_source_id=worker.source_id,
            object_result_callback=worker.record_objects,
            heartbeat_callback=lambda now: setattr(worker, 'last_frame_at', now),
        )
    except Exception as exc:
        logging.exception("Browser stream failed: %s", worker.label)
        worker.status = f"error: {exc}"
    finally:
        worker.stop_event.set()
        with worker.condition:
            worker.condition.notify_all()
        APP_STATE.remove_worker(worker.source_id)


def start_camera(payload, restore=False):
    settings = payload.get("settings", {})
    if not isinstance(settings, dict) or not isinstance(payload.get("stream", {}), dict):
        raise ValueError("Invalid camera settings")
    stream = stream_from_payload(payload.get("stream", {}))
    if not stream.stream_id:
        raise ValueError("stream_id is required")
    sensitivity = parse_sensitivity(payload.get("sensitivity"))
    algorithm = payload.get("motion_algorithm", "difference")
    if algorithm not in ("difference", "background"):
        raise ValueError("Invalid motion detection method.")
    exclusions = payload.get("excluded_areas")
    if exclusions is None or restore:
        exclusions = load_exclusion_map().get(stream.stream_id, [])
    FrameDifferenceMotion.validate_exclusions(exclusions)
    vehicle_area = load_vehicle_areas().get(stream.stream_id, LocalVehicleVerifier.default_crop(stream.stream_id))
    worker = WorkerState(source_id=stream.stream_id, label=stream.label, stop_event=threading.Event(),
                         thread=None, kind="camera", sensitivity=sensitivity, motion_algorithm=algorithm,
                         excluded_areas=exclusions, vehicle_area=vehicle_area)
    worker.thread = threading.Thread(target=run_camera_worker,
        args=(worker, stream, settings, sensitivity, exclusions, algorithm), daemon=True)
    # Hold the state lock while checking intent, so a stop cannot race a restore.
    with APP_STATE.lock:
        if restore and stream.stream_id not in CAMERA_REGISTRY.read():
            raise ValueError("Camera no longer selected")
        if APP_STATE.get_worker(stream.stream_id):
            raise ValueError("Camera is already running")
        if not restore:
            CAMERA_REGISTRY.put(stream.stream_id, {"stream": stream_to_json(stream), "settings": settings,
                "sensitivity": sensitivity, "motion_algorithm": algorithm})
        APP_STATE.add_worker(worker)
        try:
            if OBJECT_DETECTOR and isinstance(OBJECT_DETECTOR.remote_shadow, LocalVehicleVerifier):
                OBJECT_DETECTOR.remote_shadow.set_crop(worker.source_id, vehicle_area)
            worker.thread.start()
        except Exception:
            APP_STATE.remove_worker(worker.source_id)
            raise
    return worker


def supervise_cameras():
    while not SUPERVISOR_STOP.is_set():
        try:
            selected = CAMERA_REGISTRY.read()
            for source_id, payload in selected.items():
                if SUPERVISOR_STOP.is_set():
                    break
                if APP_STATE.get_worker(source_id) is None:
                    try:
                        start_camera(payload, restore=True)
                    except Exception as exc:
                        # Camera settings can contain passwords; never log the payload or exception.
                        logging.warning("Could not restore selected camera (%s)", type(exc).__name__)
        except (OSError, ValueError) as exc:
            logging.warning("Could not read camera registry (%s)", type(exc).__name__)
        SUPERVISOR_STOP.wait(10)


def read_log_tail(max_bytes=60000):
    log_path = Path(LOG_FILE)
    if not log_path.is_file():
        return ""

    with log_path.open("rb") as file:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(max(0, size - max_bytes))
        data = file.read()

    return data.decode("utf-8", errors="replace")


def media_folder_config(folder_key):
    return MEDIA_FOLDERS.get(folder_key)


def summarize_detection_decisions(metadata):
    """Display each distinct check once while retaining the full saved history."""
    return sorted({reason.strip() for reason in (metadata or {}).get('alert_decisions', [])
                   if isinstance(reason, str) and reason.strip()})


def list_saved_media(folder_key, limit=80):
    folder = media_folder_config(folder_key)
    if folder is None:
        return []

    folder_path = folder["path"]
    if not folder_path.is_dir():
        return []

    items = []
    for path in folder_path.iterdir():
        if not path.is_file() or path.suffix.lower() not in folder["extensions"]:
            continue

        try:
            stat = path.stat()
        except OSError:
            continue

        metadata = load_metadata(path) if folder_key in ('clips', 'object_clips') else None
        delivery_status = None
        if folder_key == 'object_clips':
            import object_alerts
            delivery_status = object_alerts.get_outbox().status(path)
        evidence = []
        for name in (metadata or {}).get('evidence', []):
            if isinstance(name, str) and Path(name).name == name and (OBJECT_OUTPUT_DIR / 'detection_frames' / name).is_file():
                evidence.append({'name': name, 'url': f'/media/detection_frames/{name}'})
        items.append(
            {
                "name": path.name,
                "detection_title": title_for_metadata(metadata) if folder_key == 'clips' else None,
                "delivery_status": delivery_status,
                "alert_reasons": sorted({str(event.get('reason')) for event in (metadata or {}).get('alert_events', []) if isinstance(event, dict) and event.get('reason')}),
                "alert_decisions": summarize_detection_decisions(metadata),
                "motion_associations": (metadata or {}).get('motion_associations', []),
                "crop_comparisons": (metadata or {}).get('crop_comparisons', []),
                "evidence": evidence,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "url": f"/media/{folder_key}/{path.name}",
                "type": folder["type"],
            }
        )

    items.sort(key=lambda item: item["modified"], reverse=True)
    return items[:limit]


def list_archived_media(folder_key, offset=0):
    kind = {"clips": "motion", "object_clips": "object", "detection_frames": "image"}[folder_key]
    result = list_archive(kind, offset)
    if result is None:
        return None
    items = []
    for row in result.get("items", []):
        try:
            wrapper = json.loads(row.get("MetadataJson") or "{}")
            metadata = wrapper.get("detections") or {}
            media_id = str(row["MediaId"])
            delivery_status = None
            if folder_key == 'object_clips':
                import object_alerts
                delivery_status = object_alerts.get_outbox().status(OBJECT_OUTPUT_DIR / row["Filename"])
            items.append({"name": row["Filename"], "delivery_status": delivery_status, "archive_id": media_id,
                "review": row.get("Review"), "review_note": row.get("ReviewNote"),
                "detection_title": title_for_metadata(metadata) if metadata else None,
                "alert_reasons": sorted({str(e.get("reason")) for e in metadata.get("alert_events", []) if isinstance(e, dict) and e.get("reason")}),
                "alert_decisions": summarize_detection_decisions(metadata),
                "motion_associations": metadata.get("motion_associations", []),
                "crop_comparisons": metadata.get("crop_comparisons", []),
                "evidence": [{"name": name, "url": "/archive-image/" + quote(name, safe="")} for name in metadata.get("evidence", []) if isinstance(name, str) and Path(name).name == name],
                "size": row.get("SizeBytes", 0), "modified": row.get("CreatedAt", ""),
                "url": "/archive-media/" + media_id, "type": MEDIA_FOLDERS[folder_key]["type"]})
        except (KeyError, TypeError, ValueError):
            logging.warning("Invalid archive record was skipped")
    return items


def sanitize_diagnostic(value):
    if isinstance(value, dict):
        return {key: sanitize_diagnostic(item) for key, item in value.items()
                if not any(word in str(key).lower() for word in ('password', 'token', 'secret', 'url', 'credential', 'settings'))}
    if isinstance(value, list):
        return [sanitize_diagnostic(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r'(?i)rtsp[s]?://[^\s]+', '[redacted stream]', value)
        value = re.sub(r'(?i)https?://[^\s]+', '[redacted link]', value)
        value = re.sub(r'(?i)(password|token|secret|sig|key)\s*[:=]\s*[^\s,;]+', r'\1=[redacted]', value)
        return value
    return value


def diagnostic_bundle():
    import object_alerts
    data = {'generated_at': datetime.now().astimezone().isoformat(), 'cameras': APP_STATE.worker_summaries(),
            'alerts': object_alerts.get_outbox().counts(),
            'archive': ARCHIVE_MANAGER.outbox.counts() if ARCHIVE_MANAGER else None, 'media': {}}
    for key in MEDIA_FOLDERS:
        local = list_saved_media(key, limit=30)
        remote = list_archived_media(key) if ARCHIVE_ENABLED else []
        data['media'][key] = {'local': local, 'archive': remote}
    payload = json.dumps(sanitize_diagnostic(data), indent=2, default=str).encode('utf-8')
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('diagnostics.json', payload)
    return output.getvalue()


def media_path_for_name(folder_key, filename):
    folder = media_folder_config(folder_key)
    if folder is None:
        return None

    filename = Path(filename).name
    media_path = (folder["path"] / filename).resolve()
    media_root = folder["path"].resolve()
    if media_root not in media_path.parents or not media_path.is_file():
        return None
    if media_path.suffix.lower() not in folder["extensions"]:
        return None
    return media_path


class SecurityCameraWebHandler(BaseHTTPRequestHandler):
    server_version = "SecurityCameraWeb/1.0"

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self.send_html(INDEX_HTML)
        if parsed.path == "/api/defaults":
            return self.send_json(
                {
                    "settings": default_settings(),
                    "default_motion_sensitivity": DEFAULT_MOTION_SENSITIVITY,
                }
            )
        if parsed.path == "/api/workers":
            return self.send_json({"workers": APP_STATE.worker_summaries()})
        if parsed.path == "/api/diagnostics/export":
            try:
                data = diagnostic_bundle()
            except (OSError, ValueError):
                return self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", 'attachment; filename="security-camera-diagnostics.zip"')
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/health":
            import shutil
            try:
                disk_free = shutil.disk_usage(OUTPUT_DIR if OUTPUT_DIR.exists() else Path.home()).free
                archive = ARCHIVE_MANAGER.outbox.counts() if ARCHIVE_MANAGER else None
                import object_alerts
                alerts = object_alerts.get_outbox().counts()
                selected = CAMERA_REGISTRY.read()
                cameras = APP_STATE.worker_summaries()
                active = {camera['source_id'] for camera in cameras}
                return self.send_json({"disk_free_bytes": disk_free, "cameras": cameras,
                    "selected_not_running": len(set(selected) - active), "archive_enabled": ARCHIVE_ENABLED, "archive": archive, "alerts": alerts})
            except (OSError, ValueError):
                return self.send_json({"error": "Health status unavailable"}, HTTPStatus.SERVICE_UNAVAILABLE)
        if parsed.path == "/api/clip-test":
            return self.send_json({"job": CLIP_REPLAY.snapshot()})
        if parsed.path.startswith('/replay-media/'):
            parts = unquote(parsed.path[len('/replay-media/'):]).split('/')
            if not parts or len(parts[0]) != 32 or not all(c in '0123456789abcdef' for c in parts[0]):
                return self.send_error(HTTPStatus.BAD_REQUEST)
            root = (CLIP_REPLAY.root / parts[0]).resolve()
            path = root.joinpath(*parts[1:]).resolve()
            if root not in path.parents or not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS | IMAGE_EXTENSIONS:
                return self.send_error(HTTPStatus.NOT_FOUND)
            return self.send_file_path(path)
        if parsed.path == "/api/logs":
            return self.send_json({"text": read_log_tail()})
        if parsed.path == "/api/media":
            query = parse_qs(parsed.query)
            folder_key = query.get("folder", ["clips"])[0]
            folder = media_folder_config(folder_key)
            if folder is None:
                return self.send_json({"error": "Unknown media folder"}, HTTPStatus.BAD_REQUEST)
            try:
                page = max(0, int(query.get("page", ["0"])[0]))
                if page > 10000:
                    raise ValueError("page out of range")
            except ValueError:
                return self.send_json({"error": "Invalid archive page"}, HTTPStatus.BAD_REQUEST)
            archived = list_archived_media(folder_key, page * 100) if ARCHIVE_ENABLED else []
            local_items = list_saved_media(folder_key) if page == 0 else []
            archived_names = {item['name'] for item in (archived or [])}
            combined = [item for item in local_items if item['name'] not in archived_names] + (archived or [])
            combined.sort(key=lambda item: item.get("modified", ""), reverse=True)
            return self.send_json(
                {
                    "folder": folder_key,
                    "label": folder["label"],
                    "archive_unavailable": ARCHIVE_ENABLED and archived is None,
                    "page": page, "has_more": bool(archived and len(archived) == 100),
                    "items": combined,
                    "folders": [
                        {"key": key, "label": config["label"], "type": config["type"]}
                        for key, config in MEDIA_FOLDERS.items()
                    ],
                }
            )
        if parsed.path == "/api/clips":
            return self.send_json({"clips": list_saved_media("clips")})
        if parsed.path.startswith("/archive-image/"):
            name = unquote(parsed.path[len("/archive-image/"):])
            if not name or Path(name).name != name:
                return self.send_error(HTTPStatus.BAD_REQUEST)
            local = media_path_for_name("detection_frames", name)
            if local:
                return self.send_media_file("detection_frames", name)
            result = archive_request("/archive/image-by-name/" + quote(name, safe="") + "/read")
            url = result.get("read_url") if result else None
            if not url or not url.startswith("https://"):
                return self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", url)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        if parsed.path.startswith("/archive-media/"):
            import uuid
            try:
                media_id = str(uuid.UUID(parsed.path[len("/archive-media/"):]))
            except ValueError:
                return self.send_error(HTTPStatus.BAD_REQUEST)
            result = archive_request(f"/archive/{media_id}/read")
            url = result.get("read_url") if result else None
            if not url or not url.startswith("https://"):
                return self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", url)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        if parsed.path.startswith("/media/"):
            media_path = parsed.path[len("/media/"):]
            folder_key, _separator, filename = media_path.partition("/")
            return self.send_media_file(folder_key, unquote(filename))
        if parsed.path.startswith("/clips/"):
            filename = unquote(parsed.path[len("/clips/"):])
            return self.send_media_file("clips", filename)
        if parsed.path.startswith("/preview/"):
            source_id = unquote(parsed.path[len("/preview/"):])
            return self.send_preview(source_id)
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/clip-test':
            return self.handle_clip_test(parsed)
        if parsed.path == "/api/archive/review":
            import uuid
            payload = self.read_json()
            try:
                media_id = str(uuid.UUID(payload.get("media_id", "")))
            except (ValueError, AttributeError):
                return self.send_json({"error": "Invalid media ID"}, HTTPStatus.BAD_REQUEST)
            result = archive_request(f"/archive/{media_id}/review", {"review": payload.get("review"), "note": payload.get("note")})
            if result is None:
                return self.send_json({"error": "Could not save review"}, HTTPStatus.SERVICE_UNAVAILABLE)
            return self.send_json(result)
        if parsed.path == "/api/scan":
            return self.handle_scan()
        if parsed.path == "/api/stream/start":
            return self.handle_stream_start()
        if parsed.path == "/api/stream/stop":
            return self.handle_stream_stop()
        if parsed.path == "/api/stream/exclusions":
            return self.handle_stream_exclusions()
        if parsed.path == "/api/stream/vehicle-area":
            return self.handle_stream_vehicle_area()
        if parsed.path == "/api/preview/set":
            return self.handle_preview_set()
        self.send_error(HTTPStatus.NOT_FOUND)

    def handle_clip_test(self, parsed):
        import shutil
        params = parse_qs(parsed.query)
        source_id = params.get('camera', [''])[0]
        worker = APP_STATE.get_worker(source_id)
        if worker is None:
            return self.send_json({'error': 'Choose a running camera.'}, HTTPStatus.BAD_REQUEST)
        if OBJECT_DETECTOR is None or OBJECT_DETECTOR.health() not in ('ready', 'degraded'):
            return self.send_json({'error': 'Object detection is not ready.'}, HTTPStatus.SERVICE_UNAVAILABLE)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 250 * 1024 * 1024:
                raise ValueError('Choose a video no larger than 250 MB.')
            suffix = Path(params.get('filename', [''])[0]).suffix.lower()
            CLIP_REPLAY.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            if shutil.disk_usage(CLIP_REPLAY.root).free < length + 2048 * 1024 * 1024:
                raise ValueError('Not enough free disk space for this test.')
            path, recorded_at = CLIP_REPLAY.reserve(worker, params.get('recorded_at', [''])[0], suffix)
        except (ValueError, OSError) as exc:
            return self.send_json({'error': str(exc)}, HTTPStatus.BAD_REQUEST)
        try:
            remaining = length
            self.connection.settimeout(30)
            with path.open('xb') as output:
                os.chmod(path, 0o600)
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise OSError('Upload ended early')
                    output.write(chunk)
                    remaining -= len(chunk)
            CLIP_REPLAY.start(path, recorded_at, worker, OBJECT_DETECTOR, worker.motion_algorithm)
            return self.send_json({'job': CLIP_REPLAY.snapshot()}, HTTPStatus.ACCEPTED)
        except (OSError, ValueError) as exc:
            path.unlink(missing_ok=True)
            CLIP_REPLAY.fail('Upload failed. Please try again.')
            logging.warning('Clip test upload failed (%s)', type(exc).__name__)
            return self.send_json({'error': 'Upload failed.'}, HTTPStatus.BAD_REQUEST)

    def handle_scan(self):
        payload = self.read_json()
        settings = payload.get("settings", {})
        messages = []

        def progress(message):
            messages.append(message)

        streams = discover_nvr_streams(settings, can_open_rtsp_channel, progress_callback=progress)
        APP_STATE.scan_messages = messages
        self.send_json({"streams": [stream_to_json(stream) for stream in streams], "messages": messages})

    def handle_stream_start(self):
        try:
            worker = start_camera(self.read_json())
        except ValueError as exc:
            return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except OSError:
            logging.exception("Could not persist camera selection")
            return self.send_json({"error": "Could not save camera selection."}, HTTPStatus.INTERNAL_SERVER_ERROR)
        self.send_json({"worker": {"source_id": worker.source_id, "label": worker.label}})

    def handle_stream_exclusions(self):
        payload = self.read_json()
        source_id = str(payload.get("source_id", "")).strip()
        areas = payload.get("excluded_areas")
        worker = APP_STATE.get_worker(source_id)
        if worker is None:
            return self.send_json({"error": "Camera is not running."}, HTTPStatus.NOT_FOUND)
        try:
            FrameDifferenceMotion.validate_exclusions(areas)
            save_camera_exclusions(source_id, areas)
            worker.set_exclusions(areas)
        except (ValueError, OSError) as exc:
            logging.warning("Could not save motion exclusions: %s", type(exc).__name__)
            return self.send_json({"error": "Could not save exclusion areas; check the rectangles and storage."}, HTTPStatus.BAD_REQUEST)
        self.send_json({"source_id": source_id, "excluded_areas": worker.current_exclusions()})

    def handle_stream_vehicle_area(self):
        payload = self.read_json()
        source_id = str(payload.get('source_id', '')).strip()
        area = payload.get('vehicle_area')
        worker = APP_STATE.get_worker(source_id)
        if worker is None:
            return self.send_json({'error': 'Camera is not running.'}, HTTPStatus.NOT_FOUND)
        if OBJECT_DETECTOR and OBJECT_DETECTOR.remote_shadow and not isinstance(OBJECT_DETECTOR.remote_shadow, LocalVehicleVerifier):
            return self.send_json({'error': 'Custom vehicle areas require the Pi-only verifier.'}, HTTPStatus.CONFLICT)
        try:
            FrameDifferenceMotion.validate_exclusions([area])
            save_vehicle_area(source_id, area)
            if OBJECT_DETECTOR and isinstance(OBJECT_DETECTOR.remote_shadow, LocalVehicleVerifier):
                OBJECT_DETECTOR.remote_shadow.set_crop(source_id, area)
            worker.set_vehicle_area(area)
        except (ValueError, OSError):
            logging.exception('Could not save vehicle area')
            return self.send_json({'error': 'Could not save vehicle area; check the rectangle and storage.'}, HTTPStatus.BAD_REQUEST)
        self.send_json({'source_id': source_id, 'vehicle_area': worker.current_vehicle_area()})

    def handle_stream_stop(self):
        payload = self.read_json()
        source_id = str(payload.get("source_id", "")).strip()
        with APP_STATE.lock:
            try:
                CAMERA_REGISTRY.remove(source_id)
            except (OSError, ValueError):
                logging.exception("Could not update camera selection")
                return self.send_json({"error": "Could not save camera selection."}, HTTPStatus.INTERNAL_SERVER_ERROR)
            worker = APP_STATE.get_worker(source_id)
            if worker is None:
                return self.send_json({"stopped": False})
            worker.status = "stopping"
            worker.stop_event.set()

        worker.status = "stopping"
        worker.stop_event.set()
        with worker.condition:
            worker.condition.notify_all()
        self.send_json({"stopped": True})

    def handle_preview_set(self):
        payload = self.read_json()
        source_id = str(payload.get("source_id", "")).strip()
        enabled = bool(payload.get("enabled"))
        worker = APP_STATE.get_worker(source_id)
        if worker is None:
            return self.send_json({"error": "Worker is not running"}, HTTPStatus.NOT_FOUND)

        worker.preview_enabled = enabled
        if not enabled:
            with worker.condition:
                worker.frame = None
                worker.frame_version += 1
                worker.condition.notify_all()
        logging.info("Preview %s for %s", "enabled" if enabled else "disabled", worker.label)
        self.send_json({"source_id": source_id, "preview_enabled": worker.preview_enabled})

    def send_preview(self, source_id):
        worker = APP_STATE.get_worker(source_id)
        if worker is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Preview source is not running")
            return

        boundary = "frame"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        last_version = -1
        try:
            while not worker.stop_event.is_set():
                with worker.condition:
                    worker.condition.wait_for(
                        lambda: worker.frame_version != last_version or worker.stop_event.is_set(),
                        timeout=5,
                    )
                    frame = worker.frame
                    last_version = worker.frame_version

                if frame is None:
                    continue

                self.wfile.write(f"--{boundary}\r\n".encode("ascii"))
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return

    def send_media_file(self, folder_key, filename):
        media_path = media_path_for_name(folder_key, filename)
        if media_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Media file not found")
            return

        return self.send_file_path(media_path)

    def send_file_path(self, media_path):
        file_size = media_path.stat().st_size
        content_type = mimetypes.guess_type(media_path.name)[0] or "application/octet-stream"
        range_header = self.headers.get("Range", "")
        start = 0
        end = file_size - 1
        status = HTTPStatus.OK

        if range_header.startswith("bytes="):
            range_value = range_header.split("=", 1)[1].split(",", 1)[0]
            start_text, _separator, end_text = range_value.partition("-")
            try:
                if start_text:
                    start = int(start_text)
                if end_text:
                    end = int(end_text)
            except ValueError:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return

            if start < 0 or end >= file_size or start > end:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            status = HTTPStatus.PARTIAL_CONTENT

        content_length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(content_length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        with media_path.open("rb") as file:
            file.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = file.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def read_json(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        return json.loads(self.rfile.read(content_length).decode("utf-8"))

    def send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Security Camera Browser Console</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #eef1f4;
      --panel: #ffffff;
      --line: #c8d0d8;
      --text: #18212b;
      --muted: #5c6875;
      --accent: #126a6f;
      --accent-dark: #0d5155;
      --danger: #a33434;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      gap: 14px;
      padding: 14px;
    }
    section {
      min-width: 0;
      max-width: 100%;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 15px;
      font-weight: 650;
    }
    label {
      display: block;
      margin: 10px 0 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    input, select {
      width: 100%;
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 9px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }
    input[type="checkbox"] {
      width: auto;
      min-height: 0;
    }
    button {
      min-height: 34px;
      border: 1px solid var(--accent);
      border-radius: 6px;
      padding: 7px 10px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }
    button.secondary {
      background: #fff;
      color: var(--accent-dark);
    }
    button.danger {
      border-color: var(--danger);
      background: var(--danger);
    }
    button:disabled {
      opacity: 0.55;
      cursor: wait;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .checks {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 7px 10px;
      margin-top: 6px;
    }
    .checks label {
      display: flex;
      align-items: center;
      gap: 7px;
      margin: 0;
      color: var(--text);
      font-size: 13px;
      font-weight: 500;
    }
    .stack {
      display: grid;
      min-width: 0;
      gap: 14px;
    }
    .status {
      min-height: 38px;
      color: var(--muted);
      white-space: pre-wrap;
    }
    .stream-list {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 230px), 1fr));
      gap: 8px;
    }
    .preview-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr));
      gap: 10px;
    }
    .preview-statuses { display: flex; flex-wrap: wrap; gap: 5px; margin: 0 0 8px; }
    .preview-badge { display: inline-flex; align-items: center; gap: 5px; min-height: 22px; padding: 2px 7px; border: 1px solid #dce4e9; border-radius: 5px; background: #f2f6f8; color: #52606d; font-size: 11px; line-height: 1.25; font-weight: 600; }
    .recording-indicator.active { border-color: #f3b9b3; background: #fff1ef; color: #b42318; }
    .recording-indicator.active::before { content: ""; width: 7px; height: 7px; flex: none; border-radius: 50%; background: #d92d20; }
    .detection-indicator.active { border-color: #a6ddc2; background: #eaf8f0; color: #17643b; }
    @media (min-width: 1320px) {
      .preview-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
    .stream-item, .preview-item {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      background: #fbfcfd;
    }
    .stream-top {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }
    .stream-controls {
      display: grid;
      grid-template-columns: 76px minmax(74px, auto);
      gap: 8px;
      align-items: end;
    }
    .stream-controls input,
    .stream-controls button {
      min-height: 30px;
      padding-top: 5px;
      padding-bottom: 5px;
    }
    .stream-title {
      font-weight: 650;
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .stream-meta {
      margin-top: 3px;
      color: var(--muted);
      font-size: 11px;
      overflow-wrap: anywhere;
    }
    .preview-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 6px 10px;
      margin-bottom: 6px;
    }
    .preview-title {
      min-width: 0;
      font-weight: 650;
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .preview-actions { display: flex; flex-wrap: wrap; gap: 4px; margin: 0; }
    .preview-actions button { min-height: 27px; padding: 4px 7px; font-size: 11px; line-height: 1.2; }
    .preview-item img {
      display: block;
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      background: #101820;
      border-radius: 6px;
    }
    dialog.zone-dialog { width: min(calc(100vw - 24px), 1000px); max-width: calc(100vw - 24px); max-height: 94vh; border: 1px solid var(--line); border-radius: 10px; padding: 16px; }
    dialog.zone-dialog::backdrop { background: rgba(0,0,0,.65); }
    .zone-stage { position: relative; width: fit-content; max-width: 100%; margin: auto; line-height: 0; background: #101820; }
    .zone-stage img { width: auto; height: auto; max-width: 100%; max-height: calc(100vh - 240px); display: block; }
    .zone-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; cursor: crosshair; touch-action: none; }
    .zone-help { color: var(--muted); margin: 8px 0; }
    .preview-placeholder {
      display: grid;
      place-items: center;
      width: 100%;
      aspect-ratio: 4 / 3;
      border: 1px solid #1f2933;
      border-radius: 6px;
      background: #101820;
      color: #d9e2ec;
      font-size: 12px;
      font-weight: 650;
      text-align: center;
      padding: 12px;
    }
    .video-controls {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 82px;
      gap: 10px;
      align-items: end;
    }
    .section-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 12px;
    }
    .section-head h2 {
      margin: 0;
    }
    .section-head button {
      min-height: 30px;
      padding: 5px 9px;
    }
    .clip-list {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr));
      gap: 10px;
      max-height: 420px;
      overflow: auto;
      padding-right: 4px;
    }
    .clip-item {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      background: #fbfcfd;
    }
    .clip-title {
      font-weight: 650;
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .clip-meta {
      margin: 4px 0 8px;
      color: var(--muted);
      font-size: 11px;
      overflow-wrap: anywhere;
    }
    .clip-item video {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #101820;
      border-radius: 6px;
    }
    .clip-item img {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #101820;
      border-radius: 6px;
    }
    .video-placeholder {
      display: grid;
      place-items: center;
      width: 100%;
      aspect-ratio: 16 / 9;
      border: 1px solid #1f2933;
      border-radius: 6px;
      background: #101820;
      color: #d9e2ec;
      font-size: 12px;
      font-weight: 650;
    }
    .clip-actions {
      display: flex;
      flex-wrap: wrap;
      min-width: 0;
      gap: 8px;
      margin-top: 8px;
    }
    .clip-actions a {
      max-width: 100%;
      overflow-wrap: anywhere;
      color: var(--accent-dark);
      font-size: 12px;
      font-weight: 650;
      text-decoration: none;
    }
    .evidence-details { margin-top: 8px; min-width: 0; }
    .evidence-details summary { color: var(--accent-dark); cursor: pointer; font-size: 12px; font-weight: 650; }
    .media-more { display: flex; justify-content: center; margin-top: 12px; }
    .media-more button { min-height: 30px; padding: 5px 14px; }
    .media-more[hidden] { display: none; }
    .media-tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 0 0 10px;
    }
    .media-tab {
      min-height: 30px;
      border-color: var(--line);
      background: #fff;
      color: var(--text);
      padding: 5px 9px;
    }
    .media-tab.active {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }
    .logs-section {
      margin: 0 14px 14px;
    }
    .logs-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 10px;
    }
    .logs-head h2 {
      margin: 0;
    }
    .logs-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .logs-actions label {
      display: flex;
      align-items: center;
      gap: 6px;
      margin: 0;
      color: var(--text);
      font-size: 13px;
      font-weight: 500;
    }
    .logs-actions button {
      min-height: 30px;
      padding: 5px 9px;
    }
    .log-pane {
      width: 100%;
      height: 230px;
      margin: 0;
      overflow: auto;
      border: 1px solid #1f2933;
      border-radius: 6px;
      padding: 10px;
      background: #101820;
      color: #d9e2ec;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      .stream-top { grid-template-columns: 1fr; }
      .row, .video-controls { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Motion Detection Console</h1>
    <div id="connectionStatus">Idle</div>
    <div id="healthStatus" role="status"></div>
    <a href="/api/diagnostics/export">Download diagnostics</a>
  </header>

  <main>
    <div class="stack">
      <section>
        <h2>Connection</h2>
        <label for="username">Username</label>
        <input id="username" autocomplete="username">
        <label for="password">Password</label>
        <input id="password" type="password" autocomplete="current-password">
        <div class="row">
          <div>
            <label for="ipAddress">DVR/NVR IP</label>
            <input id="ipAddress">
          </div>
          <div>
            <label for="port">RTSP Port</label>
            <input id="port">
          </div>
        </div>
        <label for="rtspTemplate">RTSP Path Template</label>
        <input id="rtspTemplate">
        <div class="actions">
          <button id="scanButton">Connect</button>
          <button id="manualButton" class="secondary">Add Manual Channel</button>
        </div>
        <label for="manualChannel">Manual Channel</label>
        <input id="manualChannel" placeholder="101">
      </section>

    </div>

    <div class="stack">
      <section>
        <h2>Streams</h2>
        <div id="scanStatus" class="status">Enter NVR details, then connect to discover streams.</div>
        <div id="streamList" class="stream-list"></div>
      </section>

      <section>
        <h2>Previews</h2>
        <div id="previewGrid" class="preview-grid"></div>
      </section>

      <section>
        <h2>Test a clip</h2>
        <p class="status">Replays the live motion, object and scenario stages using the selected camera settings. No WhatsApp is sent. Tests run at recording speed and may slow live detection. Test files stay on this Pi, separate from the live archive.</p>
        <label for="clipTestCamera">Camera settings</label>
        <select id="clipTestCamera"><option value="">Choose a running camera</option></select>
        <label for="clipTestTime">When was the clip recorded? (South Africa time)</label>
        <input id="clipTestTime" type="datetime-local" step="1">
        <label for="clipTestFile">Video (up to 250 MB)</label>
        <input id="clipTestFile" type="file" accept=".mp4,.mov,.avi,.mkv,.m4v,video/*">
        <div class="actions"><button id="clipTestStart" type="button">Run clip test</button></div>
        <div id="clipTestStatus" class="status" role="status">No clip test running.</div>
        <div id="clipTestResults"></div>
      </section>

      <section>
        <div class="section-head">
          <h2>Saved Media</h2>
          <button id="refreshClipsButton" class="secondary">Refresh</button>
        </div>
        <div id="mediaTabs" class="media-tabs"></div>
        <div id="clipList" class="clip-list"></div>
        <div id="mediaMore" class="media-more" hidden><button id="loadMoreMedia" class="secondary" type="button">Show more</button></div>
      </section>
    </div>
  </main>
  <dialog id="zoneDialog" class="zone-dialog" aria-labelledby="zoneTitle">
    <h2 id="zoneTitle">Draw excluded areas</h2>
    <p id="zoneHelp" class="zone-help">Drag on the image to add rectangles. All motion inside a shaded area is ignored, including people and vehicles.</p>
    <div class="zone-stage"><img id="zoneImage" alt="Live camera view"><canvas id="zoneCanvas" width="1000" height="1000" aria-label="Draw exclusion rectangles on the live camera image"></canvas></div>
    <div id="zoneStatus" class="status" role="status"></div>
    <div class="actions"><button id="zoneUndo" type="button" class="secondary">Undo</button><button id="zoneRemove" type="button" class="secondary">Remove selected</button><button id="zoneClear" type="button" class="secondary">Clear all</button><button id="zoneCancel" type="button" class="secondary">Cancel</button><button id="zoneSave" type="button">Save areas</button></div>
  </dialog>

  <section class="logs-section">
    <div class="logs-head">
      <h2>App Logs</h2>
      <div class="logs-actions">
        <label><input id="autoScrollLogs" type="checkbox" checked> Auto-scroll</label>
        <button id="refreshLogsButton" class="secondary">Refresh</button>
      </div>
    </div>
    <pre id="appLogs" class="log-pane">Loading logs...</pre>
  </section>

  <script>
    const STORAGE_KEY = "securityCameraWebState";

    const state = {
      streams: [],
      sensitivities: {},
      algorithms: {},
      previews: new Map(),
      mediaItems: [],
      mediaPage: 0,
      mediaHasMore: false,
      mediaFolders: [
        { key: "clips", label: "Clips", type: "video" }
      ],
      activeMediaFolder: "clips",
      defaults: null
    };

    const $ = (id) => document.getElementById(id);

    function setStatus(message) {
      $("connectionStatus").textContent = message;
    }

    function readSavedState() {
      try {
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
      } catch (_error) {
        return {};
      }
    }

    function saveState() {
      const saved = {
        settings: settings(),
        streams: state.streams,
        sensitivities: state.sensitivities,
        algorithms: state.algorithms
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(saved));
    }

    function applySettings(savedSettings) {
      if (!savedSettings) return;
      $("username").value = savedSettings.username || "";
      $("password").value = savedSettings.password || "";
      $("ipAddress").value = savedSettings.ip_address || "";
      $("port").value = savedSettings.port || "554";
      $("rtspTemplate").value = savedSettings.rtsp_path_template || "/Streaming/Channels/{channel}";
    }

    function settings() {
      return {
        username: $("username").value.trim(),
        password: $("password").value,
        ip_address: $("ipAddress").value.trim(),
        port: $("port").value.trim(),
        rtsp_path_template: $("rtspTemplate").value.trim()
      };
    }

    function renderStreams() {
      const list = $("streamList");
      if (!state.streams.length) {
        list.innerHTML = "";
        return;
      }

      list.innerHTML = state.streams.map((stream, index) => `
        <div class="stream-item">
          <div class="stream-top">
            <div>
              <div class="stream-title">${escapeHtml(compactStreamLabel(stream.label))}</div>
              <div class="stream-meta">${escapeHtml(stream.stream_id)}</div>
            </div>
            <div class="stream-controls">
              <input id="sensitivity-${index}" data-sensitivity-id="${escapeAttribute(stream.stream_id)}" type="number" min="1" max="10" value="${state.sensitivities[stream.stream_id] || 5}" title="Sensitivity">
              <select id="algorithm-${index}" data-algorithm-id="${escapeAttribute(stream.stream_id)}" title="Motion detection method">
                <option value="difference" ${(state.algorithms[stream.stream_id] || "difference") === "difference" ? "selected" : ""}>Frame difference (test)</option>
                <option value="background" ${state.algorithms[stream.stream_id] === "background" ? "selected" : ""}>Original background</option>
              </select>
              <button data-start-index="${index}">Start</button>
            </div>
          </div>
        </div>
      `).join("");
    }

    function addPreview(sourceId, label, previewEnabled = true, excludedAreas = [], recording = false, status = "", objectRecording = false, objectStatus = "disabled", sensitivity = null, vehicleArea = [0,0,1,1], objectProcessing = false, detectedObjects = []) {
      const existing = state.previews.get(sourceId);
      if (existing) {
        existing.label = label;
        existing.previewEnabled = previewEnabled;
        existing.excludedAreas = excludedAreas;
        existing.recording = recording;
        existing.status = status;
        existing.objectRecording = objectRecording;
        existing.objectStatus = objectStatus;
        existing.sensitivity = sensitivity;
        existing.vehicleArea = vehicleArea;
        existing.objectProcessing = objectProcessing;
        existing.detectedObjects = detectedObjects;
      } else {
        state.previews.set(sourceId, { sourceId, label, previewEnabled, excludedAreas, recording, status, objectRecording, objectStatus, sensitivity, vehicleArea, objectProcessing, detectedObjects });
      }
      renderPreviews();
    }

    function removePreview(sourceId) {
      state.previews.delete(sourceId);
      renderPreviews();
    }

    function renderPreviews() {
      const grid = $("previewGrid");
      if (!state.previews.size) {
        grid.innerHTML = `<div class="status">No active previews.</div>`;
        return;
      }

      grid.innerHTML = Array.from(state.previews.values()).map((preview) => `
        <div class="preview-item">
          <div class="preview-head">
            <div class="preview-title">${escapeHtml(preview.label)}</div>
            <div class="preview-actions">
              <button class="secondary" data-preview-id="${encodeURIComponent(preview.sourceId)}" data-preview-enabled="${preview.previewEnabled ? "0" : "1"}">
                ${preview.previewEnabled ? "Hide Preview" : "Show Preview"}
              </button>
              ${preview.previewEnabled ? `<button class="secondary" data-zone-id="${encodeURIComponent(preview.sourceId)}">Excluded areas</button>` : ""}
              ${preview.previewEnabled ? `<button class="secondary" data-vehicle-zone-id="${encodeURIComponent(preview.sourceId)}">Vehicle area</button>` : ""}
              <button class="danger" data-stop-id="${encodeURIComponent(preview.sourceId)}">Stop</button>
            </div>
          </div>
          <div class="preview-statuses">
            <span class="preview-badge recording-indicator ${preview.recording ? "active" : ""}" role="status">${preview.recording ? "Motion: recording" : (preview.status && preview.status.toLowerCase().includes("paused") ? escapeHtml(preview.status) : "Motion: idle")}</span>
            <span class="preview-badge">Sensitivity ${preview.sensitivity == null ? "starting" : `${escapeHtml(String(preview.sensitivity))}/10`}</span>
            <span class="preview-badge recording-indicator ${preview.objectRecording ? "active" : ""}" role="status">${preview.objectRecording ? "Object: recording" : (preview.objectStatus === "ready" ? "Object: idle" : `Object detection: ${escapeHtml(preview.objectStatus)}`)}</span>
            <span class="preview-badge detection-indicator ${preview.detectedObjects.length ? "active" : ""}" role="status">${preview.detectedObjects.length ? `Detected: ${escapeHtml(preview.detectedObjects.join(", "))}` : (preview.objectProcessing ? "Objects: checking" : "Objects: none recent")}</span>
          </div>
          ${preview.previewEnabled
            ? `<img src="/preview/${encodeURIComponent(preview.sourceId)}" alt="">`
            : `<div class="preview-placeholder">Preview disabled<br>Detection is still running.</div>`}
        </div>
      `).join("");
    }

    function renderMediaTabs() {
      $("mediaTabs").innerHTML = state.mediaFolders.map((folder) => `
        <button class="media-tab ${folder.key === state.activeMediaFolder ? "active" : ""}" data-media-folder="${escapeAttribute(folder.key)}">
          ${escapeHtml(folder.label)}
        </button>
      `).join("");
    }

    function motionGateSummary(comparisons) {
      const checks = Array.isArray(comparisons) ? comparisons : [];
      if (!checks.length) return '';
      const errors = checks.filter(c => c.error).length;
      const passed = checks.filter(c => (c.moving_gate || []).some(d => d.qualified));
      const labels = [...new Set(passed.flatMap(c => (c.moving_gate || [])
        .filter(d => d.qualified).map(d => d.label).filter(Boolean)))];
      const result = passed.length ? `${passed.length} passed${labels.length ? ` (${labels.join(', ')})` : ''}` : 'none passed';
      return `${checks.length} recent ${checks.length === 1 ? 'check' : 'checks'} · ${result}${errors ? ` · ${errors} ${errors === 1 ? 'error' : 'errors'}` : ''}`;
    }

    function renderMedia() {
      const list = $("clipList");
      renderMediaTabs();
      if (!state.mediaItems.length) {
        list.innerHTML = `<div class="status">No saved media found.</div>`;
        return;
      }

      list.innerHTML = state.mediaItems.map((item) => `
        <div class="clip-item">
          <div class="clip-title">${escapeHtml(item.name)}${item.detection_title ? ` — ${escapeHtml(item.detection_title)}` : ''}</div>
          <div class="clip-meta">${escapeHtml(formatDateTime(item.modified))} · ${escapeHtml(formatBytes(item.size))}</div>
          ${item.delivery_status ? `<div class="clip-meta">WhatsApp queue: ${escapeHtml(item.delivery_status)}${item.delivery_status === 'accepted' ? ' (provider accepted; delivery not confirmed)' : ''}</div>` : ''}
          ${(item.alert_reasons || []).length ? `<div class="clip-meta">Alert eligible: ${escapeHtml(item.alert_reasons.join('; '))}</div>` : ''}
          ${(item.alert_decisions || []).length ? `<div class="clip-meta">Detection checks: ${escapeHtml(item.alert_decisions.join('; '))}</div>` : ''}
          ${(item.motion_associations || []).length ? `<div class="clip-meta">Motion association (test only): ${escapeHtml([...new Set(item.motion_associations.map(a => `${a.label}: ${a.status}`))].join('; '))}</div>` : ''}
          ${(item.crop_comparisons || []).length ? `<div class="clip-meta">Motion gate: ${escapeHtml(motionGateSummary(item.crop_comparisons))}</div>` : ''}
          ${mediaPreviewHtml(item)}
          ${item.archive_id ? `<div class="clip-meta">Cloud archive${item.review ? ` · Review: ${escapeHtml(item.review)}` : ''}</div>
          <div class="clip-actions"><button data-review-id="${escapeAttribute(item.archive_id)}" data-review="correct">Correct</button>
          <button data-review-id="${escapeAttribute(item.archive_id)}" data-review="false_alarm">False alarm</button>
          <button data-review-id="${escapeAttribute(item.archive_id)}" data-review="missed">Missed</button></div>` : ''}
          <div class="clip-actions">
            <a href="${escapeAttribute(item.url)}" download>Download</a>
            <a href="${escapeAttribute(item.url)}" target="_blank" rel="noreferrer">Open</a>
          </div>
          ${(item.evidence || []).length ? `<details class="evidence-details"><summary>Detection images (${item.evidence.length})</summary><div class="clip-actions">${item.evidence.map((frame, index) => `<a href="${escapeAttribute(frame.url)}" target="_blank" rel="noreferrer">Detection image ${index + 1}</a>`).join('')}</div></details>` : ''}
        </div>
      `).join("");
    }

    function mediaPreviewHtml(item) {
      if (item.type === "image") {
        return `<a href="${escapeAttribute(item.url)}" target="_blank" rel="noreferrer"><img src="${escapeAttribute(item.url)}" loading="lazy" alt=""></a>`;
      }
      return `<a class="video-placeholder" href="${escapeAttribute(item.url)}" target="_blank" rel="noreferrer">Open Video</a>`;
    }

    function compactStreamLabel(label) {
      return String(label || "").replace(/\s+\([^)]*\{[^)]*\}\)\s*$/, "");
    }

    function formatBytes(size) {
      const value = Number(size || 0);
      if (value < 1024) return `${value} B`;
      if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
      if (value < 1024 * 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MB`;
      return `${(value / 1024 / 1024 / 1024).toFixed(1)} GB`;
    }

    function formatDateTime(value) {
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return value || "";
      return date.toLocaleString();
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[char]));
    }

    function escapeAttribute(value) {
      return escapeHtml(value);
    }

    async function jsonFetch(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || response.statusText);
      return data;
    }

    async function loadDefaults() {
      const response = await fetch("/api/defaults");
      const defaults = await response.json();
      const saved = readSavedState();
      state.defaults = defaults;
      applySettings(defaults.settings);
      applySettings(saved.settings);
      if (Array.isArray(saved.streams)) state.streams = saved.streams;
      state.sensitivities = saved.sensitivities || {};
      state.algorithms = saved.algorithms || {};
      renderStreams();
      renderPreviews();
      await pollWorkers();
    }

    async function scanStreams() {
      $("scanButton").disabled = true;
      $("scanStatus").textContent = "Connecting: trying ONVIF discovery, then RTSP probing...";
      setStatus("Scanning");
      try {
        const data = await jsonFetch("/api/scan", { settings: settings() });
        state.streams = data.streams;
        state.sensitivities = {};
        $("scanStatus").textContent = data.messages.join("\n") || `Found ${data.streams.length} stream(s).`;
        if (!data.streams.length) {
          $("scanStatus").textContent += "\nNo streams found. Add a manual channel if you know the channel number.";
        }
        renderStreams();
        saveState();
        setStatus("Ready");
      } catch (error) {
        $("scanStatus").textContent = error.message;
        setStatus("Error");
      } finally {
        $("scanButton").disabled = false;
      }
    }

    function addManualChannel() {
      const channel = $("manualChannel").value.trim();
      if (!channel) {
        $("scanStatus").textContent = "Enter a manual channel first.";
        return;
      }
      const label = `Manual Channel (${channel})`;
      state.streams.push({
        label,
        stream_id: `manual:${channel}`,
        rtsp_url: null,
        rtsp_path_template: $("rtspTemplate").value.trim(),
        channel,
        source: "manual"
      });
      $("manualChannel").value = "";
      $("scanStatus").textContent = "Manual channel added. Start it from the stream list.";
      renderStreams();
      saveState();
    }

    async function startStream(index) {
      const stream = state.streams[index];
      const sensitivity = Number($(`sensitivity-${index}`).value || 5);
      state.sensitivities[stream.stream_id] = sensitivity;
      saveState();
      try {
        state.algorithms[stream.stream_id] = $(`algorithm-${index}`).value;
        saveState();
        const data = await jsonFetch("/api/stream/start", {
          settings: settings(),
          stream,
          sensitivity,
          motion_algorithm: state.algorithms[stream.stream_id]
        });
        addPreview(data.worker.source_id, data.worker.label);
        setStatus("Running");
      } catch (error) {
        $("scanStatus").textContent = error.message;
        setStatus("Error");
      }
    }

    let zoneCamera = null;
    let zoneAreas = [];
    let zoneStart = null;
    let zoneCurrent = null;
    let zoneMode = 'exclusions';
    let zoneSelected = -1;
    let zoneAction = null;
    let zoneHistory = [];

    function zoneSnapshot() { return zoneAreas.map((area) => [...area]); }
    function zoneRemember() { zoneHistory.push(zoneSnapshot()); if (zoneHistory.length > 30) zoneHistory.shift(); }
    const zoneClamp = (value, low, high) => Math.max(low, Math.min(high, value));
    function zoneHandles(area) {
      const [x,y,w,h] = area;
      return [['nw',x,y],['n',x+w/2,y],['ne',x+w,y],['e',x+w,y+h/2],
              ['se',x+w,y+h],['s',x+w/2,y+h],['sw',x,y+h],['w',x,y+h/2]];
    }
    function zoneHit(point) {
      const canvas = $('zoneCanvas'), rect = canvas.getBoundingClientRect();
      if (zoneSelected >= 0 && zoneAreas[zoneSelected]) {
        for (const [name,x,y] of zoneHandles(zoneAreas[zoneSelected])) {
          if (Math.abs(point[0]-x)*rect.width <= 10 && Math.abs(point[1]-y)*rect.height <= 10)
            return {type:'resize', index:zoneSelected, handle:name};
        }
      }
      for (let i=zoneAreas.length-1; i>=0; i--) {
        const [x,y,w,h] = zoneAreas[i];
        if (point[0]>=x && point[0]<=x+w && point[1]>=y && point[1]<=y+h) return {type:'move', index:i};
      }
      return {type:'draw'};
    }

    function applyZoneAction(point) {
      if (!zoneAction || zoneAction.type === 'draw') return;
      const a = zoneAction, [x,y,w,h] = a.original;
      if (a.type === 'move') {
        const nx = zoneClamp(x + point[0] - a.start[0], 0, 1-w);
        const ny = zoneClamp(y + point[1] - a.start[1], 0, 1-h);
        zoneAreas[a.index] = [nx, ny, Math.min(w,1-nx), Math.min(h,1-ny)];
      } else {
        let left=x, top=y, right=x+w, bottom=y+h;
        if (a.handle.includes('w')) left=zoneClamp(point[0], 0, right-.005);
        if (a.handle.includes('e')) right=zoneClamp(point[0], left+.005, 1);
        if (a.handle.includes('n')) top=zoneClamp(point[1], 0, bottom-.005);
        if (a.handle.includes('s')) bottom=zoneClamp(point[1], top+.005, 1);
        zoneAreas[a.index] = [left, top, Math.min(right-left,1-left), Math.min(bottom-top,1-top)];
      }
    }

    function zonePoint(event) {
      const bounds = $("zoneCanvas").getBoundingClientRect();
      return [Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
              Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height))];
    }

    function drawZones() {
      const canvas = $("zoneCanvas"), ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const draw = (area, draft = false) => {
        const [x,y,w,h] = area;
        ctx.fillStyle = draft ? "rgba(255, 190, 30, .35)" : "rgba(225, 48, 48, .35)";
        ctx.strokeStyle = draft ? "#ffbd1a" : "#ff4747";
        ctx.lineWidth = 4;
        ctx.fillRect(x*canvas.width, y*canvas.height, w*canvas.width, h*canvas.height);
        ctx.strokeRect(x*canvas.width, y*canvas.height, w*canvas.width, h*canvas.height);
      };
      zoneAreas.forEach((area) => draw(area));
      if (zoneSelected >= 0 && zoneAreas[zoneSelected]) {
        ctx.fillStyle = '#fff'; ctx.strokeStyle = '#b42318'; ctx.lineWidth = 3;
        for (const [,x,y] of zoneHandles(zoneAreas[zoneSelected])) {
          const px=x*canvas.width, py=y*canvas.height;
          ctx.fillRect(px-8,py-8,16,16); ctx.strokeRect(px-8,py-8,16,16);
        }
      }
      if (zoneStart && zoneCurrent) draw(zoneRectangle(zoneStart, zoneCurrent), true);
      $("zoneStatus").textContent = zoneMode === 'vehicle' ? (zoneAreas.length ? 'Vehicle area selected. Save to apply.' : 'No area drawn: the whole camera view will be checked.') : `${zoneAreas.length} area(s) selected. Save to apply.`;
    }

    function zoneRectangle(a, b) {
      const x = Math.min(a[0],b[0]), y = Math.min(a[1],b[1]);
      return [x, y, Math.min(1-x, Math.abs(a[0]-b[0])), Math.min(1-y, Math.abs(a[1]-b[1]))];
    }

    function openZoneEditor(sourceId, mode = 'exclusions') {
      const preview = state.previews.get(sourceId);
      if (!preview || !preview.previewEnabled) return;
      zoneCamera = sourceId;
      zoneMode = mode;
      zoneAreas = mode === 'vehicle' ? [[...(preview.vehicleArea || [0,0,1,1])]] : (preview.excludedAreas || []).map((area) => [...area]);
      zoneSelected = zoneAreas.length - 1;
      zoneAction = null; zoneHistory = [];
      zoneStart = zoneCurrent = null;
      $("zoneTitle").textContent = `${mode === 'vehicle' ? 'Vehicle detection area' : 'Excluded areas'} — ${preview.label}`;
      $("zoneHelp").textContent = mode === 'vehicle' ? 'Drag inside the area to move it; drag a white handle to resize. Draw outside it to replace it, or clear all to use the whole view. Motion exclusions are separate.' : 'Draw on an empty part of the image to add an area. Drag inside an area to move it, or drag a white handle to resize. Motion inside shaded areas is ignored.';
      $("zoneImage").src = `/preview/${encodeURIComponent(sourceId)}`;
      $("zoneDialog").showModal();
      drawZones();
    }

    function closeZoneEditor() {
      $("zoneDialog").close();
      $("zoneImage").removeAttribute("src");
      zoneCamera = null;
      zoneAction = null;
    }

    $("zoneCanvas").addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || !$("zoneImage").naturalWidth) return;
      event.preventDefault();
      const point = zonePoint(event), hit = zoneHit(point);
      zoneRemember();
      zoneAction = {...hit, start:point, original:hit.index === undefined ? null : [...zoneAreas[hit.index]]};
      if (hit.type === 'draw') { zoneStart = zoneCurrent = point; zoneSelected = -1; }
      else { zoneSelected = hit.index; zoneStart = zoneCurrent = null; }
      event.currentTarget.setPointerCapture(event.pointerId);
      drawZones();
    });
    $("zoneCanvas").addEventListener("pointermove", (event) => {
      if (!zoneAction) return;
      if (zoneAction.type === 'draw') zoneCurrent = zonePoint(event);
      else applyZoneAction(zonePoint(event));
      drawZones();
    });
    $("zoneCanvas").addEventListener("pointerup", (event) => {
      if (!zoneAction) return;
      if (zoneAction.type === 'draw') {
        const area = zoneRectangle(zoneStart, zonePoint(event));
        if (area[2] >= .005 && area[3] >= .005) {
          if (zoneMode === 'vehicle') zoneAreas = [area];
          else if (zoneAreas.length < 20) zoneAreas.push(area);
          zoneSelected = zoneAreas.length - 1;
        } else zoneHistory.pop();
      } else {
        applyZoneAction(zonePoint(event));
      }
      zoneStart = zoneCurrent = null; zoneAction = null;
      drawZones();
    });
    $("zoneCanvas").addEventListener("pointercancel", () => { if (zoneAction && zoneHistory.length) zoneAreas = zoneHistory.pop(); zoneStart = zoneCurrent = zoneAction = null; zoneSelected = -1; drawZones(); });
    $("zoneUndo").addEventListener("click", () => { if (zoneHistory.length) { zoneAreas = zoneHistory.pop(); zoneSelected = zoneAreas.length - 1; drawZones(); } });
    $("zoneRemove").addEventListener("click", () => { if (zoneSelected >= 0) { zoneRemember(); zoneAreas.splice(zoneSelected,1); zoneSelected = zoneAreas.length - 1; drawZones(); } });
    $("zoneClear").addEventListener("click", () => { if (zoneAreas.length) { zoneRemember(); zoneAreas = []; zoneSelected = -1; drawZones(); } });
    $("zoneCancel").addEventListener("click", closeZoneEditor);
    $("zoneDialog").addEventListener("cancel", (event) => { event.preventDefault(); closeZoneEditor(); });
    $("zoneSave").addEventListener("click", async () => {
      if (!zoneCamera) return;
      $("zoneSave").disabled = true;
      try {
        const data = zoneMode === 'vehicle'
          ? await jsonFetch("/api/stream/vehicle-area", {source_id:zoneCamera, vehicle_area:zoneAreas[0] || [0,0,1,1]})
          : await jsonFetch("/api/stream/exclusions", {source_id:zoneCamera, excluded_areas:zoneAreas});
        const preview = state.previews.get(zoneCamera);
        if (preview) { if (zoneMode === 'vehicle') preview.vehicleArea = data.vehicle_area; else preview.excludedAreas = data.excluded_areas; }
        closeZoneEditor();
      } catch (error) {
        $("zoneStatus").textContent = error.message;
      } finally {
        $("zoneSave").disabled = false;
      }
    });

    async function stopSource(sourceId) {
      await jsonFetch("/api/stream/stop", { source_id: sourceId });
      removePreview(sourceId);
      setStatus("Stopping");
    }

    async function setPreviewEnabled(sourceId, enabled) {
      const preview = state.previews.get(sourceId);
      if (preview) {
        preview.previewEnabled = enabled;
        renderPreviews();
      }

      try {
        const data = await jsonFetch("/api/preview/set", { source_id: sourceId, enabled });
        const updatedPreview = state.previews.get(sourceId);
        if (updatedPreview) {
          updatedPreview.previewEnabled = data.preview_enabled;
          renderPreviews();
        }
      } catch (error) {
        $("scanStatus").textContent = error.message;
        setStatus("Error");
      }
    }

    async function pollWorkers() {
      try {
        const response = await fetch("/api/workers");
        const data = await response.json();
        const cameraSelect = $('clipTestCamera');
        const selectedCamera = cameraSelect.value;
        cameraSelect.innerHTML = '<option value="">Choose a running camera</option>' + data.workers.map(worker =>
          `<option value="${escapeAttribute(worker.source_id)}">${escapeHtml(worker.label)} (sensitivity ${worker.sensitivity})</option>`).join('');
        if (data.workers.some(worker => worker.source_id === selectedCamera)) cameraSelect.value = selectedCamera;
        const activeIds = new Set(data.workers.map((worker) => worker.source_id));
        for (const worker of data.workers) {
          addPreview(worker.source_id, worker.label, worker.preview_enabled, worker.excluded_areas || [], worker.recording, worker.status, worker.object_recording, worker.object_detection_status, worker.sensitivity, worker.vehicle_area, worker.object_processing, worker.detected_objects || []);
        }
        for (const sourceId of Array.from(state.previews.keys())) {
          if (!activeIds.has(sourceId)) state.previews.delete(sourceId);
        }
        if (state.previews.size) {
          setStatus(`${state.previews.size} running`);
        } else if ($("connectionStatus").textContent !== "Scanning") {
          setStatus("Ready");
        }
        renderPreviews();
      } catch (_error) {
        setStatus("Disconnected");
      }
    }

    function renderClipTest(job) {
      if (!job) return;
      const progress = job.duration ? `${Math.round(job.position)} / ${Math.round(job.duration)} seconds` : `${Math.round(job.position)} seconds`;
      $('clipTestStatus').textContent = `Test ${job.status} · ${progress} · ${job.motion_checks} motion checks · ${job.object_checks} object checks${job.error ? ` · ${job.error}` : ''}`;
      $('clipTestStart').disabled = ['uploading', 'running'].includes(job.status);
      const files = (job.files || []).map(file => `<a href="${escapeAttribute(file.url)}" target="_blank" rel="noreferrer">${escapeHtml(file.kind)}: ${escapeHtml(file.name)}</a>`).join('');
      const checks = job.events || [];
      const labels = [...new Set(checks.flatMap(event => event.objects || []))];
      const matches = checks.flatMap(event => event.scenarios || []);
      const summary = job.object_checks ? `Objects: ${labels.length ? labels.join(', ') : 'none identified'} · Scenario matches: ${matches.length}` : 'No object checks yet';
      const events = checks.map(event => `<div class="clip-meta">${event.captured_at ? escapeHtml(formatDateTime(event.captured_at)) : 'Time unavailable'}: ${escapeHtml((event.objects || []).join(', ') || 'no objects')}${(event.scenarios || []).length ? ` · Rule: ${escapeHtml(event.scenarios.map(x => x.reason).join('; '))}` : ''}${(event.decisions || []).length ? ` · ${escapeHtml([...new Set(event.decisions)].join('; '))}` : ''}</div>`).join('');
      $('clipTestResults').innerHTML = `<div class="clip-meta">${escapeHtml(summary)}</div>${files ? `<div class="clip-actions">${files}</div>` : ''}${events ? `<details class="evidence-details"><summary>Object check details (${checks.length})</summary>${events}</details>` : ''}`;
    }
    async function pollClipTest() {
      try {
        const response = await fetch('/api/clip-test');
        if (response.ok) renderClipTest((await response.json()).job);
      } catch (_error) { /* Keep the last status visible. */ }
    }
    $('clipTestStart').addEventListener('click', async () => {
      const file = $('clipTestFile').files[0], camera = $('clipTestCamera').value, recorded = $('clipTestTime').value;
      if (!file || !camera || !recorded) { $('clipTestStatus').textContent = 'Choose a camera, recording time and video.'; return; }
      if (file.size > 250 * 1024 * 1024) { $('clipTestStatus').textContent = 'The video must be 250 MB or smaller.'; return; }
      $('clipTestStart').disabled = true;
      $('clipTestStatus').textContent = 'Uploading clip…';
      try {
        const query = new URLSearchParams({camera, recorded_at:recorded, filename:file.name});
        const response = await fetch(`/api/clip-test?${query}`, {method:'POST', headers:{'Content-Type':'application/octet-stream'}, body:file});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Clip upload failed.');
        renderClipTest(data.job);
      } catch (error) { $('clipTestStatus').textContent = error.message; $('clipTestStart').disabled = false; }
    });

    async function pollHealth() {
      try {
        const response = await fetch('/api/health');
        if (!response.ok) throw new Error('unavailable');
        const health = await response.json();
        const warnings = [];
        if (health.selected_not_running) warnings.push(`${health.selected_not_running} selected camera(s) stopped`);
        for (const camera of health.cameras || []) {
          if (camera.frame_age_seconds > 30) warnings.push(`${camera.label}: no recent frames`);
          if (['error', 'disabled', 'stalled', 'degraded'].includes(camera.object_detection_status)) warnings.push(`${camera.label}: object detection ${camera.object_detection_status}`);
          if ((camera.status || '').toLowerCase().includes('paused')) warnings.push(`${camera.label}: ${camera.status}`);
        }
        if (health.disk_free_bytes < 2147483648) warnings.push('Low disk space');
        if (health.alerts && health.alerts.unknown) warnings.push(`${health.alerts.unknown} WhatsApp alert(s) have an unknown outcome`);
        if (!health.archive_enabled) warnings.push('Cloud archive is disabled');
        if (health.archive && health.archive.failed) warnings.push(`${health.archive.failed} archive upload(s) need retry`);
        $('healthStatus').textContent = warnings.length ? warnings.join(' · ') : 'System health: no warnings';
      } catch (_error) { $('healthStatus').textContent = 'System health unavailable'; }
    }

    async function loadLogs() {
      const pane = $("appLogs");
      const shouldStickToBottom = $("autoScrollLogs").checked ||
        pane.scrollTop + pane.clientHeight >= pane.scrollHeight - 24;
      try {
        const response = await fetch("/api/logs");
        const data = await response.json();
        pane.textContent = data.text || "No log entries yet.";
        if (shouldStickToBottom) pane.scrollTop = pane.scrollHeight;
      } catch (_error) {
        pane.textContent = "Could not load logs.";
      }
    }

    async function loadMedia(folderKey = state.activeMediaFolder, page = 0) {
      state.activeMediaFolder = folderKey;
      $('loadMoreMedia').disabled = true;
      try {
        const response = await fetch(`/api/media?folder=${encodeURIComponent(folderKey)}&page=${page}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || response.statusText);
        state.mediaFolders = data.folders || state.mediaFolders;
        state.mediaPage = page;
        const incoming = data.items || [];
        if (page === 0) state.mediaItems = incoming;
        else {
          const seen = new Set(state.mediaItems.map(item => item.archive_id || item.url));
          state.mediaItems.push(...incoming.filter(item => !seen.has(item.archive_id || item.url)));
        }
        state.mediaHasMore = Boolean(data.has_more);
        $('mediaMore').hidden = !state.mediaHasMore;
        renderMedia();
        if (data.archive_unavailable) $("clipList").insertAdjacentHTML('afterbegin', '<div class="status">Cloud archive is temporarily unavailable; showing local media.</div>');
      } catch (_error) {
        renderMediaTabs();
        if (page === 0) $("clipList").innerHTML = `<div class="status">Could not load saved media.</div>`;
        else $("clipList").insertAdjacentHTML('beforeend', '<div class="status">Could not load more media. Please try again.</div>');
      } finally {
        $('loadMoreMedia').disabled = false;
      }
    }

    $("clipList").addEventListener("click", async (event) => {
      const button = event.target.closest('[data-review-id]');
      if (!button) return;
      button.disabled = true;
      try {
        await jsonFetch('/api/archive/review', {media_id:button.dataset.reviewId, review:button.dataset.review, note:''});
        await loadMedia(state.activeMediaFolder);
      } catch (error) { alert(error.message); button.disabled = false; }
    });
    $('loadMoreMedia').addEventListener('click', () => { if (state.mediaHasMore) loadMedia(state.activeMediaFolder, state.mediaPage + 1); });
    $("scanButton").addEventListener("click", scanStreams);
    $("manualButton").addEventListener("click", addManualChannel);
    $("refreshLogsButton").addEventListener("click", loadLogs);
    $("refreshClipsButton").addEventListener("click", () => loadMedia());
    document.addEventListener("input", (event) => {
      if (event.target.matches("input")) {
        const streamId = event.target.dataset.sensitivityId;
        if (streamId) state.sensitivities[streamId] = Number(event.target.value || 5);
        saveState();
      }
    });
    document.addEventListener("change", (event) => {
      if (event.target.dataset.algorithmId) state.algorithms[event.target.dataset.algorithmId] = event.target.value;
      if (event.target.matches("input, select")) saveState();
    });
    document.addEventListener("click", (event) => {
      const startIndex = event.target.dataset.startIndex;
      const stopId = event.target.dataset.stopId;
      const previewId = event.target.dataset.previewId;
      const previewEnabled = event.target.dataset.previewEnabled;
      const mediaFolder = event.target.dataset.mediaFolder;
      const zoneId = event.target.dataset.zoneId;
      const vehicleZoneId = event.target.dataset.vehicleZoneId;
      if (startIndex !== undefined) startStream(Number(startIndex));
      if (stopId !== undefined) stopSource(decodeURIComponent(stopId));
      if (previewId !== undefined) setPreviewEnabled(decodeURIComponent(previewId), previewEnabled === "1");
      if (mediaFolder !== undefined) loadMedia(mediaFolder);
      if (zoneId !== undefined) openZoneEditor(decodeURIComponent(zoneId));
      if (vehicleZoneId !== undefined) openZoneEditor(decodeURIComponent(vehicleZoneId), 'vehicle');
    });

    loadDefaults();
    loadLogs();
    loadMedia();
    pollHealth();
    pollClipTest();
    setInterval(pollClipTest, 3000);
    setInterval(pollHealth, 10000);
    setInterval(pollWorkers, 3000);
    setInterval(loadLogs, 3000);
  </script>
</body>
</html>
"""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Serve the security camera browser console.")
    parser.add_argument("--host", default="0.0.0.0", help="Host/IP to bind. Default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind. Default: 8080")
    return parser.parse_args(argv)


def stop_all_workers():
    for worker in list(APP_STATE.workers.values()):
        worker.stop_event.set()
        with worker.condition:
            worker.condition.notify_all()


def main(argv=None):
    global ARCHIVE_MANAGER
    configure_logging()
    load_dotenv()
    args = parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), SecurityCameraWebHandler)

    def stop(_signum, _frame):
        logging.info("Stop requested.")
        stop_all_workers()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    import object_alerts
    if ARCHIVE_ENABLED:
        from archive_manager import ArchiveManager
        ARCHIVE_MANAGER = ArchiveManager(Path.home() / '.local/state/securitycamera/archive.db',
            {'motion': OUTPUT_DIR, 'object': OBJECT_OUTPUT_DIR,
             'image': OBJECT_OUTPUT_DIR / 'detection_frames'},
            alert_status=object_alerts.get_outbox().status,
            alerts_enabled=os.getenv('WHATSAPP_OBJECT_ALERTS_ENABLED', '').lower() in ('1', 'true', 'yes', 'on'))
        ARCHIVE_MANAGER.recover_interrupted()
        ARCHIVE_MANAGER.start()
    if os.getenv('WHATSAPP_OBJECT_ALERTS_ENABLED', '').lower() in ('1', 'true', 'yes', 'on'):
        object_alerts.start_alert_worker(on_accepted=ARCHIVE_MANAGER.outbox.release if ARCHIVE_MANAGER else None)
    SUPERVISOR_STOP.clear()
    supervisor = threading.Thread(target=supervise_cameras, daemon=True)
    supervisor.start()
    logging.info("Security camera web console listening on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    finally:
        SUPERVISOR_STOP.set()
        supervisor.join(timeout=2)
        workers = list(APP_STATE.workers.values())
        stop_all_workers()
        CLIP_REPLAY.stop()
        for worker in workers:
            worker.thread.join(timeout=30)
        server.server_close()
        if ARCHIVE_MANAGER:
            ARCHIVE_MANAGER.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
