from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from app_config import load_dotenv
from camera_discovery import DiscoveredStream, discover_nvr_streams
from network_guess import likely_dvr_ip
from security_camera_core import (
    AVAILABLE_TARGET_OBJECTS,
    DEFAULT_RTSP_PATH_TEMPLATE,
    DEFAULT_RTSP_PORT,
    DEFAULT_TARGET_OBJECTS,
    build_rtsp_url,
    can_open_rtsp_channel,
    run_detection,
    run_video_clip_detection,
)


LOG_FILE = "security_camera_web.log"
DEFAULT_MOTION_SENSITIVITY = 5
UPLOAD_DIR = Path.home() / "SecurityCamera" / "uploads"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@dataclass
class WorkerState:
    source_id: str
    label: str
    stop_event: threading.Event
    thread: threading.Thread
    kind: str
    status: str = "starting"
    analysis: dict = field(default_factory=dict)
    frame: bytes | None = None
    frame_version: int = 0
    condition: threading.Condition = field(default_factory=threading.Condition)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


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
                    "status": worker.status,
                    "analysis": worker.analysis,
                    "started_at": worker.started_at,
                }
                for worker in self.workers.values()
            ]


APP_STATE = WebAppState()


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


def selected_targets(raw_targets):
    if not raw_targets:
        return set(DEFAULT_TARGET_OBJECTS)

    targets = {str(target).strip() for target in raw_targets}
    return targets & set(AVAILABLE_TARGET_OBJECTS)


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


def update_analysis(worker, clip_name, message, progress):
    worker.analysis = {
        "clip_name": clip_name,
        "message": message,
        "progress": progress,
    }


def run_camera_worker(worker, stream, settings, target_objects, sensitivity):
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
            target_objects,
            worker.stop_event.is_set,
            lambda frame: update_frame(worker, frame),
            sensitivity,
            lambda message: update_status(worker, message),
            lambda clip_name, message, progress: update_analysis(worker, clip_name, message, progress),
        )
    except Exception as exc:
        logging.exception("Browser stream failed: %s", worker.label)
        worker.status = f"error: {exc}"
    finally:
        worker.stop_event.set()
        with worker.condition:
            worker.condition.notify_all()
        APP_STATE.remove_worker(worker.source_id)


def run_video_worker(worker, video_path, target_objects, sensitivity):
    try:
        worker.status = "running"
        run_video_clip_detection(
            str(video_path),
            target_objects,
            worker.stop_event.is_set,
            lambda frame: update_frame(worker, frame),
            sensitivity,
        )
        worker.status = "complete"
    except Exception as exc:
        logging.exception("Browser video clip failed: %s", video_path)
        worker.status = f"error: {exc}"
    finally:
        worker.stop_event.set()
        with worker.condition:
            worker.condition.notify_all()
        APP_STATE.remove_worker(worker.source_id)


def safe_upload_path(filename):
    suffix = Path(filename).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise ValueError("Upload must be a video file: mp4, mov, avi, mkv, or m4v")

    safe_stem = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in Path(filename).stem
    ).strip("_") or "video"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return UPLOAD_DIR / f"{timestamp}_{safe_stem}{suffix}"


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


class SecurityCameraWebHandler(BaseHTTPRequestHandler):
    server_version = "SecurityCameraWeb/1.0"

    def log_message(self, fmt, *args):
        logging.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self.send_html(INDEX_HTML)
        if parsed.path == "/api/defaults":
            return self.send_json(
                {
                    "settings": default_settings(),
                    "available_target_objects": list(AVAILABLE_TARGET_OBJECTS),
                    "default_target_objects": list(DEFAULT_TARGET_OBJECTS),
                    "default_motion_sensitivity": DEFAULT_MOTION_SENSITIVITY,
                }
            )
        if parsed.path == "/api/workers":
            return self.send_json({"workers": APP_STATE.worker_summaries()})
        if parsed.path == "/api/logs":
            return self.send_json({"text": read_log_tail()})
        if parsed.path.startswith("/preview/"):
            source_id = unquote(parsed.path[len("/preview/"):])
            return self.send_preview(source_id)
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/scan":
            return self.handle_scan()
        if parsed.path == "/api/stream/start":
            return self.handle_stream_start()
        if parsed.path == "/api/stream/stop":
            return self.handle_stream_stop()
        if parsed.path == "/api/video/start":
            return self.handle_video_start(parsed)
        self.send_error(HTTPStatus.NOT_FOUND)

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
        payload = self.read_json()
        settings = payload.get("settings", {})
        stream = stream_from_payload(payload.get("stream", {}))
        if not stream.stream_id:
            return self.send_json({"error": "stream_id is required"}, HTTPStatus.BAD_REQUEST)

        target_objects = selected_targets(payload.get("target_objects", []))
        if not target_objects:
            return self.send_json({"error": "Select at least one object to detect."}, HTTPStatus.BAD_REQUEST)

        sensitivity = parse_sensitivity(payload.get("sensitivity"))
        stop_event = threading.Event()
        worker = WorkerState(
            source_id=stream.stream_id,
            label=stream.label,
            stop_event=stop_event,
            thread=None,
            kind="camera",
        )
        thread = threading.Thread(
            target=run_camera_worker,
            args=(worker, stream, settings, target_objects, sensitivity),
            daemon=True,
        )
        worker.thread = thread

        try:
            APP_STATE.add_worker(worker)
        except ValueError as exc:
            return self.send_json({"error": str(exc)}, HTTPStatus.CONFLICT)

        thread.start()
        self.send_json({"worker": {"source_id": worker.source_id, "label": worker.label}})

    def handle_stream_stop(self):
        payload = self.read_json()
        source_id = str(payload.get("source_id", "")).strip()
        worker = APP_STATE.get_worker(source_id)
        if worker is None:
            return self.send_json({"stopped": False})

        worker.status = "stopping"
        worker.stop_event.set()
        with worker.condition:
            worker.condition.notify_all()
        self.send_json({"stopped": True})

    def handle_video_start(self, parsed):
        query = parse_qs(parsed.query)
        filename = query.get("filename", [""])[0]
        if not filename:
            return self.send_json({"error": "filename is required"}, HTTPStatus.BAD_REQUEST)

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return self.send_json({"error": "Upload body is empty"}, HTTPStatus.BAD_REQUEST)

        try:
            video_path = safe_upload_path(filename)
        except ValueError as exc:
            return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        remaining = content_length
        with video_path.open("wb") as output_file:
            while remaining:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                output_file.write(chunk)
                remaining -= len(chunk)

        if remaining:
            return self.send_json({"error": "Upload ended before the complete file was received"}, HTTPStatus.BAD_REQUEST)

        targets = selected_targets(query.get("targets", []))
        if not targets:
            targets = set(DEFAULT_TARGET_OBJECTS)
        sensitivity = parse_sensitivity(query.get("sensitivity", [DEFAULT_MOTION_SENSITIVITY])[0])
        source_id = f"clip:{video_path.name}"
        stop_event = threading.Event()
        worker = WorkerState(
            source_id=source_id,
            label=f"Video Clip: {video_path.name}",
            stop_event=stop_event,
            thread=None,
            kind="video",
        )
        thread = threading.Thread(
            target=run_video_worker,
            args=(worker, video_path, targets, sensitivity),
            daemon=True,
        )
        worker.thread = thread

        try:
            APP_STATE.add_worker(worker)
        except ValueError as exc:
            return self.send_json({"error": str(exc)}, HTTPStatus.CONFLICT)

        thread.start()
        self.send_json({"worker": {"source_id": worker.source_id, "label": worker.label}})

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
      gap: 14px;
    }
    .status {
      min-height: 38px;
      color: var(--muted);
      white-space: pre-wrap;
    }
    .stream-list {
      display: grid;
      gap: 10px;
    }
    .preview-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr));
      gap: 10px;
    }
    @media (min-width: 1320px) {
      .preview-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
    .stream-item, .preview-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfd;
    }
    .stream-top {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 76px 74px;
      gap: 8px;
      align-items: center;
    }
    .stream-title {
      font-weight: 650;
      overflow-wrap: anywhere;
    }
    .stream-meta {
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .preview-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
    }
    .preview-title {
      min-width: 0;
      font-weight: 650;
      overflow-wrap: anywhere;
    }
    .preview-item img {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #101820;
      border-radius: 6px;
    }
    .video-controls {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 82px;
      gap: 10px;
      align-items: end;
    }
    .logs-section {
      margin: 0 14px 14px;
    }
    .logs-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }
    .logs-head h2 {
      margin: 0;
    }
    .logs-actions {
      display: flex;
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
    <h1>Security Camera Browser Console</h1>
    <div id="connectionStatus">Idle</div>
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
        <label>Record Objects</label>
        <div id="targetObjects" class="checks"></div>
        <div class="actions">
          <button id="scanButton">Connect</button>
          <button id="manualButton" class="secondary">Add Manual Channel</button>
        </div>
        <label for="manualChannel">Manual Channel</label>
        <input id="manualChannel" placeholder="101">
      </section>

      <section>
        <h2>Video Clip</h2>
        <div class="video-controls">
          <div>
            <label for="videoFile">Upload Clip</label>
            <input id="videoFile" type="file" accept=".mp4,.mov,.avi,.mkv,.m4v,video/*">
          </div>
          <div>
            <label for="videoSensitivity">Sensitivity</label>
            <input id="videoSensitivity" type="number" min="1" max="10" value="5">
          </div>
        </div>
        <div class="actions">
          <button id="uploadButton">Upload & Annotate</button>
        </div>
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
    </div>
  </main>

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
      previews: new Map(),
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
        target_objects: targetObjects(),
        streams: state.streams,
        sensitivities: state.sensitivities,
        video_sensitivity: $("videoSensitivity").value || "5"
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

    function applySavedTargets(savedTargets) {
      if (!Array.isArray(savedTargets) || !savedTargets.length) return;
      const selected = new Set(savedTargets);
      for (const checkbox of document.querySelectorAll("[data-target-object]")) {
        checkbox.checked = selected.has(checkbox.value);
      }
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

    function targetObjects() {
      return Array.from(document.querySelectorAll("[data-target-object]:checked")).map((box) => box.value);
    }

    function renderTargetObjects(objects, defaults) {
      $("targetObjects").innerHTML = objects.map((objectName) => `
        <label>
          <input type="checkbox" value="${objectName}" data-target-object ${defaults.includes(objectName) ? "checked" : ""}>
          ${titleCase(objectName)}
        </label>
      `).join("");
    }

    function titleCase(value) {
      return value.charAt(0).toUpperCase() + value.slice(1);
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
              <div class="stream-title">${escapeHtml(stream.label)}</div>
              <div class="stream-meta">${escapeHtml(stream.stream_id)}</div>
            </div>
            <input id="sensitivity-${index}" data-sensitivity-id="${escapeAttribute(stream.stream_id)}" type="number" min="1" max="10" value="${state.sensitivities[stream.stream_id] || 5}" title="Sensitivity">
            <button data-start-index="${index}">Start</button>
          </div>
        </div>
      `).join("");
    }

    function addPreview(sourceId, label) {
      if (state.previews.has(sourceId)) return;
      state.previews.set(sourceId, { sourceId, label });
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
            <button class="danger" data-stop-id="${encodeURIComponent(preview.sourceId)}">Stop</button>
          </div>
          <img src="/preview/${encodeURIComponent(preview.sourceId)}" alt="">
        </div>
      `).join("");
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
      $("videoSensitivity").value = saved.video_sensitivity || defaults.default_motion_sensitivity || 5;
      renderTargetObjects(defaults.available_target_objects, defaults.default_target_objects);
      applySavedTargets(saved.target_objects);
      if (Array.isArray(saved.streams)) state.streams = saved.streams;
      state.sensitivities = saved.sensitivities || {};
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
        const data = await jsonFetch("/api/stream/start", {
          settings: settings(),
          stream,
          target_objects: targetObjects(),
          sensitivity
        });
        addPreview(data.worker.source_id, data.worker.label);
        setStatus("Running");
      } catch (error) {
        $("scanStatus").textContent = error.message;
        setStatus("Error");
      }
    }

    async function stopSource(sourceId) {
      await jsonFetch("/api/stream/stop", { source_id: sourceId });
      removePreview(sourceId);
      setStatus("Stopping");
    }

    async function uploadVideo() {
      const file = $("videoFile").files[0];
      if (!file) {
        $("scanStatus").textContent = "Choose a video clip first.";
        return;
      }
      $("uploadButton").disabled = true;
      setStatus("Uploading");
      try {
        const params = new URLSearchParams();
        params.set("filename", file.name);
        params.set("sensitivity", $("videoSensitivity").value || "5");
        for (const target of targetObjects()) params.append("targets", target);
        const response = await fetch(`/api/video/start?${params.toString()}`, {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream" },
          body: file
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || response.statusText);
        addPreview(data.worker.source_id, data.worker.label);
        $("scanStatus").textContent = `Uploaded ${file.name}. Annotation started.`;
        setStatus("Running");
      } catch (error) {
        $("scanStatus").textContent = error.message;
        setStatus("Error");
      } finally {
        $("uploadButton").disabled = false;
      }
    }

    async function pollWorkers() {
      try {
        const response = await fetch("/api/workers");
        const data = await response.json();
        const activeIds = new Set(data.workers.map((worker) => worker.source_id));
        for (const worker of data.workers) {
          addPreview(worker.source_id, worker.label);
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

    $("scanButton").addEventListener("click", scanStreams);
    $("manualButton").addEventListener("click", addManualChannel);
    $("uploadButton").addEventListener("click", uploadVideo);
    $("refreshLogsButton").addEventListener("click", loadLogs);
    document.addEventListener("input", (event) => {
      if (event.target.matches("input")) {
        const streamId = event.target.dataset.sensitivityId;
        if (streamId) state.sensitivities[streamId] = Number(event.target.value || 5);
        saveState();
      }
    });
    document.addEventListener("change", (event) => {
      if (event.target.matches("input")) saveState();
    });
    document.addEventListener("click", (event) => {
      const startIndex = event.target.dataset.startIndex;
      const stopId = event.target.dataset.stopId;
      if (startIndex !== undefined) startStream(Number(startIndex));
      if (stopId !== undefined) stopSource(decodeURIComponent(stopId));
    });

    loadDefaults();
    loadLogs();
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

    logging.info("Security camera web console listening on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    finally:
        stop_all_workers()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
