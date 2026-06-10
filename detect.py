import sys
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import cv2
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QCheckBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

MIN_MOTION_AREA = 1000
CLIP_SECONDS = 10
YOLO_MODEL_FILENAME = "yolo11s.pt"
YOLO_CONFIDENCE = 0.55
OUTPUT_DIR = Path.home() / "SecurityCamera" / "clips"
DEFAULT_TARGET_OBJECTS = {"car"}
AVAILABLE_TARGET_OBJECTS = ("car", "person")
DEFAULT_RTSP_PORT = "554"
DEFAULT_RTSP_PATH_TEMPLATE = "/Streaming/Channels/{channel}"
CHANNEL_SCAN_LIMIT = 16
STREAM_OPEN_TIMEOUT_MS = 1200
STREAM_READ_TIMEOUT_MS = 1200
STREAM_RECONNECT_DELAY_SECONDS = 2
READ_FAILURES_BEFORE_RECONNECT = 3
PREVIEW_EVERY_N_FRAMES = 3
JPEG_PREVIEW_QUALITY = 70
LOG_FILE = Path("security_camera_app.log")


logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def log_uncaught_exception(exc_type, exc_value, exc_traceback):
    logging.error(
        "Uncaught exception:\n%s",
        "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
    )


sys.excepthook = log_uncaught_exception


def resource_path(filename):
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_path / filename


class CameraScanWorker(QObject):
    camera_found = Signal(int, str, str)
    finished = Signal(int)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        found_count = 0

        for camera_number, stream_name, channel in iter_common_hikvision_channels():
            if self._stopped:
                break

            try:
                if can_open_rtsp_channel(self.settings, channel):
                    found_count += 1
                    self.camera_found.emit(camera_number, stream_name, channel)
            except Exception as exc:
                logging.warning("Scan failed for channel %s: %s", channel, type(exc).__name__)

        self.finished.emit(found_count)


class CameraDetectionWorker(QObject):
    error = Signal(str, str)
    finished = Signal(str)
    frame_ready = Signal(str, bytes)

    def __init__(self, channel, rtsp_url, window_title, target_objects):
        super().__init__()
        self.channel = channel
        self.rtsp_url = rtsp_url
        self.window_title = window_title
        self.target_objects = target_objects
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            run_detection(
                self.rtsp_url,
                self.window_title,
                self.target_objects,
                self.should_stop,
                self.emit_frame,
            )
        except Exception as exc:
            logging.exception("Stream %s failed", self.channel)
            self.error.emit(self.channel, str(exc))
        finally:
            self.finished.emit(self.channel)

    def should_stop(self):
        return self._stopped

    def emit_frame(self, frame_bytes):
        self.frame_ready.emit(self.channel, frame_bytes)


class StreamPreviewWindow(QWidget):
    close_requested = Signal(str)

    def __init__(self, channel, label):
        super().__init__()
        self.channel = channel
        self.setWindowTitle(label)
        self.resize(720, 420)

        self.preview = QLabel(f"Starting {label}...")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(160, 90)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview.setStyleSheet("background: #111827; color: white;")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.preview)
        self.setLayout(layout)

    def update_frame(self, frame_bytes):
        pixmap = QPixmap()
        if not pixmap.loadFromData(frame_bytes, "JPG"):
            return

        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.width(),
                max(self.preview.height(), 1),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def closeEvent(self, event):
        self.close_requested.emit(self.channel)
        super().closeEvent(event)


class CameraSettingsDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Security Camera Setup")
        self.setMinimumSize(520, 560)

        self.scan_thread = None
        self.scan_worker = None
        self.stream_workers = {}
        self.preview_windows = {}

        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.ip_input = QLineEdit()
        self.port_input = QLineEdit(DEFAULT_RTSP_PORT)
        self.rtsp_path_template_input = QLineEdit(DEFAULT_RTSP_PATH_TEMPLATE)
        self.rtsp_path_template_input.setPlaceholderText("/Streaming/Channels/{channel}")
        self.manual_channel_input = QLineEdit()
        self.object_checkboxes = {}

        for object_name in AVAILABLE_TARGET_OBJECTS:
            checkbox = QCheckBox(object_name.title())
            checkbox.setChecked(object_name in DEFAULT_TARGET_OBJECTS)
            self.object_checkboxes[object_name] = checkbox

        self.camera_list = QListWidget()
        self.camera_list.setMinimumHeight(180)
        self.camera_list.itemChanged.connect(self.toggle_stream)
        self.scan_status = QLabel("Scan for cameras, then tick a stream to start it.")
        self.scan_button = QPushButton("Scan Cameras")
        self.scan_button.clicked.connect(self.scan_cameras)
        self.add_manual_button = QPushButton("Add Manual Channel")
        self.add_manual_button.clicked.connect(self.add_manual_channel)

        form = QFormLayout()
        form.addRow("Username", self.username_input)
        form.addRow("Password", self.password_input)
        form.addRow("DVR/NVR IP", self.ip_input)
        form.addRow("RTSP Port", self.port_input)
        form.addRow("RTSP Path Template", self.rtsp_path_template_input)
        form.addRow("Manual Channel", self.manual_channel_input)
        for object_name, checkbox in self.object_checkboxes.items():
            form.addRow("Record Objects", checkbox)
        form.addRow("", self.scan_button)
        form.addRow("", self.add_manual_button)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.scan_status)
        layout.addWidget(self.camera_list)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def reject(self):
        if self.is_scanning():
            QMessageBox.warning(self, "Scan Running", "Wait for the camera scan to finish.")
            return

        self.stop_all_streams()
        super().reject()

    def closeEvent(self, event):
        if self.is_scanning():
            QMessageBox.warning(self, "Scan Running", "Wait for the camera scan to finish.")
            event.ignore()
            return

        self.stop_all_streams()
        super().closeEvent(event)

    def settings(self):
        return {
            "username": self.username_input.text().strip(),
            "password": self.password_input.text(),
            "ip_address": self.ip_input.text().strip(),
            "port": self.port_input.text().strip(),
            "rtsp_path_template": self.rtsp_path_template_input.text().strip(),
        }

    def selected_target_objects(self):
        return {
            object_name
            for object_name, checkbox in self.object_checkboxes.items()
            if checkbox.isChecked()
        }

    def scan_cameras(self):
        if self.is_scanning():
            return

        settings = self.settings()
        required_connection_fields = (
            settings["username"],
            settings["password"],
            settings["ip_address"],
            settings["port"],
        )

        if not all(required_connection_fields):
            QMessageBox.warning(
                self,
                "Missing Details",
                "Enter username, password, DVR/NVR IP, RTSP port, and RTSP path template before scanning.",
            )
            return

        self.stop_all_streams()
        self.camera_list.clear()
        self.scan_status.setText("Scanning common camera/stream channels...")
        self.scan_button.setEnabled(False)
        self.scan_button.setText("Scanning...")

        self.scan_thread = QThread(self)
        self.scan_worker = CameraScanWorker(settings)
        self.scan_worker.moveToThread(self.scan_thread)

        self.scan_thread.started.connect(self.scan_worker.run)
        self.scan_worker.camera_found.connect(self.add_found_camera)
        self.scan_worker.finished.connect(self.finish_scan)
        self.scan_worker.finished.connect(self.scan_thread.quit)
        self.scan_worker.finished.connect(self.scan_worker.deleteLater)
        self.scan_thread.finished.connect(self.scan_thread.deleteLater)
        self.scan_thread.finished.connect(self.clear_scan_thread)
        self.scan_thread.start()

    def add_found_camera(self, camera_number, stream_name, channel):
        label = f"Camera {camera_number} {stream_name} ({channel})"
        self.add_camera_item(label, channel)
        self.scan_status.setText(f"Found {self.camera_list.count()} camera stream(s)...")

    def add_camera_item(self, label, channel):
        item = QListWidgetItem(label)
        item.setData(Qt.UserRole, channel)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Unchecked)
        self.camera_list.addItem(item)

    def finish_scan(self, found_count):
        self.scan_button.setEnabled(True)
        self.scan_button.setText("Scan Cameras")

        if found_count:
            self.scan_status.setText(f"Scan complete. Tick a stream to start it. Found {found_count}.")
        else:
            self.scan_status.setText("Scan complete. No streams found. You can add a channel manually.")

    def clear_scan_thread(self):
        self.scan_thread = None
        self.scan_worker = None

    def is_scanning(self):
        return self.scan_thread is not None and self.scan_thread.isRunning()

    def add_manual_channel(self):
        settings = self.settings()
        channel = self.manual_channel_input.text().strip()

        if not all(settings.values()) or not channel:
            QMessageBox.warning(
                self,
                "Missing Details",
                "Enter username, password, DVR/NVR IP, RTSP port, RTSP path template, and manual channel.",
            )
            return

        label = f"Manual Channel ({channel})"
        self.add_camera_item(label, channel)
        self.manual_channel_input.clear()
        self.scan_status.setText("Manual channel added. Tick it to start the stream.")

    def toggle_stream(self, item):
        channel = str(item.data(Qt.UserRole)).strip()

        if item.checkState() == Qt.Checked:
            self.start_stream(channel, item.text(), item)
        else:
            self.stop_stream(channel)

    def start_stream(self, channel, label, item):
        if channel in self.stream_workers:
            return

        settings = self.settings()
        target_objects = self.selected_target_objects()
        if not all(settings.values()):
            QMessageBox.warning(
                self,
                "Missing Details",
                "Enter username, password, DVR/NVR IP, RTSP port, and RTSP path template before starting a stream.",
            )
            self.camera_list.blockSignals(True)
            item.setCheckState(Qt.Unchecked)
            self.camera_list.blockSignals(False)
            return

        if not target_objects:
            QMessageBox.warning(
                self,
                "Missing Objects",
                "Select at least one object to record.",
            )
            self.camera_list.blockSignals(True)
            item.setCheckState(Qt.Unchecked)
            self.camera_list.blockSignals(False)
            return

        try:
            rtsp_url = build_rtsp_url(settings, channel=channel)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid RTSP Template", str(exc))
            self.camera_list.blockSignals(True)
            item.setCheckState(Qt.Unchecked)
            self.camera_list.blockSignals(False)
            return

        thread = QThread(self)
        worker = CameraDetectionWorker(channel, rtsp_url, label, target_objects)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.frame_ready.connect(self.update_preview)
        worker.error.connect(self.show_stream_error)
        worker.finished.connect(self.finish_stream)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(lambda channel=channel: self.cleanup_stream_thread(channel))
        thread.finished.connect(thread.deleteLater)

        self.stream_workers[channel] = {
            "thread": thread,
            "worker": worker,
            "stopping": False,
        }
        self.add_preview_window(channel, label)
        self.scan_status.setText(f"Started stream {channel}.")
        thread.start()

    def stop_stream(self, channel):
        stream = self.stream_workers.get(channel)
        if not stream:
            return

        stream["stopping"] = True
        stream["worker"].stop()
        self.scan_status.setText(f"Stopping stream {channel}...")

    def stop_all_streams(self):
        for stream in list(self.stream_workers.values()):
            stream["stopping"] = True
            stream["worker"].stop()

        for stream in list(self.stream_workers.values()):
            if not stream["thread"].wait(5000):
                logging.warning("Timed out waiting for stream worker to stop")

        for channel in list(self.preview_windows):
            self.remove_preview_window(channel)

    def finish_stream(self, channel):
        self.uncheck_channel(channel)
        self.remove_preview_window(channel)
        self.scan_status.setText(f"Stopped stream {channel}.")

    def cleanup_stream_thread(self, channel):
        self.stream_workers.pop(channel, None)

    def uncheck_channel(self, channel):
        for index in range(self.camera_list.count()):
            item = self.camera_list.item(index)
            if str(item.data(Qt.UserRole)).strip() != channel:
                continue

            self.camera_list.blockSignals(True)
            item.setCheckState(Qt.Unchecked)
            self.camera_list.blockSignals(False)
            break

    def show_stream_error(self, channel, error):
        QMessageBox.critical(self, "Stream Error", f"Stream {channel} stopped:\n{error}")

    def add_preview_window(self, channel, label):
        preview_window = StreamPreviewWindow(channel, label)
        preview_window.close_requested.connect(self.stop_stream)
        self.preview_windows[channel] = preview_window
        preview_window.show()

    def update_preview(self, channel, frame_bytes):
        preview_window = self.preview_windows.get(channel)
        if preview_window is not None:
            preview_window.update_frame(frame_bytes)

    def remove_preview_window(self, channel):
        preview_window = self.preview_windows.pop(channel, None)
        if preview_window is None:
            return

        preview_window.close_requested.disconnect(self.stop_stream)
        preview_window.close()
        preview_window.deleteLater()


def iter_common_hikvision_channels():
    for camera_number in range(1, CHANNEL_SCAN_LIMIT + 1):
        yield camera_number, "Main Stream", f"{camera_number}01"
        yield camera_number, "Sub Stream", f"{camera_number}02"


def can_open_rtsp_channel(settings, channel):
    cap = cv2.VideoCapture()

    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, STREAM_OPEN_TIMEOUT_MS)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, STREAM_READ_TIMEOUT_MS)

    try:
        if not cap.open(build_rtsp_url(settings, channel=channel), cv2.CAP_FFMPEG):
            return False

        ret, _frame = cap.read()
        return ret
    finally:
        cap.release()


def build_rtsp_url(settings, channel=None):
    username = quote(settings["username"], safe="")
    password = quote(settings["password"], safe="")
    ip_address = settings["ip_address"]
    port = settings["port"]
    camera, stream = channel_parts(channel)
    template = settings["rtsp_path_template"]
    template_values = {
        "username": username,
        "password": password,
        "ip": ip_address,
        "ip_address": ip_address,
        "port": port,
        "channel": channel or "",
        "camera": camera,
        "stream": stream,
    }

    try:
        rendered_template = template.format(**template_values)
    except KeyError as exc:
        raise ValueError(f"Unknown RTSP template field: {exc}") from exc

    if rendered_template.startswith("rtsp://"):
        return rendered_template

    if not rendered_template.startswith("/"):
        rendered_template = f"/{rendered_template}"

    return f"rtsp://{username}:{password}@{ip_address}:{port}{rendered_template}"


def channel_parts(channel):
    channel_text = str(channel or "").strip()
    if channel_text.isdigit() and len(channel_text) >= 3:
        return channel_text[:-2], channel_text[-1]

    return channel_text, channel_text


def open_rtsp_capture(rtsp_url):
    cap = cv2.VideoCapture()

    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, STREAM_OPEN_TIMEOUT_MS)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, STREAM_READ_TIMEOUT_MS)

    if cap.open(rtsp_url, cv2.CAP_FFMPEG):
        return cap

    cap.release()
    return None


def wait_before_reconnect(stop_requested):
    end_time = time.monotonic() + STREAM_RECONNECT_DELAY_SECONDS
    while not stop_requested() and time.monotonic() < end_time:
        time.sleep(0.1)


def create_video_writer(clip_path, fps, size):
    for codec in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, size)

        if writer.isOpened():
            print(f"Using video codec: {codec}")
            return writer

        writer.release()

    raise RuntimeError("Could not create video writer")


def run_detection(rtsp_url, window_title, target_objects, stop_requested, frame_callback):
    model = YOLO(resource_path(YOLO_MODEL_FILENAME))
    motion_detector = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = None
    fps = 20
    writer = None
    frames_left_in_clip = 0
    frame_count = 0
    consecutive_read_failures = 0

    try:
        while not stop_requested():
            if cap is None:
                cap = open_rtsp_capture(rtsp_url)
                if cap is None:
                    logging.warning("Could not open RTSP stream %s. Retrying...", window_title)
                    wait_before_reconnect(stop_requested)
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 0:
                    fps = 20
                consecutive_read_failures = 0
                logging.info("Connected RTSP stream %s", window_title)

            ret, frame = cap.read()

            if not ret:
                consecutive_read_failures += 1
                if consecutive_read_failures < READ_FAILURES_BEFORE_RECONNECT:
                    continue

                logging.warning("Lost RTSP stream %s. Reconnecting...", window_title)
                if writer is not None:
                    writer.release()
                    writer = None

                cap.release()
                cap = None
                wait_before_reconnect(stop_requested)
                continue

            consecutive_read_failures = 0

            motion_mask = motion_detector.apply(frame)
            motion_mask = cv2.threshold(motion_mask, 244, 255, cv2.THRESH_BINARY)[1]
            motion_mask = cv2.dilate(motion_mask, None, iterations=2)

            contours, _ = cv2.findContours(
                motion_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            motion_detected = False
            target_object_detected = False
            matched_objects = set()
            annotated_frame = frame.copy()

            for contour in contours:
                if cv2.contourArea(contour) < MIN_MOTION_AREA:
                    continue

                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            if motion_detected:
                results = model(frame, verbose=False, conf=YOLO_CONFIDENCE)

                annotated_frame = results[0].plot()
                detected_objects = {
                    model.names[int(box.cls[0])]
                    for box in results[0].boxes
                }
                matched_objects = detected_objects & target_objects
                target_object_detected = bool(matched_objects)

            if writer is None and target_object_detected:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_window_title = "".join(
                    character if character.isalnum() else "_"
                    for character in window_title
                ).strip("_")
                object_label = "_".join(sorted(matched_objects))
                clip_path = OUTPUT_DIR / f"{object_label}_{safe_window_title}_{timestamp}.mp4"
                height, width = annotated_frame.shape[:2]
                writer = create_video_writer(clip_path, fps, (width, height))
                frames_left_in_clip = int(fps * CLIP_SECONDS)
                print(f"Recording {object_label} clip: {clip_path}")

            if writer is not None:
                writer.write(annotated_frame)
                frames_left_in_clip -= 1

                if frames_left_in_clip <= 0:
                    writer.release()
                    writer = None

            frame_count += 1
            if frame_count % PREVIEW_EVERY_N_FRAMES == 0:
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_PREVIEW_QUALITY]
                encoded, buffer = cv2.imencode(".jpg", annotated_frame, encode_params)
                if encoded:
                    frame_callback(buffer.tobytes())
    finally:
        if writer is not None:
            writer.release()

        if cap is not None:
            cap.release()


def main():
    logging.info("Starting security camera app")
    app = QApplication(sys.argv)
    window = CameraSettingsDialog()
    window.show()
    exit_code = app.exec()
    logging.info("Security camera app stopped")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
