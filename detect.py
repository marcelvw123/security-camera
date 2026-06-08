import sys
import logging
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
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

MIN_MOTION_AREA = 1000
CLIP_SECONDS = 10
OUTPUT_DIR = Path("clips")
TARGET_OBJECTS = {"car"}
DEFAULT_RTSP_PORT = "554"
CHANNEL_SCAN_LIMIT = 16
STREAM_OPEN_TIMEOUT_MS = 1200
STREAM_READ_TIMEOUT_MS = 1200
PREVIEW_EVERY_N_FRAMES = 3
JPEG_PREVIEW_QUALITY = 70
PREVIEW_COLUMNS = 4
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

    def __init__(self, channel, rtsp_url, window_title):
        super().__init__()
        self.channel = channel
        self.rtsp_url = rtsp_url
        self.window_title = window_title
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            run_detection(
                self.rtsp_url,
                self.window_title,
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


class CameraSettingsDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Security Camera Setup")
        self.setMinimumSize(1200, 720)

        self.scan_thread = None
        self.scan_worker = None
        self.stream_workers = {}
        self.preview_labels = {}

        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.ip_input = QLineEdit()
        self.port_input = QLineEdit(DEFAULT_RTSP_PORT)
        self.manual_channel_input = QLineEdit()
        self.camera_list = QListWidget()
        self.camera_list.setMinimumHeight(180)
        self.camera_list.itemChanged.connect(self.toggle_stream)
        self.scan_status = QLabel("Scan for cameras, then tick a stream to start it.")
        self.scan_button = QPushButton("Scan Cameras")
        self.scan_button.clicked.connect(self.scan_cameras)
        self.add_manual_button = QPushButton("Add Manual Channel")
        self.add_manual_button.clicked.connect(self.add_manual_channel)
        self.preview_layout = QGridLayout()
        self.preview_layout.setSpacing(10)

        form = QFormLayout()
        form.addRow("Username", self.username_input)
        form.addRow("Password", self.password_input)
        form.addRow("DVR/NVR IP", self.ip_input)
        form.addRow("RTSP Port", self.port_input)
        form.addRow("Manual Channel", self.manual_channel_input)
        form.addRow("", self.scan_button)
        form.addRow("", self.add_manual_button)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)

        left_panel = QVBoxLayout()
        left_panel.addLayout(form)
        left_panel.addWidget(self.scan_status)
        left_panel.addWidget(self.camera_list)
        left_panel.addWidget(buttons)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setFixedWidth(380)

        preview_container = QWidget()
        preview_container.setLayout(self.preview_layout)

        preview_scroll = QScrollArea()
        preview_scroll.setWidgetResizable(True)
        preview_scroll.setWidget(preview_container)

        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("Camera Previews"))
        right_panel.addWidget(preview_scroll)

        right_widget = QWidget()
        right_widget.setLayout(right_panel)

        layout = QHBoxLayout()
        layout.addWidget(left_widget)
        layout.addWidget(right_widget, 1)
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
                "Enter username, password, DVR/NVR IP, and RTSP port before scanning.",
            )
            return

        self.stop_all_streams()
        self.camera_list.clear()
        self.scan_status.setText("Scanning common Hikvision channels...")
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
                "Enter username, password, DVR/NVR IP, RTSP port, and manual channel.",
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
        if not all(settings.values()):
            QMessageBox.warning(
                self,
                "Missing Details",
                "Enter username, password, DVR/NVR IP, and RTSP port before starting a stream.",
            )
            self.camera_list.blockSignals(True)
            item.setCheckState(Qt.Unchecked)
            self.camera_list.blockSignals(False)
            return

        rtsp_url = build_rtsp_url(settings, channel=channel)
        thread = QThread(self)
        worker = CameraDetectionWorker(channel, rtsp_url, label)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.frame_ready.connect(self.update_preview)
        worker.error.connect(self.show_stream_error)
        worker.finished.connect(self.finish_stream)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self.stream_workers[channel] = {
            "thread": thread,
            "worker": worker,
        }
        self.add_preview(channel, label)
        self.scan_status.setText(f"Started stream {channel}.")
        thread.start()

    def stop_stream(self, channel):
        stream = self.stream_workers.get(channel)
        if not stream:
            return

        stream["worker"].stop()
        self.scan_status.setText(f"Stopping stream {channel}...")

    def stop_all_streams(self):
        for stream in list(self.stream_workers.values()):
            stream["worker"].stop()

        for stream in list(self.stream_workers.values()):
            if not stream["thread"].wait(5000):
                logging.warning("Timed out waiting for stream worker to stop")

        self.stream_workers.clear()
        for channel in list(self.preview_labels):
            self.remove_preview(channel)

    def finish_stream(self, channel):
        self.stream_workers.pop(channel, None)
        self.uncheck_channel(channel)
        self.remove_preview(channel)
        self.scan_status.setText(f"Stopped stream {channel}.")

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

    def add_preview(self, channel, label):
        preview = QLabel(f"Starting {label}...")
        preview.setAlignment(Qt.AlignCenter)
        preview.setMinimumSize(260, 150)
        preview.setStyleSheet("border: 1px solid #bac4ce; background: #111827; color: white;")
        self.preview_labels[channel] = preview
        self.relayout_previews()

    def update_preview(self, channel, frame_bytes):
        preview = self.preview_labels.get(channel)
        if preview is None:
            return

        pixmap = QPixmap()
        if not pixmap.loadFromData(frame_bytes, "JPG"):
            return

        preview.setPixmap(
            pixmap.scaled(
                preview.width(),
                max(preview.height(), 1),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def remove_preview(self, channel):
        preview = self.preview_labels.pop(channel, None)
        if preview is None:
            return

        self.preview_layout.removeWidget(preview)
        preview.deleteLater()
        self.relayout_previews()

    def relayout_previews(self):
        for index, preview in enumerate(self.preview_labels.values()):
            row = index // PREVIEW_COLUMNS
            column = index % PREVIEW_COLUMNS
            self.preview_layout.addWidget(preview, row, column)


def get_camera_settings():
    dialog = CameraSettingsDialog()
    dialog.exec()


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

    return f"rtsp://{username}:{password}@{ip_address}:{port}/Streaming/Channels/{channel}"


def create_video_writer(clip_path, fps, size):
    for codec in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, size)

        if writer.isOpened():
            print(f"Using video codec: {codec}")
            return writer

        writer.release()

    raise RuntimeError("Could not create video writer")


def run_detection(rtsp_url, window_title, stop_requested, frame_callback):
    model = YOLO(resource_path("yolov8n.pt"))
    motion_detector = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=True,
    )

    cap = cv2.VideoCapture()

    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, STREAM_OPEN_TIMEOUT_MS)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, STREAM_READ_TIMEOUT_MS)

    if not cap.open(rtsp_url, cv2.CAP_FFMPEG):
        raise RuntimeError("Could not open RTSP stream.")

    OUTPUT_DIR.mkdir(exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 20

    writer = None
    frames_left_in_clip = 0
    frame_count = 0

    try:
        while not stop_requested():
            ret, frame = cap.read()

            if not ret:
                print("Could not read frame")
                break

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
            annotated_frame = frame.copy()

            for contour in contours:
                if cv2.contourArea(contour) < MIN_MOTION_AREA:
                    continue

                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            if motion_detected:
                results = model(frame, verbose=False)

                annotated_frame = results[0].plot()
                detected_objects = {
                    model.names[int(box.cls[0])]
                    for box in results[0].boxes
                }
                target_object_detected = bool(detected_objects & TARGET_OBJECTS)

            if writer is None and target_object_detected:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_window_title = "".join(
                    character if character.isalnum() else "_"
                    for character in window_title
                ).strip("_")
                clip_path = OUTPUT_DIR / f"car_{safe_window_title}_{timestamp}.mp4"
                height, width = annotated_frame.shape[:2]
                writer = create_video_writer(clip_path, fps, (width, height))
                frames_left_in_clip = int(fps * CLIP_SECONDS)
                print(f"Recording car clip: {clip_path}")

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

        cap.release()


def main():
    logging.info("Starting security camera app")
    app = QApplication(sys.argv)
    get_camera_settings()
    app.quit()
    logging.info("Security camera app stopped")


if __name__ == "__main__":
    main()
