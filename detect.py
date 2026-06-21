import sys
import logging
import traceback
from pathlib import Path

from camera_discovery import DiscoveredStream, discover_nvr_streams
from network_guess import likely_dvr_ip
from PySide6.QtCore import QObject, Qt, QThread, Signal, QSize, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QCheckBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QAbstractScrollArea,
)
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

LOG_FILE = Path("security_camera_app.log")
DEFAULT_MOTION_SENSITIVITY = 5


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)


def log_uncaught_exception(exc_type, exc_value, exc_traceback):
    logging.error(
        "Uncaught exception:\n%s",
        "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
    )


sys.excepthook = log_uncaught_exception


class CameraScanWorker(QObject):
    camera_found = Signal(object)
    progress = Signal(str)
    finished = Signal(int)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            streams = discover_nvr_streams(
                self.settings,
                can_open_rtsp_channel,
                progress_callback=self.progress.emit,
            )
        except Exception:
            logging.exception("NVR discovery failed")
            self.progress.emit("Discovery failed unexpectedly. See log for details.")
            streams = []

        found_count = 0
        for stream in streams:
            if self._stopped:
                break

            found_count += 1
            self.camera_found.emit(stream)

        self.finished.emit(found_count)


class CameraDetectionWorker(QObject):
    error = Signal(str, str)
    finished = Signal(str)
    frame_ready = Signal(str, bytes)
    status = Signal(str, str)
    analysis = Signal(str, str, str, int)

    def __init__(self, channel, rtsp_url, window_title, target_objects, motion_sensitivity):
        super().__init__()
        self.channel = channel
        self.rtsp_url = rtsp_url
        self.window_title = window_title
        self.target_objects = target_objects
        self.motion_sensitivity = motion_sensitivity
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
                self.motion_sensitivity,
                self.emit_status,
                self.emit_analysis,
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

    def emit_status(self, message):
        self.status.emit(self.channel, message)

    def emit_analysis(self, clip_name, message, progress):
        logging.info(
            "Analysis UI event channel=%s clip=%s progress=%s message=%s",
            self.channel,
            clip_name,
            progress,
            message,
        )
        self.analysis.emit(self.channel, clip_name, message, progress)


class VideoClipDetectionWorker(QObject):
    error = Signal(str, str)
    finished = Signal(str)
    frame_ready = Signal(str, bytes)

    def __init__(self, source_id, video_path, window_title, target_objects):
        super().__init__()
        self.source_id = source_id
        self.video_path = video_path
        self.window_title = window_title
        self.target_objects = target_objects
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            run_video_clip_detection(
                self.video_path,
                self.target_objects,
                self.should_stop,
                self.emit_frame,
            )
        except Exception as exc:
            logging.exception("Video clip %s failed", self.video_path)
            self.error.emit(self.source_id, str(exc))
        finally:
            self.finished.emit(self.source_id)

    def should_stop(self):
        return self._stopped

    def emit_frame(self, frame_bytes):
        self.frame_ready.emit(self.source_id, frame_bytes)


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
        self.setMinimumSize(1040, 620)

        self.scan_thread = None
        self.scan_worker = None
        self.stream_workers = {}
        self.preview_windows = {}
        self.camera_controls = {}

        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.ip_input = QLineEdit()
        guessed_ip = likely_dvr_ip()
        if guessed_ip:
            self.ip_input.setText(guessed_ip)
            self.ip_input.setToolTip("Pre-filled from .env, a local network scan, or the default gateway.")
        self.port_input = QLineEdit(DEFAULT_RTSP_PORT)
        self.rtsp_path_template_input = QLineEdit(DEFAULT_RTSP_PATH_TEMPLATE)
        self.rtsp_path_template_input.setPlaceholderText("/Streaming/Channels/{channel}")
        self.rtsp_path_template_label = QLabel("RTSP Path Template")
        self.rtsp_path_template_label.setVisible(False)
        self.rtsp_path_template_input.setVisible(False)
        self.manual_channel_input = QLineEdit()
        self.video_clip_input = QLineEdit()
        self.video_clip_input.setPlaceholderText("/path/to/test-video.mp4")
        self.object_checkboxes = {}

        for object_name in AVAILABLE_TARGET_OBJECTS:
            checkbox = QCheckBox(object_name.title())
            checkbox.setChecked(object_name in DEFAULT_TARGET_OBJECTS)
            self.object_checkboxes[object_name] = checkbox

        self.camera_list = QListWidget()
        self.camera_list.setMinimumHeight(240)
        self.camera_list.setMinimumWidth(680)
        self.camera_list.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.scan_status = QLabel("Enter NVR details, then connect to discover streams.")
        self.analysis_clip_label = QLabel("Clip: none")
        self.analysis_clip_label.setWordWrap(True)
        self.analysis_step_label = QLabel("Status: idle")
        self.analysis_step_label.setWordWrap(True)
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setRange(0, 100)
        self.analysis_progress.setValue(0)
        self.analysis_current_event = None
        self.scan_button = QPushButton("Connect")
        self.scan_button.clicked.connect(self.scan_cameras)
        self.add_manual_button = QPushButton("Add Manual Channel")
        self.add_manual_button.clicked.connect(self.add_manual_channel)
        self.browse_video_button = QPushButton("Browse Video Clip")
        self.browse_video_button.clicked.connect(self.browse_video_clip)
        self.test_video_button = QPushButton("Annotate Video Clip")
        self.test_video_button.clicked.connect(self.test_video_clip)

        form = QFormLayout()
        form.addRow("Username", self.username_input)
        form.addRow("Password", self.password_input)
        form.addRow("DVR/NVR IP", self.ip_input)
        form.addRow("RTSP Port", self.port_input)
        form.addRow(self.rtsp_path_template_label, self.rtsp_path_template_input)
        form.addRow("Manual Channel", self.manual_channel_input)
        form.addRow("Video Clip", self.video_clip_input)
        for object_name, checkbox in self.object_checkboxes.items():
            form.addRow("Record Objects", checkbox)
        form.addRow("", self.scan_button)
        form.addRow("", self.add_manual_button)
        form.addRow("", self.browse_video_button)
        form.addRow("", self.test_video_button)

        left_column = QVBoxLayout()
        left_column.addLayout(form)
        left_column.addStretch(1)

        right_column = QVBoxLayout()
        right_column.addWidget(self.camera_list)
        right_column.addWidget(self.build_video_analysis_section())
        right_column.addStretch(1)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_column, 2)
        main_layout.addLayout(right_column, 3)
        self.setLayout(main_layout)

    def build_video_analysis_section(self):
        group = QGroupBox("Video Analysis")
        group.setMinimumWidth(320)
        layout = QVBoxLayout()
        layout.addWidget(self.analysis_clip_label)
        layout.addWidget(self.analysis_step_label)
        layout.addWidget(self.analysis_progress)
        group.setLayout(layout)
        return group

    def reject(self):
        if self.is_scanning():
            QMessageBox.warning(self, "Connection Running", "Wait for camera discovery to finish.")
            return

        self.stop_all_streams()
        super().reject()

    def closeEvent(self, event):
        if self.is_scanning():
            QMessageBox.warning(self, "Connection Running", "Wait for camera discovery to finish.")
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
                "Enter username, password, DVR/NVR IP, and RTSP port before connecting.",
            )
            return

        self.stop_all_streams()
        self.camera_list.clear()
        self.camera_controls.clear()
        self.hide_rtsp_template_input()
        self.scan_status.setText("Connecting: trying ONVIF discovery, then RTSP probing...")
        self.scan_button.setEnabled(False)
        self.scan_button.setText("Connecting...")

        self.scan_thread = QThread(self)
        self.scan_worker = CameraScanWorker(settings)
        self.scan_worker.moveToThread(self.scan_thread)

        self.scan_thread.started.connect(self.scan_worker.run)
        self.scan_worker.progress.connect(self.update_scan_status)
        self.scan_worker.camera_found.connect(self.add_found_camera)
        self.scan_worker.finished.connect(self.finish_scan)
        self.scan_worker.finished.connect(self.scan_thread.quit)
        self.scan_worker.finished.connect(self.scan_worker.deleteLater)
        self.scan_thread.finished.connect(self.scan_thread.deleteLater)
        self.scan_thread.finished.connect(self.clear_scan_thread)
        self.scan_thread.start()

    def add_found_camera(self, stream):
        self.add_camera_item(stream.label, stream)
        if stream.rtsp_path_template:
            self.rtsp_path_template_input.setText(stream.rtsp_path_template)
        self.scan_status.setText(f"Found {self.camera_list.count()} camera stream(s)...")

    def update_scan_status(self, message):
        self.scan_status.setText(message)

    def add_camera_item(self, label, stream_data):
        item = QListWidgetItem()
        item.setData(Qt.UserRole, stream_data)
        item.setData(Qt.UserRole + 1, label)
        item.setToolTip(label)
        self.camera_list.addItem(item)

        stream_id = stream_data.stream_id if hasattr(stream_data, "stream_id") else str(stream_data).strip()
        row_widget = QWidget()
        row_widget.setMinimumHeight(50)
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(8, 4, 28, 4)
        row_layout.setSpacing(10)

        stream_checkbox = QCheckBox()
        stream_checkbox.setFixedWidth(24)
        stream_checkbox.setMinimumHeight(30)
        stream_checkbox.stateChanged.connect(lambda _state, list_item=item: self.toggle_stream(list_item))

        stream_label = QLabel(label)
        stream_label.setWordWrap(True)
        stream_label.setMinimumHeight(32)
        stream_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        sensitivity_label = QLabel("Sensitivity")
        sensitivity_label.setMinimumWidth(72)
        sensitivity_input = QSpinBox()
        sensitivity_input.setRange(1, 10)
        sensitivity_input.setValue(DEFAULT_MOTION_SENSITIVITY)
        sensitivity_input.setFixedWidth(72)
        sensitivity_input.setMinimumHeight(30)
        sensitivity_input.setToolTip("1 is least sensitive. 10 is most sensitive. 5 matches the default.")

        row_layout.addWidget(stream_checkbox)
        row_layout.addWidget(stream_label, 1)
        row_layout.addWidget(sensitivity_label)
        row_layout.addWidget(sensitivity_input)
        row_widget.setLayout(row_layout)

        item.setSizeHint(QSize(0, 54))
        self.camera_list.setItemWidget(item, row_widget)
        self.camera_controls[stream_id] = {
            "checkbox": stream_checkbox,
            "label": label,
            "sensitivity": sensitivity_input,
        }

    def finish_scan(self, found_count):
        self.scan_button.setEnabled(True)
        self.scan_button.setText("Connect")

        if found_count:
            self.scan_status.setText(f"Connection complete. Tick a stream to start it. Found {found_count}.")
        else:
            self.show_rtsp_template_input()
            self.scan_status.setText(
                "No streams found by ONVIF or RTSP probing. Enter an RTSP path template and add a channel manually."
            )

    def clear_scan_thread(self):
        self.scan_thread = None
        self.scan_worker = None

    def is_scanning(self):
        return self.scan_thread is not None and self.scan_thread.isRunning()

    def show_rtsp_template_input(self):
        self.rtsp_path_template_label.setVisible(True)
        self.rtsp_path_template_input.setVisible(True)

    def hide_rtsp_template_input(self):
        self.rtsp_path_template_label.setVisible(False)
        self.rtsp_path_template_input.setVisible(False)

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
        self.add_camera_item(
            label,
            DiscoveredStream(
                label=label,
                stream_id=f"manual:{channel}",
                rtsp_path_template=settings["rtsp_path_template"],
                channel=channel,
                source="manual",
            ),
        )
        self.manual_channel_input.clear()
        self.scan_status.setText("Manual channel added. Tick it to start the stream.")

    def browse_video_clip(self):
        video_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Choose Video Clip",
            str(Path.home()),
            "Video Files (*.mp4 *.mov *.avi *.mkv *.m4v);;All Files (*)",
        )
        if video_path:
            self.video_clip_input.setText(video_path)

    def test_video_clip(self):
        video_path_text = self.video_clip_input.text().strip()
        target_objects = self.selected_target_objects()

        if not video_path_text:
            QMessageBox.warning(self, "Missing Video", "Choose a video clip to test.")
            return

        video_path = Path(video_path_text).expanduser()
        if not video_path.is_file():
            QMessageBox.warning(self, "Missing Video", f"Video clip does not exist:\n{video_path}")
            return

        if not target_objects:
            QMessageBox.warning(self, "Missing Objects", "Select at least one object to detect.")
            return

        source_id = f"clip:{video_path}"
        if source_id in self.stream_workers:
            self.scan_status.setText(f"Video clip is already running: {video_path.name}")
            return

        label = f"Video Clip: {video_path.name}"
        thread = QThread(self)
        worker = VideoClipDetectionWorker(source_id, str(video_path), label, target_objects)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.frame_ready.connect(self.update_preview)
        worker.error.connect(self.show_stream_error)
        worker.finished.connect(self.finish_stream)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(lambda channel=source_id: self.cleanup_stream_thread(channel))
        thread.finished.connect(thread.deleteLater)

        self.stream_workers[source_id] = {
            "thread": thread,
            "worker": worker,
            "stopping": False,
        }
        self.add_preview_window(source_id, label)
        self.scan_status.setText(f"Testing video clip: {video_path.name}")
        thread.start()

    def toggle_stream(self, item):
        stream = item.data(Qt.UserRole)
        stream_id = stream.stream_id if hasattr(stream, "stream_id") else str(stream).strip()
        controls = self.camera_controls.get(stream_id)
        if controls is None:
            return

        if controls["checkbox"].isChecked():
            self.start_stream(stream, self.item_label(item), item)
        else:
            self.stop_stream(stream_id)

    def item_label(self, item):
        return item.data(Qt.UserRole + 1) or ""

    def uncheck_stream_item(self, item):
        stream = item.data(Qt.UserRole)
        stream_id = stream.stream_id if hasattr(stream, "stream_id") else str(stream).strip()
        controls = self.camera_controls.get(stream_id)
        if controls is None:
            return

        checkbox = controls["checkbox"]
        checkbox.blockSignals(True)
        checkbox.setChecked(False)
        checkbox.blockSignals(False)

    def item_motion_sensitivity(self, item):
        stream = item.data(Qt.UserRole)
        stream_id = stream.stream_id if hasattr(stream, "stream_id") else str(stream).strip()
        controls = self.camera_controls.get(stream_id)
        if controls is None:
            return DEFAULT_MOTION_SENSITIVITY

        return controls["sensitivity"].value()

    def set_stream_sensitivity_enabled(self, stream_id, enabled):
        controls = self.camera_controls.get(stream_id)
        if controls is not None:
            controls["sensitivity"].setEnabled(enabled)

    def start_stream(self, stream, label, item):
        stream_id = stream.stream_id if hasattr(stream, "stream_id") else str(stream).strip()
        if stream_id in self.stream_workers:
            return

        settings = self.settings()
        target_objects = self.selected_target_objects()
        rtsp_url = getattr(stream, "rtsp_url", None)
        rtsp_path_template = getattr(stream, "rtsp_path_template", None)
        channel = getattr(stream, "channel", None)

        if not rtsp_url and not all(settings.values()):
            QMessageBox.warning(
                self,
                "Missing Details",
                "Enter username, password, DVR/NVR IP, RTSP port, and RTSP path template before starting a stream.",
            )
            self.uncheck_stream_item(item)
            return

        if not target_objects:
            QMessageBox.warning(
                self,
                "Missing Objects",
                "Select at least one object to record.",
            )
            self.uncheck_stream_item(item)
            return

        try:
            if rtsp_url:
                stream_url = rtsp_url
            else:
                stream_settings = dict(settings)
                if rtsp_path_template:
                    stream_settings["rtsp_path_template"] = rtsp_path_template
                stream_url = build_rtsp_url(stream_settings, channel=channel)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid RTSP Template", str(exc))
            self.uncheck_stream_item(item)
            return

        thread = QThread(self)
        motion_sensitivity = self.item_motion_sensitivity(item)
        worker = CameraDetectionWorker(stream_id, stream_url, label, target_objects, motion_sensitivity)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.frame_ready.connect(self.update_preview)
        worker.status.connect(self.update_stream_status)
        worker.analysis.connect(self.update_video_analysis)
        worker.error.connect(self.show_stream_error)
        worker.finished.connect(self.finish_stream)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(lambda channel=stream_id: self.cleanup_stream_thread(channel))
        thread.finished.connect(thread.deleteLater)

        self.stream_workers[stream_id] = {
            "thread": thread,
            "worker": worker,
            "stopping": False,
        }
        self.set_stream_sensitivity_enabled(stream_id, False)
        self.add_preview_window(stream_id, label)
        self.scan_status.setText(f"Started stream {label}.")
        thread.start()

    def update_stream_status(self, channel, message):
        status_message = f"{channel}: {message}"
        self.scan_status.setText(status_message)

    def update_video_analysis(self, channel, clip_name, message, progress):
        logging.info(
            "Updating Video Analysis panel channel=%s clip=%s progress=%s message=%s",
            channel,
            clip_name,
            progress,
            message,
        )
        self.analysis_clip_label.setText(f"Clip: {clip_name}")
        self.analysis_step_label.setText(f"{channel}: {message}")
        if progress < 0:
            self.analysis_progress.setRange(0, 0)
        else:
            if self.analysis_progress.minimum() != 0 or self.analysis_progress.maximum() != 100:
                self.analysis_progress.setRange(0, 100)
            self.analysis_progress.setValue(max(0, min(progress, 100)))
        self.analysis_current_event = (channel, clip_name, message)
        if progress >= 100:
            QTimer.singleShot(
                3000,
                lambda event=self.analysis_current_event: self.reset_video_analysis_if_current(event),
            )

    def reset_video_analysis_if_current(self, event):
        if self.analysis_current_event != event:
            return

        self.analysis_current_event = None
        self.analysis_clip_label.setText("Clip: none")
        self.analysis_step_label.setText("Status: idle")
        if self.analysis_progress.minimum() != 0 or self.analysis_progress.maximum() != 100:
            self.analysis_progress.setRange(0, 100)
        self.analysis_progress.setValue(0)

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
        self.set_stream_sensitivity_enabled(channel, True)
        self.remove_preview_window(channel)
        self.scan_status.setText(f"Stopped stream {channel}.")

    def cleanup_stream_thread(self, channel):
        self.stream_workers.pop(channel, None)

    def uncheck_channel(self, channel):
        for index in range(self.camera_list.count()):
            item = self.camera_list.item(index)
            stream = item.data(Qt.UserRole)
            stream_id = stream.stream_id if hasattr(stream, "stream_id") else str(stream).strip()
            if stream_id != channel:
                continue

            self.uncheck_stream_item(item)
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
