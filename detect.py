import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import cv2
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)
from ultralytics import YOLO

MIN_MOTION_AREA = 1000
CLIP_SECONDS = 10
OUTPUT_DIR = Path("clips")
TARGET_OBJECTS = {"car"}
DEFAULT_CHANNEL = "702"
DEFAULT_RTSP_PORT = "554"


def resource_path(filename):
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_path / filename


class CameraSettingsDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Security Camera Setup")
        self.setMinimumWidth(420)

        self.username_input = QLineEdit("admin")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("192.168.68.103")
        self.port_input = QLineEdit(DEFAULT_RTSP_PORT)
        self.channel_input = QLineEdit(DEFAULT_CHANNEL)

        form = QFormLayout()
        form.addRow("Username", self.username_input)
        form.addRow("Password", self.password_input)
        form.addRow("DVR/NVR IP", self.ip_input)
        form.addRow("RTSP Port", self.port_input)
        form.addRow("Channel", self.channel_input)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        buttons.button(QDialogButtonBox.Ok).setText("Start")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def accept(self):
        if not all(self.settings().values()):
            QMessageBox.warning(self, "Missing Details", "Please complete all fields.")
            return

        super().accept()

    def settings(self):
        return {
            "username": self.username_input.text().strip(),
            "password": self.password_input.text(),
            "ip_address": self.ip_input.text().strip(),
            "port": self.port_input.text().strip(),
            "channel": self.channel_input.text().strip(),
        }


def get_camera_settings():
    dialog = CameraSettingsDialog()
    if dialog.exec() != QDialog.Accepted:
        return None

    return dialog.settings()


def build_rtsp_url(settings):
    username = quote(settings["username"], safe="")
    password = quote(settings["password"], safe="")
    ip_address = settings["ip_address"]
    port = settings["port"]
    channel = settings["channel"]

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


def run_detection(rtsp_url):
    model = YOLO(resource_path("yolov8n.pt"))
    motion_detector = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=True,
    )

    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        QMessageBox.critical(None, "Connection Failed", "Could not open RTSP stream.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 20

    writer = None
    frames_left_in_clip = 0

    try:
        while True:
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
                clip_path = OUTPUT_DIR / f"car_{timestamp}.mp4"
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

            cv2.imshow("Hikvision Object Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if writer is not None:
            writer.release()

        cap.release()

        cv2.destroyAllWindows()


def main():
    app = QApplication(sys.argv)

    settings = get_camera_settings()
    if settings is None:
        return

    rtsp_url = build_rtsp_url(settings)
    run_detection(rtsp_url)
    app.quit()


if __name__ == "__main__":
    main()
