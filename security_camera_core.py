import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

matplotlib_cache_dir = Path(tempfile.gettempdir()) / "security_camera_matplotlib"
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

import cv2
from cloud_upload import upload_clip_async
from scenario_detector import ScenarioDetector, scenario_relevant_objects
from video_compat import make_whatsapp_compatible_mp4


MIN_MOTION_AREA = 1000
CLIP_SECONDS = 20
YOLO_MODEL_FILENAME = "yolo11s.pt"
YOLO_CONFIDENCE = 0.55
OUTPUT_DIR = Path.home() / "SecurityCamera" / "clips"
DEFAULT_TARGET_OBJECTS = {"person"}
AVAILABLE_TARGET_OBJECTS = ("car", "person", "truck", "bus", "motorcycle")
DEFAULT_RTSP_PORT = "554"
DEFAULT_RTSP_PATH_TEMPLATE = "/Streaming/Channels/{channel}"
STREAM_OPEN_TIMEOUT_MS = 1200
STREAM_READ_TIMEOUT_MS = 1200
STREAM_RECONNECT_DELAY_SECONDS = 2
READ_FAILURES_BEFORE_RECONNECT = 3
PREVIEW_EVERY_N_FRAMES = 3
JPEG_PREVIEW_QUALITY = 70


def resource_path(filename):
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_path / filename


def noop_frame_callback(_frame_bytes):
    return None


def load_yolo_model():
    from ultralytics import YOLO

    return YOLO(resource_path(YOLO_MODEL_FILENAME))


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
            logging.info("Using video codec: %s", codec)
            return writer

        writer.release()

    raise RuntimeError("Could not create video writer")


def upload_saved_clip(clip_path, source_name, detected_objects, clip_type, scenario_name):
    compatible_clip_path = make_whatsapp_compatible_mp4(clip_path)
    metadata = {
        "camera": source_name,
        "detected": sorted(detected_objects),
        "source_name": source_name,
        "detected_objects": sorted(detected_objects),
        "clip_type": clip_type,
        "scenario": scenario_name,
    }
    upload_clip_async(str(compatible_clip_path), metadata)


def upload_non_scenario_clip(clip_path, source_name, detected_objects, clip_type, reason):
    if clip_path is None:
        return

    logging.info("Uploading non-scenario clip %s: %s", clip_path, reason)
    upload_saved_clip(
        clip_path,
        source_name,
        detected_objects,
        clip_type,
        "no_scenario_detected",
    )


def target_class_ids(model, target_objects):
    names = model.names.items() if hasattr(model.names, "items") else enumerate(model.names)
    return [
        class_id
        for class_id, class_name in names
        if class_name in target_objects
    ]


def emit_preview_frame(frame, frame_callback):
    if frame_callback is None:
        return

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_PREVIEW_QUALITY]
    encoded, buffer = cv2.imencode(".jpg", frame, encode_params)
    if encoded:
        frame_callback(buffer.tobytes())


def detect_motion(frame, motion_detector):
    motion_mask = motion_detector.apply(frame)
    motion_mask = cv2.threshold(motion_mask, 244, 255, cv2.THRESH_BINARY)[1]
    motion_mask = cv2.dilate(motion_mask, None, iterations=2)

    contours, _ = cv2.findContours(
        motion_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    motion_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_MOTION_AREA:
            continue

        motion_boxes.append(cv2.boundingRect(contour))

    return motion_boxes


def draw_motion_boxes(frame, motion_boxes):
    for x, y, width, height in motion_boxes:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 255), 2)


def draw_detection_status(frame, motion_detected, matched_objects):
    status_lines = [
        f"Motion: {'YES' if motion_detected else 'NO'}",
        f"Objects: {', '.join(sorted(matched_objects)) if matched_objects else 'none'}",
    ]
    for index, text in enumerate(status_lines):
        y = 30 + (index * 28)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def run_video_clip_detection(video_path, target_objects, stop_requested, frame_callback=noop_frame_callback):
    model = load_yolo_model()
    class_ids = target_class_ids(model, target_objects)
    scenario_detector = ScenarioDetector(window_seconds=CLIP_SECONDS, cooldown_seconds=0)
    motion_detector = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=True,
    )
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video clip: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 20
    frame_delay = min(1 / fps, 0.1)
    writer = None
    clip_path = None
    clip_objects = set()
    scenario_match = None
    frames_left_in_clip = 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        while not stop_requested():
            ret, frame = cap.read()
            if not ret:
                break

            motion_boxes = detect_motion(frame, motion_detector)
            motion_detected = bool(motion_boxes)
            results = model(frame, verbose=False, conf=YOLO_CONFIDENCE, classes=class_ids)
            detected_object_names = [
                model.names[int(box.cls[0])]
                for box in results[0].boxes
            ]
            detected_objects = set(detected_object_names)
            matched_objects = detected_objects & target_objects
            relevant_objects = scenario_relevant_objects(detected_object_names)
            current_scenario_match = scenario_detector.record_detection(Path(video_path).name, detected_object_names)
            annotated_frame = results[0].plot()
            draw_motion_boxes(annotated_frame, motion_boxes)
            draw_detection_status(annotated_frame, motion_detected, matched_objects)

            if writer is None and relevant_objects:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_video_name = "".join(
                    character if character.isalnum() else "_"
                    for character in Path(video_path).stem
                ).strip("_")
                clip_path = OUTPUT_DIR / f"pending_scenario_{safe_video_name}_{timestamp}.mp4"
                height, width = annotated_frame.shape[:2]
                writer = create_video_writer(clip_path, fps, (width, height))
                clip_objects = set(relevant_objects)
                frames_left_in_clip = int(fps * CLIP_SECONDS)
                logging.info("Started pending scenario video clip: %s", clip_path)

            if writer is not None:
                clip_objects.update(relevant_objects)

            if scenario_match is None and current_scenario_match is not None:
                scenario_match = current_scenario_match
                logging.info("Scenario matched for pending video clip: %s", clip_path)

            if writer is not None:
                writer.write(annotated_frame)
                frames_left_in_clip -= 1

                if frames_left_in_clip <= 0:
                    writer.release()
                    if clip_path is not None and scenario_match is not None:
                        upload_saved_clip(
                            clip_path,
                            Path(video_path).name,
                            scenario_match.objects,
                            "scenario_video_test",
                            scenario_match.name,
                        )
                    elif clip_path is not None:
                        upload_non_scenario_clip(
                            clip_path,
                            Path(video_path).name,
                            clip_objects,
                            "video_test",
                            "20 second clip ended without scenario match",
                        )

                    writer = None
                    clip_path = None
                    clip_objects = set()
                    scenario_match = None
                    frames_left_in_clip = 0
                    scenario_detector = ScenarioDetector(window_seconds=CLIP_SECONDS, cooldown_seconds=0)
            emit_preview_frame(annotated_frame, frame_callback)
            time.sleep(frame_delay)
    finally:
        if writer is not None:
            writer.release()
            if clip_path is not None and scenario_match is not None:
                upload_saved_clip(
                    clip_path,
                    Path(video_path).name,
                    scenario_match.objects,
                    "scenario_video_test",
                    scenario_match.name,
                )
            elif clip_path is not None:
                upload_non_scenario_clip(
                    clip_path,
                    Path(video_path).name,
                    clip_objects,
                    "video_test",
                    "video ended before scenario matched",
                )

        cap.release()


def run_detection(rtsp_url, window_title, target_objects, stop_requested, frame_callback=noop_frame_callback):
    model = load_yolo_model()
    scenario_detector = ScenarioDetector(window_seconds=CLIP_SECONDS, cooldown_seconds=0)
    motion_detector = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = None
    fps = 20
    writer = None
    active_clip_path = None
    active_clip_objects = set()
    active_clip_scenario = None
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
                    if active_clip_path is not None and active_clip_scenario is not None:
                        upload_saved_clip(
                            active_clip_path,
                            window_title,
                            active_clip_objects,
                            "scenario_camera_detection",
                            active_clip_scenario,
                        )
                    elif active_clip_path is not None:
                        upload_non_scenario_clip(
                            active_clip_path,
                            window_title,
                            active_clip_objects,
                            "camera_detection",
                            "stream lost before scenario matched",
                        )
                    writer = None
                    active_clip_path = None
                    active_clip_objects = set()
                    active_clip_scenario = None
                    frames_left_in_clip = 0
                    scenario_detector = ScenarioDetector(window_seconds=CLIP_SECONDS, cooldown_seconds=0)

                cap.release()
                cap = None
                wait_before_reconnect(stop_requested)
                continue

            consecutive_read_failures = 0

            motion_boxes = detect_motion(frame, motion_detector)
            motion_detected = bool(motion_boxes)
            matched_objects = set()
            relevant_objects = set()
            scenario_match = None
            annotated_frame = frame.copy()
            draw_motion_boxes(annotated_frame, motion_boxes)

            if motion_detected:
                results = model(frame, verbose=False, conf=YOLO_CONFIDENCE)

                annotated_frame = results[0].plot()
                draw_motion_boxes(annotated_frame, motion_boxes)
                detected_object_names = [
                    model.names[int(box.cls[0])]
                    for box in results[0].boxes
                ]
                detected_objects = set(detected_object_names)
                matched_objects = detected_objects & target_objects
                relevant_objects = scenario_relevant_objects(detected_object_names)
                scenario_match = scenario_detector.record_detection(window_title, detected_object_names)

            draw_detection_status(annotated_frame, motion_detected, matched_objects)

            if writer is None and relevant_objects:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_window_title = "".join(
                    character if character.isalnum() else "_"
                    for character in window_title
                ).strip("_")
                clip_path = OUTPUT_DIR / f"pending_scenario_{safe_window_title}_{timestamp}.mp4"
                height, width = annotated_frame.shape[:2]
                writer = create_video_writer(clip_path, fps, (width, height))
                active_clip_path = clip_path
                active_clip_objects = set(relevant_objects)
                frames_left_in_clip = int(fps * CLIP_SECONDS)
                logging.info("Started pending scenario clip for %s: %s", window_title, clip_path)

            if writer is not None:
                active_clip_objects.update(relevant_objects)

            if active_clip_scenario is None and scenario_match is not None:
                active_clip_objects = set(scenario_match.objects)
                active_clip_scenario = scenario_match.name
                logging.info("Scenario matched for pending clip %s: %s", window_title, active_clip_path)

            if writer is not None:
                writer.write(annotated_frame)
                frames_left_in_clip -= 1

                if frames_left_in_clip <= 0:
                    writer.release()
                    if active_clip_path is not None and active_clip_scenario is not None:
                        upload_saved_clip(
                            active_clip_path,
                            window_title,
                            active_clip_objects,
                            "scenario_camera_detection",
                            active_clip_scenario,
                        )
                    elif active_clip_path is not None:
                        upload_non_scenario_clip(
                            active_clip_path,
                            window_title,
                            active_clip_objects,
                            "camera_detection",
                            "20 second clip ended without scenario match",
                        )
                    writer = None
                    active_clip_path = None
                    active_clip_objects = set()
                    active_clip_scenario = None
                    frames_left_in_clip = 0
                    scenario_detector = ScenarioDetector(window_seconds=CLIP_SECONDS, cooldown_seconds=0)

            frame_count += 1
            if frame_count % PREVIEW_EVERY_N_FRAMES == 0:
                emit_preview_frame(annotated_frame, frame_callback)
    finally:
        if writer is not None:
            writer.release()
            if active_clip_path is not None and active_clip_scenario is not None:
                upload_saved_clip(
                    active_clip_path,
                    window_title,
                    active_clip_objects,
                    "scenario_camera_detection",
                    active_clip_scenario,
                )
            elif active_clip_path is not None:
                upload_non_scenario_clip(
                    active_clip_path,
                    window_title,
                    active_clip_objects,
                    "camera_detection",
                    "stream stopped before scenario matched",
                )

        if cap is not None:
            cap.release()
