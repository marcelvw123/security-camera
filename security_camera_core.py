import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from itertools import count
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

matplotlib_cache_dir = Path(tempfile.gettempdir()) / "security_camera_matplotlib"
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

import cv2
from cloud_upload import upload_clip_async
from gemma_analysis import analyze_clip_with_gemma
from scenario_detector import ScenarioDetector, scenario_relevant_objects
from video_compat import make_whatsapp_compatible_mp4


MIN_MOTION_AREA = 1000
MAX_MOTION_AREA_RATIO = 0.60
MAX_MOTION_BOX_ASPECT_RATIO = 8.0
MIN_OBJECT_MOTION_OVERLAP_RATIO = 0.08
CLIP_SECONDS = 20
YOLO_MODEL_FILENAME = "yolo11s.pt"
YOLO_CONFIDENCE = 0.55
OUTPUT_DIR = Path.home() / "SecurityCamera" / "clips"
DEBUG_FRAME_DIR = Path.home() / "SecurityCamera" / "debug_frames"
SCENARIO_FRAME_DIR = Path.home() / "SecurityCamera" / "scenario_frames"
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
_TRIGGER_COUNTER = count(1)


@dataclass(frozen=True)
class MotionSensitivityConfig:
    level: int
    min_motion_area: int
    min_object_motion_overlap_ratio: float


def resource_path(filename):
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_path / filename


def noop_frame_callback(_frame_bytes):
    return None


def load_yolo_model():
    from ultralytics import YOLO

    return YOLO(resource_path(YOLO_MODEL_FILENAME))


def new_trigger_id():
    return f"ID{next(_TRIGGER_COUNTER):03d}"


def log_trigger_step(trigger_id, step, message, *args):
    if args:
        message = message % args

    if trigger_id:
        logging.info("%s %s %s", trigger_id, step, message)
    else:
        logging.info("%s %s", step, message)


def motion_sensitivity_config(level=None):
    level = int(level or 5)
    level = max(1, min(level, 10))
    threshold_multiplier = 5 / level
    return MotionSensitivityConfig(
        level=level,
        min_motion_area=max(1, int(MIN_MOTION_AREA * threshold_multiplier)),
        min_object_motion_overlap_ratio=min(1.0, MIN_OBJECT_MOTION_OVERLAP_RATIO * threshold_multiplier),
    )


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


def upload_saved_clip(clip_path, source_name, detected_objects, clip_type, scenario_name, trigger_id=None):
    clip_path = Path(clip_path)
    log_trigger_step(trigger_id, "STEP_7_GEMMA_ANALYSIS_STARTED", "clip=%s", clip_path)
    gemma_analysis = analyze_clip_with_gemma(clip_path, trigger_id=trigger_id)
    if gemma_analysis:
        log_trigger_step(
            trigger_id,
            "STEP_8_GEMMA_ANALYSIS_COMPLETED",
            "break_in_likely=%r confidence=%r",
            gemma_analysis.get("break_in_likely"),
            gemma_analysis.get("confidence"),
        )
    else:
        log_trigger_step(trigger_id, "STEP_8_GEMMA_ANALYSIS_SKIPPED", "clip=%s", clip_path)

    log_trigger_step(trigger_id, "STEP_9_TRANSCODE_STARTED", "clip=%s", clip_path)
    compatible_clip_path = make_whatsapp_compatible_mp4(clip_path)
    log_trigger_step(trigger_id, "STEP_10_TRANSCODE_COMPLETED", "clip=%s", compatible_clip_path)
    metadata = {
        "camera": source_name,
        "detected": sorted(detected_objects),
        "source_name": source_name,
        "detected_objects": sorted(detected_objects),
        "clip_type": clip_type,
        "scenario": scenario_name,
        "trigger_id": trigger_id,
    }
    if gemma_analysis:
        metadata.update(
            {
                "gemma_scene_description": gemma_analysis.get("scene_description"),
                "gemma_break_in_likely": gemma_analysis.get("break_in_likely"),
                "gemma_confidence": gemma_analysis.get("confidence"),
                "gemma_reasoning": gemma_analysis.get("reasoning"),
            }
        )
    upload_clip_async(str(compatible_clip_path), metadata)
    log_trigger_step(trigger_id, "STEP_11_UPLOAD_QUEUED", "clip=%s", compatible_clip_path)


def upload_non_scenario_clip(clip_path, source_name, detected_objects, clip_type, reason, trigger_id=None):
    if clip_path is None:
        return

    log_trigger_step(trigger_id, "STEP_6_CLIP_SAVED_NO_SCENARIO", "clip=%s reason=%s", clip_path, reason)
    upload_saved_clip(
        clip_path,
        source_name,
        detected_objects,
        clip_type,
        "no_scenario_detected",
        trigger_id=trigger_id,
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


def detect_motion(frame, motion_detector, sensitivity_config=None):
    sensitivity_config = sensitivity_config or motion_sensitivity_config()
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height
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
        if cv2.contourArea(contour) < sensitivity_config.min_motion_area:
            continue

        motion_boxes.append(cv2.boundingRect(contour))

    return [
        motion_box
        for motion_box in motion_boxes
        if not is_motion_artifact(motion_box, frame_area)
    ]


def is_motion_artifact(motion_box, frame_area):
    _x, _y, width, height = motion_box
    if width <= 0 or height <= 0:
        return True

    box_area = width * height
    if frame_area > 0 and box_area / frame_area > MAX_MOTION_AREA_RATIO:
        return True

    aspect_ratio = max(width / height, height / width)
    return aspect_ratio > MAX_MOTION_BOX_ASPECT_RATIO


def draw_motion_boxes(frame, motion_boxes):
    for x, y, width, height in motion_boxes:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 255), 2)


def boxes_overlap(box_a, box_b):
    left_a, top_a, right_a, bottom_a = box_a
    left_b, top_b, right_b, bottom_b = box_b
    return left_a < right_b and right_a > left_b and top_a < bottom_b and bottom_a > top_b


def overlap_area(box_a, box_b):
    left_a, top_a, right_a, bottom_a = box_a
    left_b, top_b, right_b, bottom_b = box_b
    overlap_width = max(0, min(right_a, right_b) - max(left_a, left_b))
    overlap_height = max(0, min(bottom_a, bottom_b) - max(top_a, top_b))
    return overlap_width * overlap_height


def motion_box_to_xyxy(motion_box):
    x, y, width, height = motion_box
    return x, y, x + width, y + height


def detection_box_to_xyxy(box):
    left, top, right, bottom = box.xyxy[0].tolist()
    return int(left), int(top), int(right), int(bottom)


def detection_overlaps_motion(box, motion_boxes, sensitivity_config=None):
    sensitivity_config = sensitivity_config or motion_sensitivity_config()
    detection_box = detection_box_to_xyxy(box)
    detection_area = max(1, (detection_box[2] - detection_box[0]) * (detection_box[3] - detection_box[1]))
    total_overlap_area = sum(
        overlap_area(detection_box, motion_box_to_xyxy(motion_box))
        for motion_box in motion_boxes
    )
    return total_overlap_area / detection_area >= sensitivity_config.min_object_motion_overlap_ratio


def draw_detection_status(frame, motion_detected, matched_objects):
    status_lines = [
        f"Motion: {'YES' if motion_detected else 'NO'}",
        f"Objects: {', '.join(sorted(matched_objects)) if matched_objects else 'none'}",
    ]
    for index, text in enumerate(status_lines):
        y = 30 + (index * 28)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def save_trigger_debug_frame(frame, trigger_id, source_name, objects):
    DEBUG_FRAME_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_source_name = "".join(
        character if character.isalnum() else "_"
        for character in source_name
    ).strip("_")
    safe_objects = "_".join(sorted(objects)) or "objects"
    debug_frame_path = DEBUG_FRAME_DIR / f"{trigger_id}_{safe_source_name}_{safe_objects}_{timestamp}.jpg"
    if cv2.imwrite(str(debug_frame_path), frame):
        log_trigger_step(trigger_id, "STEP_3_DEBUG_FRAME_SAVED", "image=%s", debug_frame_path)
    else:
        log_trigger_step(trigger_id, "STEP_3_DEBUG_FRAME_SAVE_FAILED", "image=%s", debug_frame_path)


def save_scenario_frame(frame, trigger_id, source_name, scenario_name):
    SCENARIO_FRAME_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_source_name = "".join(
        character if character.isalnum() else "_"
        for character in source_name
    ).strip("_")
    safe_scenario_name = "".join(
        character if character.isalnum() else "_"
        for character in scenario_name
    ).strip("_")
    scenario_frame_path = SCENARIO_FRAME_DIR / f"{trigger_id}_{safe_source_name}_{safe_scenario_name}_{timestamp}.jpg"
    if cv2.imwrite(str(scenario_frame_path), frame):
        log_trigger_step(trigger_id, "STEP_5_SCENARIO_FRAME_SAVED", "image=%s", scenario_frame_path)
    else:
        log_trigger_step(trigger_id, "STEP_5_SCENARIO_FRAME_SAVE_FAILED", "image=%s", scenario_frame_path)


def run_video_clip_detection(video_path, target_objects, stop_requested, frame_callback=noop_frame_callback, motion_sensitivity=None):
    sensitivity_config = motion_sensitivity_config(motion_sensitivity)
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

            motion_boxes = detect_motion(frame, motion_detector, sensitivity_config)
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


def run_detection(rtsp_url, window_title, target_objects, stop_requested, frame_callback=noop_frame_callback, motion_sensitivity=None):
    sensitivity_config = motion_sensitivity_config(motion_sensitivity)
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
    active_trigger_id = None
    pending_trigger_id = None
    frames_left_in_clip = 0
    frame_count = 0
    consecutive_read_failures = 0
    logging.info(
        "Motion sensitivity for %s: level=%s min_motion_area=%s min_object_motion_overlap_ratio=%.3f",
        window_title,
        sensitivity_config.level,
        sensitivity_config.min_motion_area,
        sensitivity_config.min_object_motion_overlap_ratio,
    )

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
                        log_trigger_step(active_trigger_id, "STEP_6_CLIP_SAVED", "clip=%s reason=stream_lost", active_clip_path)
                        upload_saved_clip(
                            active_clip_path,
                            window_title,
                            active_clip_objects,
                            "scenario_camera_detection",
                            active_clip_scenario,
                            trigger_id=active_trigger_id,
                        )
                    elif active_clip_path is not None:
                        upload_non_scenario_clip(
                            active_clip_path,
                            window_title,
                            active_clip_objects,
                            "camera_detection",
                            "stream lost before scenario matched",
                            trigger_id=active_trigger_id,
                        )
                    writer = None
                    active_clip_path = None
                    active_clip_objects = set()
                    active_clip_scenario = None
                    active_trigger_id = None
                    pending_trigger_id = None
                    frames_left_in_clip = 0
                    scenario_detector = ScenarioDetector(window_seconds=CLIP_SECONDS, cooldown_seconds=0)

                cap.release()
                cap = None
                wait_before_reconnect(stop_requested)
                continue

            consecutive_read_failures = 0

            motion_boxes = detect_motion(frame, motion_detector, sensitivity_config)
            motion_detected = bool(motion_boxes)
            matched_objects = set()
            relevant_objects = set()
            moving_target_objects = set()
            scenario_match = None
            detected_object_names = []
            annotated_frame = frame.copy()
            draw_motion_boxes(annotated_frame, motion_boxes)

            if motion_detected:
                if writer is None and pending_trigger_id is None:
                    pending_trigger_id = new_trigger_id()
                    log_trigger_step(
                        pending_trigger_id,
                        "STEP_1_MOTION_DETECTED",
                        "camera=%s motion_boxes=%s",
                        window_title,
                        len(motion_boxes),
                    )

                results = model(frame, verbose=False, conf=YOLO_CONFIDENCE)

                annotated_frame = results[0].plot()
                draw_motion_boxes(annotated_frame, motion_boxes)
                detected_object_names = [
                    model.names[int(box.cls[0])]
                    for box in results[0].boxes
                ]
                moving_detected_object_names = [
                    model.names[int(box.cls[0])]
                    for box in results[0].boxes
                    if detection_overlaps_motion(box, motion_boxes, sensitivity_config)
                ]
                detected_objects = set(detected_object_names)
                moving_detected_objects = set(moving_detected_object_names)
                matched_objects = detected_objects & target_objects
                moving_target_objects = moving_detected_objects & target_objects
                relevant_objects = scenario_relevant_objects(detected_object_names)
                scenario_match = scenario_detector.record_detection(window_title, detected_object_names)
                if writer is None and pending_trigger_id:
                    if detected_object_names:
                        log_trigger_step(
                            pending_trigger_id,
                            "STEP_2_OBJECTS_DETECTED",
                            "objects=%s",
                            sorted(detected_objects),
                        )
                    else:
                        log_trigger_step(pending_trigger_id, "STEP_2_NO_OBJECTS_DETECTED", "camera=%s", window_title)

            draw_detection_status(annotated_frame, motion_detected, matched_objects)

            if writer is None and moving_target_objects:
                if pending_trigger_id is None:
                    pending_trigger_id = new_trigger_id()
                log_trigger_step(
                    pending_trigger_id,
                    "STEP_3_MOVING_TARGET_OBJECTS_DETECTED",
                    "objects=%s",
                    sorted(moving_target_objects),
                )
                save_trigger_debug_frame(
                    annotated_frame,
                    pending_trigger_id,
                    window_title,
                    moving_target_objects,
                )
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_window_title = "".join(
                    character if character.isalnum() else "_"
                    for character in window_title
                ).strip("_")
                clip_path = OUTPUT_DIR / f"pending_scenario_{safe_window_title}_{timestamp}.mp4"
                height, width = annotated_frame.shape[:2]
                writer = create_video_writer(clip_path, fps, (width, height))
                active_clip_path = clip_path
                active_clip_objects = set(moving_target_objects)
                active_trigger_id = pending_trigger_id
                pending_trigger_id = None
                frames_left_in_clip = int(fps * CLIP_SECONDS)
                log_trigger_step(
                    active_trigger_id,
                    "STEP_4_CLIP_RECORDING_STARTED",
                    "camera=%s clip=%s seconds=%s",
                    window_title,
                    clip_path,
                    CLIP_SECONDS,
                )
            elif writer is None and pending_trigger_id and motion_detected:
                log_trigger_step(
                    pending_trigger_id,
                    "STEP_3_NO_MOVING_TARGET_OBJECTS_DETECTED",
                    "detected_objects=%s moving_target_objects=%s",
                    sorted(detected_object_names),
                    sorted(moving_target_objects),
                )
                pending_trigger_id = None

            if writer is not None:
                active_clip_objects.update(relevant_objects)

            if active_clip_scenario is None and scenario_match is not None:
                active_clip_objects = set(scenario_match.objects)
                active_clip_scenario = scenario_match.name
                log_trigger_step(
                    active_trigger_id,
                    "STEP_5_SCENARIO_MATCHED",
                    "scenario=%s clip=%s",
                    active_clip_scenario,
                    active_clip_path,
                )
                save_scenario_frame(
                    annotated_frame,
                    active_trigger_id,
                    window_title,
                    active_clip_scenario,
                )

            if writer is not None:
                writer.write(annotated_frame)
                frames_left_in_clip -= 1

                if frames_left_in_clip <= 0:
                    writer.release()
                    if active_clip_path is not None and active_clip_scenario is not None:
                        log_trigger_step(active_trigger_id, "STEP_6_CLIP_SAVED", "clip=%s", active_clip_path)
                        upload_saved_clip(
                            active_clip_path,
                            window_title,
                            active_clip_objects,
                            "scenario_camera_detection",
                            active_clip_scenario,
                            trigger_id=active_trigger_id,
                        )
                    elif active_clip_path is not None:
                        upload_non_scenario_clip(
                            active_clip_path,
                            window_title,
                            active_clip_objects,
                            "camera_detection",
                            "20 second clip ended without scenario match",
                            trigger_id=active_trigger_id,
                        )
                    writer = None
                    active_clip_path = None
                    active_clip_objects = set()
                    active_clip_scenario = None
                    active_trigger_id = None
                    pending_trigger_id = None
                    frames_left_in_clip = 0
                    scenario_detector = ScenarioDetector(window_seconds=CLIP_SECONDS, cooldown_seconds=0)

            frame_count += 1
            if frame_count % PREVIEW_EVERY_N_FRAMES == 0:
                emit_preview_frame(annotated_frame, frame_callback)
    finally:
        if writer is not None:
            writer.release()
            if active_clip_path is not None and active_clip_scenario is not None:
                log_trigger_step(active_trigger_id, "STEP_6_CLIP_SAVED", "clip=%s reason=stream_stopped", active_clip_path)
                upload_saved_clip(
                    active_clip_path,
                    window_title,
                    active_clip_objects,
                    "scenario_camera_detection",
                    active_clip_scenario,
                    trigger_id=active_trigger_id,
                )
            elif active_clip_path is not None:
                upload_non_scenario_clip(
                    active_clip_path,
                    window_title,
                    active_clip_objects,
                    "camera_detection",
                    "stream stopped before scenario matched",
                    trigger_id=active_trigger_id,
                )

        if cap is not None:
            cap.release()
