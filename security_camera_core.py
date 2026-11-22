import logging
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from itertools import count
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Set
from urllib.parse import quote

from motion_difference import FrameDifferenceMotion
from motion_preroll import AsyncPreRollWriter, MotionPreRoll
from motion_shadow import MotionShadowComparison
from motion_clip_metadata import start_metadata, update_metadata, finish_metadata, load_metadata
from runtime_warnings import configure_runtime_warnings

configure_runtime_warnings()

matplotlib_cache_dir = Path(tempfile.gettempdir()) / "security_camera_matplotlib"
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

import cv2
from app_config import load_dotenv
from cloud_upload import upload_clip_async, upload_file_async
from gemma_analysis import analyze_clip_with_gemma
from scenario_detector import SCENARIO_OBJECTS, ScenarioDetector, ScenarioMatch, scenario_relevant_objects
from video_compat import make_whatsapp_compatible_mp4


load_dotenv()


def int_env(name, default, min_value=None, max_value=None):
    value = os.getenv(name, "").strip()
    if not value:
        return default

    try:
        parsed = int(value)
    except ValueError:
        logging.warning("Invalid %s value %r; using %s", name, value, default)
        return default

    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


def float_env(name, default, min_value=None, max_value=None):
    value = os.getenv(name, "").strip()
    if not value:
        return default

    try:
        parsed = float(value)
    except ValueError:
        logging.warning("Invalid %s value %r; using %s", name, value, default)
        return default

    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


def bool_env(name, default=False):
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default

    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    logging.warning("Invalid %s value %r; using %s", name, value, default)
    return default


def parse_class_aliases(value):
    aliases = {}
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        separator = "=" if "=" in item else ":"
        if separator not in item:
            logging.warning("Ignoring invalid YOLO class alias %r", item)
            continue
        source, target = [part.strip().lower() for part in item.split(separator, 1)]
        if source and target:
            aliases[source] = target
    return aliases


MIN_MOTION_AREA = 1000
MAX_MOTION_AREA_RATIO = 0.60
MAX_MOTION_BOX_ASPECT_RATIO = 8.0
CLIP_SECONDS = 20
YOLO_MODEL_FILENAME = os.getenv("YOLO_MODEL_FILENAME", "models/yolo11n.pt").strip() or "models/yolo11n.pt"
YOLO_CONFIDENCE = float_env("YOLO_CONFIDENCE", 0.55, min_value=0.0, max_value=1.0)
YOLO_IMAGE_SIZE = int_env("YOLO_IMAGE_SIZE", 640, min_value=320, max_value=1920)
YOLO_CLASS_ALIASES = parse_class_aliases(
    os.getenv("YOLO_CLASS_ALIASES", "pedestrian=person,human=person,people=person,pedestrians=person")
)
OUTPUT_DIR = Path.home() / "SecurityCamera" / "clips"
OBJECT_OUTPUT_DIR = Path.home() / "SecurityCamera" / "object_clips"
DEBUG_FRAME_DIR = Path.home() / "SecurityCamera" / "debug_frames"
SCENARIO_FRAME_DIR = Path.home() / "SecurityCamera" / "scenario_frames"
AVAILABLE_TARGET_OBJECTS = ("car", "person", "truck", "bus", "motorcycle")
DEFAULT_TARGET_OBJECTS = set(AVAILABLE_TARGET_OBJECTS)
DEFAULT_RTSP_PORT = "554"
DEFAULT_RTSP_PATH_TEMPLATE = "/Streaming/Channels/{channel}"
STREAM_OPEN_TIMEOUT_MS = int_env("RTSP_OPEN_TIMEOUT_MS", 5000, min_value=500)
STREAM_READ_TIMEOUT_MS = int_env("RTSP_READ_TIMEOUT_MS", 5000, min_value=500)
STREAM_RECONNECT_DELAY_SECONDS = float_env("RTSP_RECONNECT_DELAY_SECONDS", 2, min_value=0.1)
READ_FAILURES_BEFORE_RECONNECT = int_env("RTSP_READ_FAILURES_BEFORE_RECONNECT", 10, min_value=1)
DEFAULT_RECORDING_FPS = int_env("RECORDING_FPS", 20, min_value=1, max_value=60)
UPLOAD_ON_MOTION = bool_env("UPLOAD_ON_MOTION", False)
DETECT_DURING_RECORDING = bool_env("DETECT_DURING_RECORDING", False)
DETECTION_INTERVAL_SECONDS = float_env("DETECTION_INTERVAL_SECONDS", 0.5, min_value=0.0)
MOTION_WARMUP_SECONDS = float_env("MOTION_WARMUP_SECONDS", 5.0, min_value=0.0)
MOTION_PREROLL_SECONDS = float_env("MOTION_PREROLL_SECONDS", 3.0, min_value=0.0, max_value=10.0)
MOTION_QUIET_SECONDS = float_env("MOTION_QUIET_SECONDS", 5.0, min_value=0.5)
MOTION_SHADOW_ENABLED = bool_env("MOTION_SHADOW_ENABLED", False)
MOTION_ALGORITHM = os.getenv("MOTION_ALGORITHM", "background").strip().lower()
MOTION_MAX_CLIPS = int_env("MOTION_MAX_CLIPS", 500, min_value=1)
MOTION_MIN_FREE_MB = int_env("MOTION_MIN_FREE_MB", 2048, min_value=100)
RECORDING_READ_RETRY_SLEEP_SECONDS = float_env("RECORDING_READ_RETRY_SLEEP_SECONDS", 0.05, min_value=0.0)
DEBUG_FRAMES_CONTAINER = os.getenv("DEBUG_FRAMES_CONTAINER", "debugframes").strip() or "debugframes"
SCENARIO_FRAMES_CONTAINER = os.getenv("SCENARIO_FRAMES_CONTAINER", "scenarioframes").strip() or "scenarioframes"
PREVIEW_EVERY_N_FRAMES = 3
JPEG_PREVIEW_QUALITY = 70
_TRIGGER_COUNTER = count(1)


@dataclass(frozen=True)
class MotionSensitivityConfig:
    level: int
    min_motion_area: int


@dataclass(frozen=True)
class ClipYoloAnalysis:
    scenario_match: Optional[ScenarioMatch]
    detected_objects: Set[str]
    annotated_clip_path: Optional[Path] = None


class ManagedVideoWriter:
    def __init__(self, writer, path, size):
        self.writer = writer
        self.path = Path(path)
        self.size = size

    def write(self, frame):
        height, width = frame.shape[:2]
        target_width, target_height = self.size
        if (width, height) != self.size:
            frame = cv2.resize(frame, (target_width, target_height))
        self.writer.write(frame)

    def release(self):
        self.writer.release()


def resource_path(filename):
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_path / filename


def yolo_model_path(model_filename=None):
    model_path = Path(model_filename or YOLO_MODEL_FILENAME).expanduser()
    if model_path.is_absolute():
        return model_path

    local_path = Path(__file__).resolve().parent / model_path
    if local_path.exists():
        return local_path

    return resource_path(str(model_filename or YOLO_MODEL_FILENAME))


def ensure_legacy_yolov5_path():
    legacy_yolov5_path = Path(__file__).resolve().parent / "vendor" / "LLVIP" / "yolov5"
    if not legacy_yolov5_path.is_dir():
        return

    legacy_yolov5_path = str(legacy_yolov5_path)
    if legacy_yolov5_path not in sys.path:
        sys.path.insert(0, legacy_yolov5_path)


def noop_frame_callback(_frame_bytes):
    return None


def noop_status_callback(_message):
    return None


def noop_analysis_callback(_clip_name, _message, _progress):
    return None


def load_yolo_model(model_filename=None):
    ensure_legacy_yolov5_path()
    from ultralytics import YOLO

    model_path = yolo_model_path(model_filename)
    logging.info(
        "Loading YOLO model: path=%s imgsz=%s confidence=%s aliases=%s",
        model_path,
        YOLO_IMAGE_SIZE,
        YOLO_CONFIDENCE,
        YOLO_CLASS_ALIASES,
    )
    return YOLO(model_path)


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


def open_video_file_capture(video_path, description="video clip"):
    video_path = Path(video_path).expanduser()
    try:
        video_path = video_path.resolve()
    except OSError:
        pass

    if not video_path.is_file():
        raise RuntimeError(f"Could not open {description}: file does not exist: {video_path}")

    file_size = video_path.stat().st_size
    if file_size <= 0:
        raise RuntimeError(f"Could not open {description}: file is empty: {video_path}")

    logging.info("Opening %s with OpenCV: path=%s size=%s bytes", description, video_path, file_size)
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        backend_name = "unknown"
        try:
            backend_name = cap.getBackendName()
        except cv2.error:
            pass
        logging.info("Opened %s with OpenCV backend: %s", description, backend_name)
        return cap, video_path

    cap.release()
    raise RuntimeError(f"Could not open {description} with OpenCV: {video_path}")


def wait_before_reconnect(stop_requested):
    end_time = time.monotonic() + STREAM_RECONNECT_DELAY_SECONDS
    while not stop_requested() and time.monotonic() < end_time:
        time.sleep(0.1)


def normalized_recording_fps(capture_fps):
    if not capture_fps or capture_fps <= 0 or capture_fps > 60:
        return DEFAULT_RECORDING_FPS
    return capture_fps


def create_video_writer(clip_path, fps, size):
    clip_path = Path(clip_path)
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if clip_path.exists():
            clip_path.unlink()
        fallback_path = clip_path.with_suffix(".avi")
        if fallback_path != clip_path and fallback_path.exists():
            fallback_path.unlink()
    except OSError as exc:
        logging.warning("Could not remove existing video file before writing %s: %s", clip_path, exc)

    normalized_size = normalize_writer_size(size)
    if normalized_size != size:
        logging.info("Adjusted video writer size from %s to %s", size, normalized_size)

    candidates = (
        (clip_path, ("mp4v", "avc1", "H264")),
        (clip_path.with_suffix(".avi"), ("MJPG", "XVID")),
    )
    api_preferences = (cv2.CAP_FFMPEG, None)

    for candidate_path, codecs in candidates:
        for codec in codecs:
            for api_preference in api_preferences:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                if api_preference is None:
                    writer = cv2.VideoWriter(str(candidate_path), fourcc, fps, normalized_size)
                    backend_name = "default"
                else:
                    writer = cv2.VideoWriter(str(candidate_path), api_preference, fourcc, fps, normalized_size)
                    backend_name = "ffmpeg"

                if writer.isOpened():
                    if candidate_path != clip_path:
                        logging.warning("MP4 writer unavailable; using fallback video file: %s", candidate_path)
                    logging.info("Using video codec: %s backend=%s path=%s", codec, backend_name, candidate_path)
                    return ManagedVideoWriter(writer, candidate_path, normalized_size)

                writer.release()

    raise RuntimeError(f"Could not create video writer for {clip_path}")


def normalize_writer_size(size):
    width, height = size
    width = max(2, int(width) - (int(width) % 2))
    height = max(2, int(height) - (int(height) % 2))
    return width, height


def write_recording_frames_until(
    writer,
    raw_writer,
    annotated_frame,
    raw_frame,
    fps,
    clip_end_time,
    next_clip_frame_time,
    now=None,
):
    now = now or time.monotonic()
    frame_interval = 1 / max(fps, 1)
    last_frame_time = min(now, clip_end_time)
    while next_clip_frame_time <= last_frame_time:
        writer.write(annotated_frame)
        if raw_writer is not None:
            raw_writer.write(raw_frame)
        next_clip_frame_time += frame_interval
    return next_clip_frame_time


def emit_recording_progress(analysis_callback, clip_path, clip_start_time, now, last_recording_progress_second):
    if clip_path is None or not clip_start_time:
        return last_recording_progress_second

    elapsed_seconds = max(0, int(now - clip_start_time))
    if elapsed_seconds == last_recording_progress_second:
        return last_recording_progress_second

    displayed_seconds = elapsed_seconds
    progress = min(99, int((min(CLIP_SECONDS, now - clip_start_time) / CLIP_SECONDS) * 100))
    analysis_callback(
        Path(clip_path).name,
        f"Recording clip: {displayed_seconds}s",
        progress,
    )
    return elapsed_seconds


def upload_saved_clip(
    clip_path,
    source_name,
    detected_objects,
    clip_type,
    scenario_name,
    trigger_id=None,
    analysis_callback=noop_analysis_callback,
    upload_on_motion=False,
):
    clip_path = Path(clip_path)
    log_trigger_step(trigger_id, "STEP_7_GEMMA_ANALYSIS_STARTED", "clip=%s", clip_path)
    analysis_callback(clip_path.name, "Gemma scene analysis", -1)
    gemma_analysis = analyze_clip_with_gemma(clip_path, trigger_id=trigger_id)
    if gemma_analysis:
        log_trigger_step(
            trigger_id,
            "STEP_8_GEMMA_ANALYSIS_COMPLETED",
            "scenarios=%r break_in_likely=%r confidence=%r",
            gemma_analysis.get("identified_scenarios"),
            gemma_analysis.get("break_in_likely"),
            gemma_analysis.get("confidence"),
        )
    else:
        log_trigger_step(trigger_id, "STEP_8_GEMMA_ANALYSIS_SKIPPED", "clip=%s", clip_path)

    log_trigger_step(trigger_id, "STEP_9_TRANSCODE_STARTED", "clip=%s", clip_path)
    analysis_callback(clip_path.name, "Transcoding clip for upload", -1)
    compatible_clip_path = make_whatsapp_compatible_mp4(clip_path)
    log_trigger_step(trigger_id, "STEP_10_TRANSCODE_COMPLETED", "clip=%s", compatible_clip_path)
    metadata = {
        "camera": source_name,
        "detected": sorted(detected_objects),
        "source_name": source_name,
        "detected_objects": sorted(detected_objects),
        "clip_type": clip_type,
        "upload_on_motion": upload_on_motion,
        "scenario": scenario_name,
        "trigger_id": trigger_id,
    }
    if gemma_analysis:
        metadata.update(
            {
                "gemma_scene_description": gemma_analysis.get("scene_description"),
                "gemma_identified_scenarios": gemma_analysis.get("identified_scenarios"),
                "gemma_scenario_reasoning": gemma_analysis.get("scenario_reasoning"),
                "gemma_break_in_likely": gemma_analysis.get("break_in_likely"),
                "gemma_confidence": gemma_analysis.get("confidence"),
                "gemma_reasoning": gemma_analysis.get("reasoning"),
                "gemma_finish_reason": gemma_analysis.get("finish_reason"),
            }
        )
    upload_clip_async(str(compatible_clip_path), metadata)
    log_trigger_step(trigger_id, "STEP_11_UPLOAD_QUEUED", "clip=%s", compatible_clip_path)
    analysis_callback(clip_path.name, "Upload queued", 100)


def upload_non_scenario_clip(
    clip_path,
    source_name,
    detected_objects,
    clip_type,
    reason,
    trigger_id=None,
    analysis_callback=noop_analysis_callback,
    upload_on_motion=False,
):
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
        analysis_callback=analysis_callback,
        upload_on_motion=upload_on_motion,
    )


def annotated_copy_path(recorded_clip_path):
    recorded_clip_path = Path(recorded_clip_path)
    return recorded_clip_path.with_name(f"annotated_{recorded_clip_path.name}")


def copy_video_for_annotation(recorded_clip_path, trigger_id=None):
    recorded_clip_path = Path(recorded_clip_path)
    destination_path = annotated_copy_path(recorded_clip_path)
    try:
        shutil.copy2(recorded_clip_path, destination_path)
        log_trigger_step(
            trigger_id,
            "STEP_6_ANNOTATION_COPY_CREATED",
            "source=%s copy=%s",
            recorded_clip_path,
            destination_path,
        )
        return destination_path
    except OSError as exc:
        log_trigger_step(
            trigger_id,
            "STEP_6_ANNOTATION_COPY_FAILED",
            "source=%s copy=%s error=%s",
            recorded_clip_path,
            destination_path,
            exc,
        )
        return None


def analyze_saved_clip_with_yolo(
    clip_path,
    source_name,
    model,
    annotated_output_path=None,
    trigger_id=None,
    status_callback=noop_status_callback,
    analysis_callback=noop_analysis_callback,
):
    clip_path = Path(clip_path)
    annotated_output_path = Path(annotated_output_path) if annotated_output_path is not None else None
    log_trigger_step(trigger_id, "STEP_6_POST_YOLO_ANALYSIS_STARTED", "clip=%s", clip_path)
    status_callback(f"Analyzing saved clip with YOLO: {clip_path.name}")
    analysis_callback(clip_path.name, "Running YOLO over saved clip", 0)
    try:
        cap, clip_path = open_video_file_capture(clip_path, "saved clip")
    except RuntimeError as exc:
        logging.warning("%s", exc)
        log_trigger_step(trigger_id, "STEP_6_POST_YOLO_ANALYSIS_FAILED", "clip=%s reason=could_not_open", clip_path)
        status_callback(f"Could not analyze saved clip: {clip_path.name}")
        analysis_callback(clip_path.name, "Could not open saved clip", 100)
        return ClipYoloAnalysis(None, set(), None)

    detector = ScenarioDetector(window_seconds=CLIP_SECONDS, cooldown_seconds=0)
    class_ids = target_class_ids(model, SCENARIO_OBJECTS)
    detected_objects = set()
    scenario_match = None
    frame_index = 0
    processed_frames = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    last_progress = -1
    fps = normalized_recording_fps(cap.get(cv2.CAP_PROP_FPS))
    annotation_writer = None
    try:
        clip_end_at = datetime.fromtimestamp(clip_path.stat().st_mtime)
    except OSError:
        clip_end_at = datetime.now()
    clip_start_at = clip_end_at - timedelta(seconds=CLIP_SECONDS)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = run_yolo_detection(model, frame, class_ids=class_ids)
            if annotated_output_path is not None and annotation_writer is None:
                height, width = frame.shape[:2]
                if annotated_output_path == clip_path:
                    annotated_output_path = clip_path.with_name(f"processed_{clip_path.name}")
                annotation_writer = create_video_writer(annotated_output_path, fps, (width, height))
                annotated_output_path = annotation_writer.path
                log_trigger_step(
                    trigger_id,
                    "STEP_6_POST_YOLO_ANNOTATED_WRITER_STARTED",
                    "clip=%s",
                    annotated_output_path,
                )

            if annotation_writer is not None:
                annotation_writer.write(results[0].plot())

            frame_objects = detected_object_names_from_result(model, results[0])
            if frame_objects:
                detected_objects.update(scenario_relevant_objects(frame_objects))
                detected_at = clip_start_at + timedelta(seconds=frame_index / max(fps, 1))
                current_match = detector.record_detection(source_name, frame_objects, detected_at=detected_at)
                if scenario_match is None and current_match is not None:
                    scenario_match = current_match
                    save_scenario_frame(
                        results[0].plot(),
                        trigger_id,
                        source_name,
                        scenario_match.name,
                        frame,
                    )

            processed_frames += 1
            frame_index += 1
            if total_frames > 0:
                progress = min(99, int(processed_frames * 100 / total_frames))
                if progress != last_progress:
                    analysis_callback(clip_path.name, f"Analyzing frame {processed_frames}/{total_frames}", progress)
                    last_progress = progress
    finally:
        cap.release()
        if annotation_writer is not None:
            annotation_writer.release()
            log_trigger_step(
                trigger_id,
                "STEP_6_POST_YOLO_ANNOTATED_CLIP_SAVED",
                "clip=%s",
                annotated_output_path,
            )

    if scenario_match:
        log_trigger_step(
            trigger_id,
            "STEP_6_POST_YOLO_SCENARIO_MATCHED",
            "clip=%s scenario=%s objects=%s frames=%s",
            clip_path,
            scenario_match.name,
            sorted(scenario_match.objects),
            processed_frames,
        )
        status_callback(f"Scenario found in saved clip: {clip_path.name} ({scenario_match.name})")
        analysis_callback(clip_path.name, f"Scenario found: {scenario_match.name}", 100)
        return ClipYoloAnalysis(scenario_match, set(scenario_match.objects), annotated_output_path)

    log_trigger_step(
        trigger_id,
        "STEP_6_POST_YOLO_NO_SCENARIO",
        "clip=%s objects=%s frames=%s",
        clip_path,
        sorted(detected_objects),
        processed_frames,
    )
    status_callback(f"No scenario found in saved clip: {clip_path.name}")
    analysis_callback(clip_path.name, "No scenario found", 100)
    return ClipYoloAnalysis(None, detected_objects, annotated_output_path)


def finalize_recorded_clip(
    clip_path,
    source_name,
    model,
    trigger_id=None,
    reason=None,
    live_scenario_match=None,
    live_detected_objects=None,
    status_callback=noop_status_callback,
    analysis_callback=noop_analysis_callback,
    upload_on_motion=False,
):
    if clip_path is None:
        return

    log_trigger_step(
        trigger_id,
        "MOTION_CLIP_SAVED" if upload_on_motion else "STEP_6_CLIP_SAVED",
        "clip=%s%s",
        clip_path,
        f" reason={reason}" if reason else "",
    )
    recorded_clip_path = Path(clip_path)
    if upload_on_motion:
        finish_metadata(recorded_clip_path)
        # Motion-only mode keeps raw recordings local for independent review.
        status_callback(f"Motion clip saved: {recorded_clip_path.name}")
        analysis_callback(recorded_clip_path.name, "Motion clip saved locally", 100)
        return

    if DETECT_DURING_RECORDING:
        detected_objects = set(live_detected_objects or [])
        log_trigger_step(
            trigger_id,
            "STEP_6_POST_YOLO_SKIPPED",
            "clip=%s reason=detect_during_recording_enabled",
            recorded_clip_path,
        )
        status_callback(f"Clip saved, using live recording analysis: {recorded_clip_path.name}")
        if live_scenario_match is not None:
            analysis_callback(recorded_clip_path.name, f"Scenario found: {live_scenario_match.name}", 100)
            upload_saved_clip(
                recorded_clip_path,
                source_name,
                detected_objects or live_scenario_match.objects,
                "scenario_camera_detection",
                live_scenario_match.name,
                trigger_id=trigger_id,
                analysis_callback=analysis_callback,
                upload_on_motion=upload_on_motion,
            )
        else:
            analysis_callback(recorded_clip_path.name, "No scenario found during recording", 100)
            upload_non_scenario_clip(
                recorded_clip_path,
                source_name,
                detected_objects,
                "camera_detection",
                "live recording YOLO found no scenario",
                trigger_id=trigger_id,
                analysis_callback=analysis_callback,
                upload_on_motion=upload_on_motion,
            )
        return

    annotated_clip_path = copy_video_for_annotation(recorded_clip_path, trigger_id=trigger_id)
    if annotated_clip_path is None:
        status_callback(f"Could not create annotated copy: {recorded_clip_path.name}")
        analysis_callback(recorded_clip_path.name, "Could not create annotated copy", 100)
        return
    annotated_output_path = annotated_clip_path

    status_callback(f"Clip saved, starting analysis: {annotated_clip_path.name}")
    status_callback(f"Analyzing saved clip: {annotated_clip_path.name}")
    analysis_callback(annotated_clip_path.name, "Queued for saved-clip analysis", 0)
    yolo_analysis = analyze_saved_clip_with_yolo(
        annotated_clip_path,
        source_name,
        model,
        annotated_output_path=annotated_output_path,
        trigger_id=trigger_id,
        status_callback=status_callback,
        analysis_callback=analysis_callback,
    )
    final_annotated_clip_path = annotated_clip_path
    if yolo_analysis.annotated_clip_path is not None and yolo_analysis.annotated_clip_path != annotated_clip_path:
        if yolo_analysis.annotated_clip_path.suffix == annotated_clip_path.suffix:
            try:
                os.replace(yolo_analysis.annotated_clip_path, annotated_clip_path)
                log_trigger_step(
                    trigger_id,
                    "STEP_6_POST_YOLO_ANNOTATED_CLIP_REPLACED",
                    "clip=%s",
                    annotated_clip_path,
                )
            except OSError as exc:
                final_annotated_clip_path = yolo_analysis.annotated_clip_path
                log_trigger_step(
                    trigger_id,
                    "STEP_6_POST_YOLO_ANNOTATED_CLIP_REPLACE_FAILED",
                    "source=%s destination=%s error=%s",
                    yolo_analysis.annotated_clip_path,
                    annotated_clip_path,
                    exc,
                )
        else:
            final_annotated_clip_path = yolo_analysis.annotated_clip_path
            log_trigger_step(
                trigger_id,
                "STEP_6_POST_YOLO_ANNOTATED_CLIP_USING_FALLBACK",
                "clip=%s",
                final_annotated_clip_path,
            )
    if yolo_analysis.scenario_match is not None:
        upload_saved_clip(
            final_annotated_clip_path,
            source_name,
            yolo_analysis.detected_objects,
            "scenario_camera_detection",
            yolo_analysis.scenario_match.name,
            trigger_id=trigger_id,
            analysis_callback=analysis_callback,
            upload_on_motion=upload_on_motion,
        )
    else:
        upload_non_scenario_clip(
            final_annotated_clip_path,
            source_name,
            yolo_analysis.detected_objects,
            "camera_detection",
            "post-recording YOLO found no scenario",
            trigger_id=trigger_id,
            analysis_callback=analysis_callback,
            upload_on_motion=upload_on_motion,
        )


def target_class_ids(model, target_objects):
    names = model.names.items() if hasattr(model.names, "items") else enumerate(model.names)
    return [
        class_id
        for class_id, class_name in names
        if normalize_detected_class_name(class_name) in target_objects
    ]


def normalize_detected_class_name(class_name):
    class_name = str(class_name).strip().lower()
    return YOLO_CLASS_ALIASES.get(class_name, class_name)


def detected_object_names_from_result(model, result):
    return [
        normalize_detected_class_name(model.names[int(box.cls[0])])
        for box in result.boxes
    ]


def run_yolo_detection(model, frame, class_ids=None):
    kwargs = {
        "verbose": False,
        "conf": YOLO_CONFIDENCE,
        "imgsz": YOLO_IMAGE_SIZE,
    }
    if class_ids is not None:
        kwargs["classes"] = class_ids
    return model(frame, **kwargs)


def emit_preview_frame(frame, frame_callback):
    if frame_callback is None:
        return
    should_emit = getattr(frame_callback, "should_emit", None)
    if should_emit is not None and not should_emit():
        return

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_PREVIEW_QUALITY]
    encoded, buffer = cv2.imencode(".jpg", frame, encode_params)
    if encoded:
        frame_callback(buffer.tobytes())


def detect_motion(frame, motion_detector, sensitivity_config=None, exclusions=None):
    sensitivity_config = sensitivity_config or motion_sensitivity_config()
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height
    motion_mask = motion_detector.apply(frame)
    motion_mask = cv2.threshold(motion_mask, 244, 255, cv2.THRESH_BINARY)[1]
    motion_mask = cv2.dilate(motion_mask, None, iterations=2)
    for x, y, w, h in exclusions or ():
        x1, y1 = int(x * frame_width), int(y * frame_height)
        x2, y2 = int((x+w) * frame_width + .999999), int((y+h) * frame_height + .999999)
        motion_mask[y1:y2, x1:x2] = 0

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


def draw_detection_status(frame, motion_detected, matched_objects):
    status_lines = [
        f"Motion: {'YES' if motion_detected else 'NO'}",
        f"Objects: {', '.join(sorted(matched_objects)) if matched_objects else 'none'}",
    ]
    for index, text in enumerate(status_lines):
        y = 30 + (index * 28)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def safe_file_stem(name):
    stem = str(name or "").strip()
    stem = re.sub(r"\s*\([^)]*(?:/|rtsp://)[^)]*\)\s*", " ", stem)
    stem = re.sub(r"\s+-\s+(?:/|rtsp://).*$", "", stem)
    stem = re.sub(r"(?:/|rtsp://).*$", "", stem)
    stem = "".join(
        character if character.isalnum() else "_"
        for character in stem
    ).strip("_")
    stem = re.sub(r"_+", "_", stem)
    return stem or "camera"


def save_trigger_debug_frame(annotated_frame, raw_frame, trigger_id, source_name, objects):
    DEBUG_FRAME_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_source_name = safe_file_stem(source_name)
    safe_objects = "_".join(sorted(objects)) or "objects"
    annotated_frame_path = DEBUG_FRAME_DIR / f"{trigger_id}_{safe_source_name}_{safe_objects}_{timestamp}.jpg"
    raw_frame_path = DEBUG_FRAME_DIR / f"raw_{trigger_id}_{safe_source_name}_{safe_objects}_{timestamp}.jpg"

    if cv2.imwrite(str(annotated_frame_path), annotated_frame):
        log_trigger_step(trigger_id, "STEP_3_DEBUG_FRAME_SAVED", "image=%s", annotated_frame_path)
        upload_frame_async(
            annotated_frame_path,
            DEBUG_FRAMES_CONTAINER,
            "debug_frame",
            trigger_id,
            source_name,
            objects=objects,
            frame_variant="annotated",
        )
    else:
        log_trigger_step(trigger_id, "STEP_3_DEBUG_FRAME_SAVE_FAILED", "image=%s", annotated_frame_path)

    if cv2.imwrite(str(raw_frame_path), raw_frame):
        log_trigger_step(trigger_id, "STEP_3_RAW_DEBUG_FRAME_SAVED", "image=%s", raw_frame_path)
        upload_frame_async(
            raw_frame_path,
            DEBUG_FRAMES_CONTAINER,
            "debug_frame",
            trigger_id,
            source_name,
            objects=objects,
            frame_variant="raw",
        )
    else:
        log_trigger_step(trigger_id, "STEP_3_RAW_DEBUG_FRAME_SAVE_FAILED", "image=%s", raw_frame_path)


def upload_frame_async(
    frame_path,
    container,
    frame_type,
    trigger_id,
    source_name,
    objects=None,
    scenario_name=None,
    frame_variant=None,
):
    metadata = {
        "camera": source_name,
        "source_name": source_name,
        "frame_type": frame_type,
        "trigger_id": trigger_id,
    }
    if objects:
        metadata["detected_objects"] = sorted(objects)
        metadata["detected"] = sorted(objects)
    if scenario_name:
        metadata["scenario"] = scenario_name
    if frame_variant:
        metadata["frame_variant"] = frame_variant

    upload_file_async(str(frame_path), metadata, container=container)


def save_scenario_frame(frame, trigger_id, source_name, scenario_name, raw_frame=None):
    SCENARIO_FRAME_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_source_name = safe_file_stem(source_name)
    safe_scenario_name = "".join(
        character if character.isalnum() else "_"
        for character in scenario_name
    ).strip("_")
    scenario_frame_path = SCENARIO_FRAME_DIR / f"{trigger_id}_{safe_source_name}_{safe_scenario_name}_{timestamp}.jpg"
    raw_scenario_frame_path = SCENARIO_FRAME_DIR / f"raw_{trigger_id}_{safe_source_name}_{safe_scenario_name}_{timestamp}.jpg"
    if cv2.imwrite(str(scenario_frame_path), frame):
        log_trigger_step(trigger_id, "STEP_5_SCENARIO_FRAME_SAVED", "image=%s", scenario_frame_path)
        upload_frame_async(
            scenario_frame_path,
            SCENARIO_FRAMES_CONTAINER,
            "scenario_frame",
            trigger_id,
            source_name,
            scenario_name=scenario_name,
            frame_variant="annotated",
        )
    else:
        log_trigger_step(trigger_id, "STEP_5_SCENARIO_FRAME_SAVE_FAILED", "image=%s", scenario_frame_path)

    if raw_frame is None:
        return

    if cv2.imwrite(str(raw_scenario_frame_path), raw_frame):
        log_trigger_step(trigger_id, "STEP_5_RAW_SCENARIO_FRAME_SAVED", "image=%s", raw_scenario_frame_path)
        upload_frame_async(
            raw_scenario_frame_path,
            SCENARIO_FRAMES_CONTAINER,
            "scenario_frame",
            trigger_id,
            source_name,
            scenario_name=scenario_name,
            frame_variant="raw",
        )
    else:
        log_trigger_step(trigger_id, "STEP_5_RAW_SCENARIO_FRAME_SAVE_FAILED", "image=%s", raw_scenario_frame_path)


def run_detection(
    rtsp_url,
    window_title,
    target_objects,
    stop_requested,
    frame_callback=noop_frame_callback,
    motion_sensitivity=None,
    status_callback=noop_status_callback,
    analysis_callback=noop_analysis_callback,
    yolo_model_filename=None,
    upload_on_motion=None,
    excluded_areas=None,
    motion_algorithm=None,
    exclusions_provider=None,
    object_detector=None,
    object_status_callback=None,
    object_source_id=None,
    object_result_callback=None,
    heartbeat_callback=None,
    replay=None,
    replay_progress_callback=None,
    replay_check_callback=None,
):
    # This branch deliberately runs motion capture only, regardless of legacy settings.
    upload_on_motion = True
    sensitivity_config = motion_sensitivity_config(motion_sensitivity)
    exclusions = FrameDifferenceMotion.validate_exclusions(excluded_areas or [])
    algorithm = motion_algorithm or MOTION_ALGORITHM
    if algorithm not in ('background', 'difference'):
        raise ValueError('unknown motion algorithm')
    difference = FrameDifferenceMotion(pixel_threshold=20 if sensitivity_config.level == 1 else 15,
        min_area=max(1, int(200 * 5 / sensitivity_config.level)), exclusions=excluded_areas) if algorithm == 'difference' else None
    model = None
    motion_detector = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=True,
    )

    motion_output_dir = Path(replay['motion_dir']) if replay else OUTPUT_DIR
    object_output_dir = Path(replay['object_dir']) if replay else OBJECT_OUTPUT_DIR
    motion_output_dir.mkdir(parents=True, exist_ok=True)
    replay_frame_number = 0
    replay_started = None
    replay_fps = None
    replay_duration = None
    replay_last_frame = None
    replay_last_progress = -1

    cap = None
    fps = 20
    writer = None
    active_clip_path = None
    active_clip_objects = set()
    active_clip_scenario_detector = None
    active_clip_scenario_match = None
    active_trigger_id = None
    pending_trigger_id = None
    clip_start_time = 0.0
    clip_detection_start = 0.0
    clip_end_time = 0.0
    next_clip_frame_time = 0.0
    next_detection_time = 0.0
    last_recording_progress_second = -1
    last_recording_frame = None
    last_recording_raw_frame = None
    last_overlay_motion_boxes = []
    last_overlay_motion_detected = False
    last_overlay_matched_objects = set()
    frame_count = 0
    consecutive_read_failures = 0
    motion_ready_at = float("inf")
    preroll = MotionPreRoll(seconds=MOTION_PREROLL_SECONDS)
    object_recorder = None
    pending_alerts = []
    if object_detector is not None:
        from object_detection import ObjectClipRecorder
        object_recorder = ObjectClipRecorder(object_detector, object_source_id or window_title, object_output_dir,
            create_video_writer, object_status_callback or (lambda _recording: None), preroll,
            MOTION_MIN_FREE_MB * 1024 * 1024, alert_camera=window_title, send_alerts=not bool(replay))
    shadow = MotionShadowComparison(window_title) if MOTION_SHADOW_ENABLED else None
    storage_warning_at = 0.0
    next_recording_space_check = 0.0
    logging.info(
        "Motion sensitivity for %s: level=%s min_motion_area=%s detect_during_recording=%s detection_interval_seconds=%s",
        window_title,
        sensitivity_config.level,
        sensitivity_config.min_motion_area,
        DETECT_DURING_RECORDING,
        DETECTION_INTERVAL_SECONDS,
    )

    try:
        while not stop_requested():
            if cap is None:
                if replay:
                    cap, _ = open_video_file_capture(replay['path'], 'test clip')
                    replay_fps = float(cap.get(cv2.CAP_PROP_FPS))
                    if not 0 < replay_fps <= 240:
                        raise ValueError('Test clip has an invalid frame rate')
                    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    replay_duration = count / replay_fps if count > 0 else None
                    replay_started = time.monotonic()
                else:
                    cap = open_rtsp_capture(rtsp_url)
                if cap is None:
                    logging.warning("Could not open RTSP stream %s. Retrying...", window_title)
                    wait_before_reconnect(stop_requested)
                    continue

                # A new stream needs a fresh background, including after reconnect.
                motion_detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
                preroll.clear()
                if object_recorder is not None:
                    object_recorder.motion_association.clear()
                    object_recorder.moving_object_gate.clear()
                    object_recorder.weak_motion_vehicle.clear()
                    object_recorder.detector.remove(object_recorder.camera)
                if difference is not None:
                    difference.reset()
                if shadow is not None:
                    shadow.reset()
                motion_ready_at = time.monotonic() + MOTION_WARMUP_SECONDS
                next_detection_time = 0.0
                reported_fps = cap.get(cv2.CAP_PROP_FPS)
                fps = normalized_recording_fps(reported_fps)
                consecutive_read_failures = 0
                logging.info(
                    "Connected RTSP stream %s reported_fps=%s recording_fps=%s open_timeout_ms=%s read_timeout_ms=%s read_failures_before_reconnect=%s",
                    window_title,
                    reported_fps,
                    fps,
                    STREAM_OPEN_TIMEOUT_MS,
                    STREAM_READ_TIMEOUT_MS,
                    READ_FAILURES_BEFORE_RECONNECT,
                )

            if replay:
                target = replay_started + replay_frame_number / replay_fps
                while not stop_requested() and time.monotonic() < target:
                    time.sleep(min(.05, max(0, target-time.monotonic())))
                if stop_requested():
                    break
            ret, frame = cap.read()

            if not ret and replay:
                if replay_last_frame is not None and object_recorder is not None:
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline and object_detector.has_pending(object_recorder.camera):
                        checks = object_recorder.observe(replay_last_frame, time.monotonic(), False)
                        if checks and replay_check_callback:
                            replay_check_callback(checks)
                        time.sleep(.05)
                    checks = object_recorder.observe(replay_last_frame, time.monotonic(), False)
                    if checks and replay_check_callback:
                        replay_check_callback(checks)
                    if object_detector.has_pending(object_recorder.camera):
                        raise RuntimeError('Object detection did not finish the last sample')
                break
            if not ret:
                consecutive_read_failures += 1
                if object_recorder is not None and consecutive_read_failures == READ_FAILURES_BEFORE_RECONNECT:
                    object_recorder.reset()
                if writer is not None and last_recording_frame is not None and last_recording_raw_frame is not None:
                    now = time.monotonic()
                    next_clip_frame_time = write_recording_frames_until(
                        writer,
                        None,
                        last_recording_frame,
                        last_recording_raw_frame,
                        fps,
                        clip_end_time,
                        next_clip_frame_time,
                        now=now,
                    )
                    last_recording_progress_second = emit_recording_progress(
                        analysis_callback,
                        active_clip_path,
                        clip_start_time,
                        now,
                        last_recording_progress_second,
                    )
                    if now < clip_end_time:
                        if consecutive_read_failures == READ_FAILURES_BEFORE_RECONNECT:
                            logging.warning(
                                "RTSP read failures during active recording for %s; keeping clip open and retrying until the recording window ends.",
                                window_title,
                            )
                        if RECORDING_READ_RETRY_SLEEP_SECONDS:
                            time.sleep(RECORDING_READ_RETRY_SLEEP_SECONDS)
                        continue

                    writer.release()
                    if active_clip_path is not None:
                        finalize_recorded_clip(
                            active_clip_path,
                            window_title,
                            model,
                            trigger_id=active_trigger_id,
                            reason="read_failures_after_recording_window",
                            live_scenario_match=active_clip_scenario_match,
                            live_detected_objects=active_clip_objects,
                            upload_on_motion=upload_on_motion,
                            status_callback=status_callback,
                            analysis_callback=analysis_callback,
                        )
                    writer = None
                    active_clip_path = None
                    active_clip_objects = set()
                    active_clip_scenario_detector = None
                    active_clip_scenario_match = None
                    active_trigger_id = None
                    pending_trigger_id = None
                    clip_start_time = 0.0
                    clip_end_time = 0.0
                    next_clip_frame_time = 0.0
                    last_recording_progress_second = -1
                    last_recording_frame = None
                    last_recording_raw_frame = None
                    cap.release()
                    cap = None
                    wait_before_reconnect(stop_requested)
                    continue

                if consecutive_read_failures < READ_FAILURES_BEFORE_RECONNECT:
                    continue

                logging.warning(
                    "Lost RTSP stream %s after %s consecutive read failures. Reconnecting...",
                    window_title,
                    consecutive_read_failures,
                )
                if writer is not None:
                    writer.release()
                    if active_clip_path is not None:
                        finalize_recorded_clip(
                            active_clip_path,
                            window_title,
                            model,
                            trigger_id=active_trigger_id,
                            reason="stream_lost",
                            live_scenario_match=active_clip_scenario_match,
                            live_detected_objects=active_clip_objects,
                            upload_on_motion=upload_on_motion,
                            status_callback=status_callback,
                            analysis_callback=analysis_callback,
                        )
                    writer = None
                    active_clip_path = None
                    active_clip_objects = set()
                    active_clip_scenario_detector = None
                    active_clip_scenario_match = None
                    active_trigger_id = None
                    pending_trigger_id = None
                    clip_start_time = 0.0
                    clip_end_time = 0.0
                    next_clip_frame_time = 0.0
                    last_recording_progress_second = -1
                    last_recording_frame = None
                    last_recording_raw_frame = None

                cap.release()
                cap = None
                wait_before_reconnect(stop_requested)
                continue

            consecutive_read_failures = 0
            replay_capture_at = None
            if replay:
                from datetime import timedelta
                position = replay_frame_number / replay_fps
                replay_capture_at = (replay['recorded_at'] + timedelta(seconds=position)).isoformat()
                replay_frame_number += 1
                replay_last_frame = frame
                if replay_progress_callback and int(position) != replay_last_progress:
                    replay_last_progress = int(position)
                    replay_progress_callback(position, replay_duration)

            loop_now = time.monotonic()
            if heartbeat_callback:
                heartbeat_callback(loop_now)
            preroll.add(loop_now, frame)
            # Keep the background model and live motion display current during recording.
            skip_detection_for_recording = False
            detection_due = DETECTION_INTERVAL_SECONDS <= 0 or loop_now >= next_detection_time
            motion_boxes = []
            motion_pair = None
            motion_detected = False
            matched_objects = set()
            relevant_objects = set()
            detected_object_names = []
            annotated_frame = frame.copy()

            if not skip_detection_for_recording and detection_due:
                next_detection_time = loop_now + DETECTION_INTERVAL_SECONDS
                if exclusions_provider is not None:
                    # The web app validates and atomically publishes each camera's areas.
                    exclusions = exclusions_provider()
                    if difference is not None:
                        difference.exclusions = exclusions
                raw_boxes = detect_motion(frame, motion_detector, sensitivity_config, exclusions) if exclusions else detect_motion(frame, motion_detector, sensitivity_config)
                motion_boxes = raw_boxes
                if difference is not None:
                    try:
                        primed = (difference.previous_time is not None and 0 < loop_now - difference.previous_time <= difference.max_gap_seconds)
                        previous_motion_frame, previous_motion_time = difference.previous, difference.previous_time
                        candidate_boxes = difference.observe(frame, loop_now)
                        if primed:
                            motion_boxes = candidate_boxes
                            motion_pair = (previous_motion_frame, difference.previous, loop_now - previous_motion_time)
                    except Exception:
                        logging.exception('Frame difference failed for %s; falling back to background detection', window_title)
                        difference = None
                motion_detected = bool(motion_boxes) and loop_now >= motion_ready_at
                if replay and replay_check_callback and motion_detected:
                    replay_check_callback([{'motion_regions': [list(box) for box in motion_boxes], 'captured_at': replay_capture_at}])
                if object_recorder is not None and loop_now >= motion_ready_at:
                    object_recorder.motion_association.add(loop_now, motion_boxes, frame.shape[1], frame.shape[0])
                if writer is not None and motion_detected:
                    clip_end_time = max(clip_end_time, loop_now + MOTION_QUIET_SECONDS)
                if shadow is not None:
                    shadow.observe(frame, loop_now, bool(raw_boxes), eligible=loop_now >= motion_ready_at)

                if motion_detected:
                    if writer is None and pending_trigger_id is None:
                        pending_trigger_id = new_trigger_id()
                        log_trigger_step(
                            pending_trigger_id,
                            "MOTION_DETECTED",
                            "camera=%s motion_boxes=%s",
                            window_title,
                            len(motion_boxes),
                        )

                    if not upload_on_motion:
                        results = run_yolo_detection(model, frame)

                        annotated_frame = results[0].plot()
                        draw_motion_boxes(annotated_frame, motion_boxes)
                        detected_object_names = detected_object_names_from_result(model, results[0])
                        detected_objects = set(detected_object_names)
                        matched_objects = detected_objects & target_objects
                        relevant_objects = scenario_relevant_objects(detected_object_names)
                        last_overlay_matched_objects = set(matched_objects)
                        if writer is not None and DETECT_DURING_RECORDING and active_clip_scenario_detector is not None:
                            current_scenario_match = active_clip_scenario_detector.record_detection(
                                window_title,
                                detected_object_names,
                            )
                            if active_clip_scenario_match is None and current_scenario_match is not None:
                                active_clip_scenario_match = current_scenario_match
                                log_trigger_step(
                                    active_trigger_id,
                                    "STEP_5_SCENARIO_MATCHED_DURING_RECORDING",
                                    "scenario=%s clip=%s",
                                    active_clip_scenario_match.name,
                                    active_clip_path,
                                )
                                save_scenario_frame(
                                    annotated_frame,
                                    active_trigger_id,
                                    window_title,
                                    active_clip_scenario_match.name,
                                    frame,
                                )
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
                else:
                    last_overlay_matched_objects = set()

                last_overlay_motion_boxes = list(motion_boxes)
                last_overlay_motion_detected = motion_detected

            if not skip_detection_for_recording:
                draw_motion_boxes(annotated_frame, last_overlay_motion_boxes)
                draw_detection_status(
                    annotated_frame,
                    last_overlay_motion_detected,
                    last_overlay_matched_objects,
                )

            # Never delete user footage to make room. Stop creating new clips at the
            # configured cap or reserve, while leaving preview and detection running.
            recording_blocked = False
            if writer is None and motion_detected:
                clip_count = sum(1 for path in motion_output_dir.glob("motion_*")
                                 if path.is_file() and path.suffix.lower() in ('.mp4', '.mov', '.avi', '.mkv', '.m4v'))
                enough_space = shutil.disk_usage(motion_output_dir).free >= MOTION_MIN_FREE_MB * 1024 * 1024
                if clip_count >= MOTION_MAX_CLIPS or not enough_space:
                    if loop_now >= storage_warning_at:
                        status_callback("Motion recording paused: clip limit or disk reserve reached")
                        logging.warning("Motion recording paused: count=%s limit=%s enough_space=%s", clip_count, MOTION_MAX_CLIPS, enough_space)
                        storage_warning_at = loop_now + 60
                    recording_blocked = True
                    pending_trigger_id = None

            if writer is None and not recording_blocked and (matched_objects or (upload_on_motion and motion_detected)):
                if pending_trigger_id is None:
                    pending_trigger_id = new_trigger_id()
                if not upload_on_motion:
                    log_trigger_step(pending_trigger_id, "STEP_3_TARGET_OBJECTS_DETECTED",
                                     "objects=%s", sorted(matched_objects))
                if not upload_on_motion:
                    save_trigger_debug_frame(
                        annotated_frame,
                        frame,
                        pending_trigger_id,
                        window_title,
                        matched_objects,
                    )
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                safe_window_title = safe_file_stem(window_title)
                clip_prefix = "motion" if upload_on_motion else ("annotated_pending_scenario" if DETECT_DURING_RECORDING else "pending_scenario")
                clip_path = motion_output_dir / f"{clip_prefix}_{safe_window_title}_{timestamp}.mp4"
                height, width = annotated_frame.shape[:2]
                raw_writer = create_video_writer(clip_path, fps, (width, height))
                writer = AsyncPreRollWriter(raw_writer, preroll, loop_now, fps)
                clip_path = writer.path
                active_clip_path = clip_path
                if upload_on_motion:
                    start_metadata(clip_path)
                active_clip_objects = set(relevant_objects)
                active_clip_scenario_detector = None if upload_on_motion else ScenarioDetector(window_seconds=CLIP_SECONDS, cooldown_seconds=0)
                active_clip_scenario_match = None
                if DETECT_DURING_RECORDING and not upload_on_motion:
                    active_clip_scenario_match = active_clip_scenario_detector.record_detection(
                        window_title,
                        detected_object_names,
                    )
                active_trigger_id = pending_trigger_id
                pending_trigger_id = None
                clip_detection_start = loop_now
                clip_start_time = time.monotonic()
                clip_end_time = clip_start_time + CLIP_SECONDS
                next_recording_space_check = clip_start_time + 5.0
                next_clip_frame_time = clip_start_time
                last_recording_progress_second = -1
                last_recording_frame = frame.copy()
                last_recording_raw_frame = frame.copy()
                log_trigger_step(
                    active_trigger_id,
                    "MOTION_CLIP_RECORDING_STARTED" if upload_on_motion else "STEP_4_CLIP_RECORDING_STARTED",
                    "camera=%s clip=%s minimum_seconds=%s detect_during_recording=%s",
                    window_title,
                    clip_path,
                    CLIP_SECONDS,
                    DETECT_DURING_RECORDING,
                )
                if active_clip_scenario_match is not None:
                    log_trigger_step(
                        active_trigger_id,
                        "STEP_5_SCENARIO_MATCHED_DURING_RECORDING",
                        "scenario=%s clip=%s",
                        active_clip_scenario_match.name,
                        active_clip_path,
                    )
                    save_scenario_frame(
                        annotated_frame,
                        active_trigger_id,
                        window_title,
                        active_clip_scenario_match.name,
                        frame,
                    )
                status_callback(f"Recording clip: {clip_path.name}")
                last_recording_progress_second = emit_recording_progress(
                    analysis_callback,
                    active_clip_path,
                    clip_start_time,
                    clip_start_time,
                    last_recording_progress_second,
                )
            elif writer is None and pending_trigger_id and motion_detected:
                log_trigger_step(
                    pending_trigger_id,
                    "STEP_3_NO_TARGET_OBJECTS_DETECTED",
                    "detected_objects=%s target_objects=%s",
                    sorted(detected_object_names),
                    sorted(target_objects),
                )
                pending_trigger_id = None

            if object_recorder is not None:
                try:
                    checks = object_recorder.observe(frame, loop_now, bool(writer is not None or motion_detected), motion_boxes if motion_detected else None, motion_pair if motion_detected else None, captured_at=replay_capture_at)
                    if replay and checks and replay_check_callback:
                        replay_check_callback(checks)
                    if object_result_callback:
                        for _sampled_at, labels, _evidence, _events, _decisions, _source, _associations, _crop_result in checks:
                            if labels:
                                object_result_callback(labels)
                    pending_alerts = [(t, event) for t, event in pending_alerts if loop_now-t <= 15]
                    for sampled_at, labels, evidence, events, decisions, source, associations, crop_result in checks:
                        if active_clip_path is not None and upload_on_motion and sampled_at >= clip_detection_start:
                            diagnostic = [{**item, 'source': source, 'seconds_after_trigger': round(sampled_at-clip_detection_start, 1)} for item in associations]
                            comparison = None if crop_result is None else {**crop_result, 'full_frame_labels': labels, 'seconds_after_trigger': round(sampled_at-clip_detection_start, 1)}
                            update_metadata(active_clip_path, labels, evidence, events, decisions, diagnostic, comparison)
                        else:
                            pending_alerts.extend((sampled_at, event) for event in events)
                    if active_clip_path is not None and upload_on_motion and pending_alerts:
                        update_metadata(active_clip_path, [], alerts=[event for _t, event in pending_alerts],
                                        decisions=['alert carried into next motion clip'])
                        pending_alerts = []
                except Exception:
                    if replay:
                        raise
                    logging.exception('Object clip processing failed for %s', window_title)
                    try:
                        object_recorder.stop()
                    except Exception:
                        logging.exception('Could not close object clip for %s', window_title)
                    if object_status_callback:
                        object_status_callback('error')
                    object_recorder = None

            if writer is not None:
                active_clip_objects.update(relevant_objects)

                if loop_now >= next_recording_space_check:
                    next_recording_space_check = loop_now + 5.0
                    if shutil.disk_usage(motion_output_dir).free < MOTION_MIN_FREE_MB * 1024 * 1024:
                        logging.warning("Stopping motion recording for %s: disk reserve reached", window_title)
                        status_callback("Motion recording paused: disk reserve reached")
                        clip_end_time = min(clip_end_time, loop_now)

            if writer is not None:
                last_recording_frame = frame.copy()
                last_recording_raw_frame = frame.copy()
                now = time.monotonic()
                next_clip_frame_time = write_recording_frames_until(
                    writer,
                    None,
                    frame,
                    frame,
                    fps,
                    clip_end_time,
                    next_clip_frame_time,
                    now=now,
                )
                last_recording_progress_second = emit_recording_progress(
                    analysis_callback,
                    active_clip_path,
                    clip_start_time,
                    now,
                    last_recording_progress_second,
                )

                if now >= clip_end_time:
                    writer.release()
                    if active_clip_path is not None:
                        finalize_recorded_clip(
                            active_clip_path,
                            window_title,
                            model,
                            trigger_id=active_trigger_id,
                            live_scenario_match=active_clip_scenario_match,
                            live_detected_objects=active_clip_objects,
                            upload_on_motion=upload_on_motion,
                            status_callback=status_callback,
                            analysis_callback=analysis_callback,
                        )
                    writer = None
                    active_clip_path = None
                    active_clip_objects = set()
                    active_clip_scenario_detector = None
                    active_clip_scenario_match = None
                    active_trigger_id = None
                    pending_trigger_id = None
                    clip_start_time = 0.0
                    clip_end_time = 0.0
                    next_clip_frame_time = 0.0
                    last_recording_progress_second = -1
                    last_recording_frame = None
                    last_recording_raw_frame = None

            frame_count += 1
            if frame_count % PREVIEW_EVERY_N_FRAMES == 0:
                emit_preview_frame(annotated_frame, frame_callback)
    finally:
        if object_recorder is not None:
            try:
                object_recorder.stop()
            except Exception:
                logging.exception('Could not close object clip for %s', window_title)
        if writer is not None:
            writer.release()
            if active_clip_path is not None:
                finalize_recorded_clip(
                    active_clip_path,
                    window_title,
                    model,
                    trigger_id=active_trigger_id,
                    reason="stream_stopped",
                    live_scenario_match=active_clip_scenario_match,
                    live_detected_objects=active_clip_objects,
                    upload_on_motion=upload_on_motion,
                    status_callback=status_callback,
                    analysis_callback=analysis_callback,
                )

        if cap is not None:
            cap.release()
