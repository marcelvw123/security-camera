import argparse
import getpass
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

from app_config import load_dotenv
from camera_discovery import discover_nvr_streams
from network_guess import likely_dvr_ip
from security_camera_core import (
    DEFAULT_RTSP_PATH_TEMPLATE,
    DEFAULT_RTSP_PORT,
    DEFAULT_TARGET_OBJECTS,
    build_rtsp_url,
    can_open_rtsp_channel,
    run_detection,
    run_video_clip_detection,
)


LOG_FILE = "security_camera_headless.log"
DEFAULT_MOTION_SENSITIVITY = 5
HEADLESS_CONFIG_FILE = Path("headless_config.json")
LEGACY_HEADLESS_STREAM_CONFIG_FILE = Path("headless_streams.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run security camera detection headlessly.")
    parser.add_argument(
        "--test-video-clip",
        metavar="PATH",
        help="Run the same video clip test/annotation flow as the GUI button, then exit.",
    )
    parser.add_argument(
        "--test-video-sensitivity",
        type=int,
        default=DEFAULT_MOTION_SENSITIVITY,
        choices=range(1, 11),
        metavar="1-10",
        help="Motion sensitivity to use with --test-video-clip. Default: 5.",
    )
    return parser.parse_args(argv)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )


def stream_url_for(settings, stream):
    if stream.rtsp_url:
        return stream.rtsp_url

    stream_settings = dict(settings)
    if stream.rtsp_path_template:
        stream_settings["rtsp_path_template"] = stream.rtsp_path_template
    return build_rtsp_url(stream_settings, channel=stream.channel)


def run_stream(stream, settings, stop_event, motion_sensitivity):
    rtsp_url = stream_url_for(settings, stream)
    logging.info("Starting stream: %s with sensitivity %s", stream.label, motion_sensitivity)
    try:
        run_detection(
            rtsp_url,
            stream.label,
            DEFAULT_TARGET_OBJECTS,
            stop_event.is_set,
            None,
            motion_sensitivity,
        )
    except Exception:
        logging.exception("Stream failed: %s", stream.label)
    finally:
        logging.info("Stopped stream: %s", stream.label)


def env_value(*names, default=""):
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def prompt_required(label, default=""):
    prompt = f"{label}"
    if default:
        prompt += f" [{default}]"
    prompt += ": "

    while True:
        value = input(prompt).strip()
        if value:
            return value
        if default:
            return default
        print(f"{label} is required.")


def prompt_password(default=""):
    prompt = "Password"
    if default:
        prompt += " [from .env]"
    prompt += ": "

    while True:
        if sys.stdin.isatty():
            value = getpass.getpass(prompt).strip()
        else:
            value = input(prompt).strip()
        if value:
            return value
        if default:
            return default
        print("Password is required.")


def build_settings():
    load_dotenv()
    default_ip = likely_dvr_ip() or ""
    default_username = env_value("CAMERA_USERNAME", "NVR_USERNAME")
    default_password = env_value("CAMERA_PASSWORD", "NVR_PASSWORD")

    settings = {
        "username": prompt_required("Username", default_username),
        "password": prompt_password(default_password),
        "ip_address": prompt_required("DVR/NVR IP", default_ip),
        "port": env_value("CAMERA_RTSP_PORT", "NVR_RTSP_PORT", default=DEFAULT_RTSP_PORT),
        "rtsp_path_template": env_value(
            "CAMERA_RTSP_PATH_TEMPLATE",
            "NVR_RTSP_PATH_TEMPLATE",
            default=DEFAULT_RTSP_PATH_TEMPLATE,
        ),
    }
    return settings


def load_headless_config():
    for config_path in (HEADLESS_CONFIG_FILE, LEGACY_HEADLESS_STREAM_CONFIG_FILE):
        if not config_path.is_file():
            continue

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("Could not read %s: %s", config_path, exc)
            continue

        logging.info("Loaded headless config from %s", config_path)
        return config, config_path

    return None, None


def settings_from_config(config):
    if not isinstance(config, dict):
        return None

    settings = config.get("settings")
    if not isinstance(settings, dict):
        return None

    normalized = {
        "username": str(settings.get("username", "")).strip(),
        "password": str(settings.get("password", "")),
        "ip_address": str(settings.get("ip_address", "")).strip(),
        "port": str(settings.get("port", DEFAULT_RTSP_PORT)).strip() or DEFAULT_RTSP_PORT,
        "rtsp_path_template": str(settings.get("rtsp_path_template", DEFAULT_RTSP_PATH_TEMPLATE)).strip()
        or DEFAULT_RTSP_PATH_TEMPLATE,
    }
    if not normalized["username"] or not normalized["password"] or not normalized["ip_address"]:
        logging.warning("Saved headless config is missing username, password, or DVR/NVR IP.")
        return None

    return normalized


def stream_choice_text(streams):
    lines = ["Discovered streams:"]
    for index, stream in enumerate(streams, start=1):
        lines.append(f"  {index}. {stream.label}")
    lines.append("Enter stream numbers to run, separated by commas. Example: 1,3,5 or 1-3")
    lines.append("Press Enter to run all discovered streams.")
    return "\n".join(lines)


def parse_stream_selection(selection, stream_count):
    selected_indexes = []
    seen_indexes = set()

    for raw_part in selection.split(","):
        part = raw_part.strip()
        if not part:
            continue

        if "-" in part:
            start_text, end_text = [value.strip() for value in part.split("-", 1)]
            if not start_text.isdigit() or not end_text.isdigit():
                raise ValueError(f"Invalid range: {part}")
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid descending range: {part}")
            candidates = range(start, end + 1)
        else:
            if not part.isdigit():
                raise ValueError(f"Invalid stream number: {part}")
            candidates = (int(part),)

        for index in candidates:
            if index < 1 or index > stream_count:
                raise ValueError(f"Stream number out of range: {index}")
            if index not in seen_indexes:
                selected_indexes.append(index)
                seen_indexes.add(index)

    return selected_indexes


def select_streams(streams):
    if not streams:
        return []

    print(stream_choice_text(streams))
    while True:
        selection = input("Streams to run: ").strip()
        if not selection:
            return list(streams)

        try:
            selected_indexes = parse_stream_selection(selection, len(streams))
        except ValueError as exc:
            print(exc)
            continue

        if selected_indexes:
            return [streams[index - 1] for index in selected_indexes]

        print("Select at least one stream, or press Enter to run all.")


def prompt_sensitivity(stream):
    while True:
        raw_value = input(f"Sensitivity for {stream.label} [{DEFAULT_MOTION_SENSITIVITY}]: ").strip()
        if not raw_value:
            return DEFAULT_MOTION_SENSITIVITY
        if not raw_value.isdigit():
            print("Enter a number from 1 to 10.")
            continue

        value = int(raw_value)
        if 1 <= value <= 10:
            return value

        print("Enter a number from 1 to 10.")


def parse_sensitivity_value(raw_value):
    raw_value = str(raw_value).strip()
    if not raw_value.isdigit():
        raise ValueError("sensitivity must be a number from 1 to 10")

    value = int(raw_value)
    if 1 <= value <= 10:
        return value

    raise ValueError("sensitivity must be a number from 1 to 10")


def prompt_stream_sensitivities(streams):
    print("Set motion sensitivity per selected stream. 1 is least sensitive, 10 is most sensitive.")
    return {
        stream.stream_id: prompt_sensitivity(stream)
        for stream in streams
    }


def load_headless_stream_config(all_streams, config, config_path):
    if not config:
        return None

    try:
        configured_streams = config.get("streams", [])
        if not configured_streams:
            raise ValueError("streams list is empty")

        streams_by_id = {stream.stream_id: stream for stream in all_streams}
        selected_streams = []
        stream_sensitivities = {}
        missing_stream_ids = []

        for configured_stream in configured_streams:
            stream_id = str(configured_stream.get("stream_id", "")).strip()
            if stream_id not in streams_by_id:
                missing_stream_ids.append(stream_id or "<missing>")
                continue

            selected_streams.append(streams_by_id[stream_id])
            stream_sensitivities[stream_id] = parse_sensitivity_value(
                configured_stream.get("sensitivity", DEFAULT_MOTION_SENSITIVITY)
            )

        if missing_stream_ids:
            raise ValueError(f"configured stream(s) not found: {', '.join(missing_stream_ids)}")

        logging.info("Loaded headless stream config from %s", config_path)
        return selected_streams, stream_sensitivities
    except (AttributeError, TypeError, ValueError) as exc:
        logging.warning("Invalid stream settings in %s: %s", config_path, exc)
        return None


def write_private_json(path, config):
    data = json.dumps(config, indent=2)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(data)
            file.write("\n")
    finally:
        try:
            os.chmod(path, 0o600)
        except OSError as exc:
            logging.warning("Could not restrict permissions on %s: %s", path, exc)


def save_headless_config(settings, streams, stream_sensitivities):
    config = {
        "settings": {
            "username": settings["username"],
            "password": settings["password"],
            "ip_address": settings["ip_address"],
            "port": settings["port"],
            "rtsp_path_template": settings["rtsp_path_template"],
        },
        "streams": [
            {
                "stream_id": stream.stream_id,
                "label": stream.label,
                "sensitivity": stream_sensitivities[stream.stream_id],
            }
            for stream in streams
        ]
    }

    try:
        write_private_json(HEADLESS_CONFIG_FILE, config)
        logging.info("Saved headless config to %s with owner-only file permissions", HEADLESS_CONFIG_FILE)
    except OSError as exc:
        logging.warning("Could not save %s: %s", HEADLESS_CONFIG_FILE, exc)


def run_test_video_clip(video_path, motion_sensitivity):
    video_path = Path(video_path).expanduser()
    if not video_path.is_file():
        logging.error("Video clip not found: %s", video_path)
        return 1

    stop_event = threading.Event()

    def stop(_signum, _frame):
        logging.info("Stop requested.")
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    logging.info("Testing video clip: %s", video_path)
    try:
        run_video_clip_detection(
            str(video_path),
            DEFAULT_TARGET_OBJECTS,
            stop_event.is_set,
            None,
            motion_sensitivity,
        )
    except Exception:
        logging.exception("Video clip test failed: %s", video_path)
        return 1

    logging.info("Video clip test complete: %s", video_path)
    return 0


def main(argv=None):
    configure_logging()
    args = parse_args(argv)

    if args.test_video_clip:
        return run_test_video_clip(args.test_video_clip, args.test_video_sensitivity)

    saved_config, saved_config_path = load_headless_config()
    settings = settings_from_config(saved_config)
    if settings is None:
        settings = build_settings()

    logging.info("Discovering streams for %s", settings["ip_address"])
    streams = discover_nvr_streams(
        settings,
        can_open_rtsp_channel,
        progress_callback=logging.info,
    )
    if not streams:
        logging.error("No camera streams discovered.")
        return 1

    configured_streams = load_headless_stream_config(streams, saved_config, saved_config_path)
    if configured_streams is None:
        streams = select_streams(streams)
        stream_sensitivities = prompt_stream_sensitivities(streams)
        save_headless_config(settings, streams, stream_sensitivities)
    else:
        streams, stream_sensitivities = configured_streams
        if saved_config_path != HEADLESS_CONFIG_FILE or settings_from_config(saved_config) is None:
            save_headless_config(settings, streams, stream_sensitivities)

    stop_event = threading.Event()

    def stop(_signum, _frame):
        logging.info("Stop requested.")
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    threads = []
    for stream in streams:
        thread = threading.Thread(
            target=run_stream,
            args=(stream, settings, stop_event, stream_sensitivities[stream.stream_id]),
            daemon=False,
        )
        thread.start()
        threads.append(thread)

    logging.info("Running %s stream(s). Stop with SIGINT or SIGTERM.", len(threads))
    while any(thread.is_alive() for thread in threads):
        time.sleep(0.5)

    return 0


if __name__ == "__main__":
    sys.exit(main())
