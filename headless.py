import getpass
import logging
import os
import signal
import sys
import threading
import time

from app_config import load_dotenv
from camera_discovery import discover_nvr_streams
from security_camera_core import (
    DEFAULT_RTSP_PATH_TEMPLATE,
    DEFAULT_RTSP_PORT,
    DEFAULT_TARGET_OBJECTS,
    build_rtsp_url,
    can_open_rtsp_channel,
    run_detection,
)


LOG_FILE = "security_camera_headless.log"


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


def run_stream(stream, settings, stop_event):
    rtsp_url = stream_url_for(settings, stream)
    logging.info("Starting stream: %s", stream.label)
    try:
        run_detection(
            rtsp_url,
            stream.label,
            DEFAULT_TARGET_OBJECTS,
            stop_event.is_set,
            None,
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
    default_ip = env_value("CAMERA_IP", "NVR_IP")
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


def main():
    configure_logging()

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

    streams = select_streams(streams)

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
            args=(stream, settings, stop_event),
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
