import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union


FFMPEG_TIMEOUT_SECONDS = 300


def ffmpeg_executable() -> Optional[str]:
    configured_path = os.getenv("FFMPEG_PATH", "").strip()
    if configured_path:
        return configured_path

    return shutil.which("ffmpeg")


def make_whatsapp_compatible_mp4(video_path: Union[str, Path]) -> Path:
    source_path = Path(video_path)
    ffmpeg_path = ffmpeg_executable()
    if ffmpeg_path is None:
        logging.warning("ffmpeg not found; uploading original clip without WhatsApp transcode: %s", source_path)
        return source_path

    temp_path = source_path.with_name(f"{source_path.stem}.whatsapp_tmp{source_path.suffix}")
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source_path),
        "-an",
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level:v",
        "3.0",
        "-pix_fmt",
        "yuv420p",
        "-tag:v",
        "avc1",
        "-movflags",
        "+faststart",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        str(temp_path),
    ]

    try:
        logging.info("Transcoding clip for WhatsApp compatibility: %s", source_path)
        completed_process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
        if completed_process.stderr:
            logging.debug("ffmpeg output for %s: %s", source_path, completed_process.stderr[-2000:])

        temp_path.replace(source_path)
        logging.info("Transcoded WhatsApp-compatible MP4: %s", source_path)
        return source_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        logging.warning("WhatsApp transcode failed for %s: %s", source_path, exc)
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return source_path
