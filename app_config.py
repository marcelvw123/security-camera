import os
import sys
from pathlib import Path


ENV_FILE_NAME = ".env"


def app_base_path() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def dotenv_candidate_paths():
    return (
        Path.cwd() / ENV_FILE_NAME,
        app_base_path() / ENV_FILE_NAME,
        Path.home() / "SecurityCamera" / ENV_FILE_NAME,
    )


def load_dotenv() -> None:
    for env_path in dotenv_candidate_paths():
        if not env_path.is_file():
            continue

        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("#") or "=" not in stripped_line:
                continue

            key, value = stripped_line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value
        return
