import json
import logging
import mimetypes
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from app_config import load_dotenv


DEFAULT_TIMEOUT_SECONDS = 120
UPLOAD_URL_ENDPOINT = "/clips/upload-url"


@dataclass(frozen=True)
class CloudUploadConfig:
    api_base_url: str
    device_id: str
    device_token: str

    @classmethod
    def from_environment(cls) -> Optional["CloudUploadConfig"]:
        load_dotenv()
        api_base_url = os.getenv("API_BASE_URL", "").strip().rstrip("/")
        device_id = os.getenv("DEVICE_ID", "").strip()
        device_token = os.getenv("DEVICE_TOKEN", "").strip()

        if not api_base_url or not device_id or not device_token:
            return None

        return cls(
            api_base_url=api_base_url,
            device_id=device_id,
            device_token=device_token,
        )


def upload_clip_async(clip_path: str, metadata: Dict[str, Any]) -> None:
    logging.info("Cloud upload queued for clip: %s", clip_path)
    thread = threading.Thread(
        target=upload_clip,
        args=(clip_path, metadata),
        daemon=True,
    )
    thread.start()


def upload_clip(clip_path: str, metadata: Dict[str, Any], config: Optional[CloudUploadConfig] = None) -> bool:
    logging.info("Cloud upload started for clip: %s", clip_path)
    config = config or CloudUploadConfig.from_environment()
    if config is None:
        logging.info("Cloud upload skipped because API environment config is missing")
        return False

    clip_file = Path(clip_path)
    if not clip_file.is_file():
        logging.warning("Cloud upload skipped because clip file does not exist: %s", clip_file)
        return False

    content_type = mimetypes.guess_type(clip_file.name)[0] or "video/mp4"
    upload_request = build_upload_url_request(config, clip_file, content_type, metadata)
    logging.info(
        "Requesting cloud upload URL for clip %s from %s",
        clip_file.name,
        config.api_base_url,
    )
    upload_details = post_json(config, UPLOAD_URL_ENDPOINT, upload_request)
    if not upload_details:
        logging.warning("Cloud upload aborted because upload URL request failed for clip: %s", clip_file)
        return False

    upload_url = upload_details.get("upload_url") or upload_details.get("uploadUrl")
    if not upload_url:
        logging.warning("Cloud upload failed because /clips/upload-url did not return upload_url")
        return False

    if not upload_to_blob(upload_url, clip_file, content_type):
        logging.warning("Cloud upload aborted because blob upload failed for clip: %s", clip_file)
        return False

    clip_id = upload_details.get("clip_id") or upload_details.get("clipId")
    if clip_id:
        logging.info("Reporting cloud upload completion for clip_id=%s", clip_id)
        complete_clip_upload(config, str(clip_id), clip_file, upload_details)

    logging.info("Cloud upload completed for clip: %s", clip_file)
    return True


def build_upload_url_request(
    config: CloudUploadConfig,
    clip_file: Path,
    content_type: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "device_id": config.device_id,
        "filename": clip_file.name,
        "content_type": content_type,
        "size_bytes": clip_file.stat().st_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **metadata,
    }


def post_json(config: CloudUploadConfig, endpoint: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    url = urljoin(f"{config.api_base_url}/", endpoint.lstrip("/"))
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.device_token}",
            "X-Device-ID": config.device_id,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            response_body = response.read().decode("utf-8")
            if not response_body:
                logging.info("Cloud API response for %s: HTTP %s with empty body", endpoint, response.status)
                return {}
            parsed_response = json.loads(response_body)
            logging.info(
                "Cloud API response for %s: HTTP %s with fields: %s",
                endpoint,
                response.status,
                sorted(parsed_response.keys()),
            )
            return parsed_response
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        logging.warning("Cloud API error %s for %s: %s", exc.code, endpoint, error_body)
    except URLError as exc:
        logging.warning("Cloud API request failed for %s: %s", endpoint, exc.reason)
    except TimeoutError:
        logging.warning("Cloud API request timed out for %s", endpoint)
    except json.JSONDecodeError as exc:
        logging.warning("Cloud API returned invalid JSON for %s: %s", endpoint, exc)

    return None


def upload_to_blob(upload_url: str, clip_file: Path, content_type: str) -> bool:
    try:
        logging.info("Uploading clip to Azure Blob Storage: %s", clip_file)
        with clip_file.open("rb") as file_handle:
            request = Request(
                upload_url,
                data=file_handle.read(),
                headers={
                    "x-ms-blob-type": "BlockBlob",
                    "Content-Type": content_type,
                },
                method="PUT",
            )
            with urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
                success = 200 <= response.status < 300
                logging.info("Azure Blob upload response for %s: HTTP %s", clip_file, response.status)
                return success
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        logging.warning("Azure Blob upload error %s: %s", exc.code, error_body)
    except URLError as exc:
        logging.warning("Azure Blob upload failed: %s", exc.reason)
    except TimeoutError:
        logging.warning("Azure Blob upload timed out")

    return False


def complete_clip_upload(
    config: CloudUploadConfig,
    clip_id: str,
    clip_file: Path,
    upload_details: Dict[str, Any],
) -> None:
    payload = {
        "device_id": config.device_id,
        "filename": clip_file.name,
        "size_bytes": clip_file.stat().st_size,
        "blob_url": upload_details.get("blob_url") or upload_details.get("blobUrl"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    post_json(config, f"/clips/{clip_id}/complete", payload)
