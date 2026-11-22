import json
import http.client
import logging
import mimetypes
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
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
        load_dotenv(override_empty_keys=('API_BASE_URL', 'DEVICE_ID', 'DEVICE_TOKEN'))
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


def upload_file_async(file_path: str, metadata: Dict[str, Any], container: Optional[str] = None) -> None:
    logging.info("Cloud upload queued for file: %s", file_path)
    thread = threading.Thread(
        target=upload_file,
        args=(file_path, metadata, container),
        daemon=True,
    )
    thread.start()


def upload_clip_async(clip_path: str, metadata: Dict[str, Any]) -> None:
    upload_file_async(clip_path, metadata)


def upload_file(
    file_path: str,
    metadata: Dict[str, Any],
    container: Optional[str] = None,
    config: Optional[CloudUploadConfig] = None,
) -> bool:
    logging.info("Cloud upload started for file: %s", file_path)
    config = config or CloudUploadConfig.from_environment()
    if config is None:
        logging.info("Cloud upload skipped because API environment config is missing")
        return False

    upload_file_path = Path(file_path)
    if not upload_file_path.is_file():
        logging.warning("Cloud upload skipped because file does not exist: %s", upload_file_path)
        return False

    content_type = mimetypes.guess_type(upload_file_path.name)[0] or "application/octet-stream"
    upload_request = build_upload_url_request(config, upload_file_path, content_type, metadata, container=container)
    logging.info(
        "Requesting cloud upload URL for file %s from %s%s",
        upload_file_path.name,
        config.api_base_url,
        f" container={container}" if container else "",
    )
    upload_details = post_json(config, UPLOAD_URL_ENDPOINT, upload_request)
    if not upload_details:
        logging.warning("Cloud upload aborted because upload URL request failed for file: %s", upload_file_path)
        return False

    upload_url = upload_details.get("upload_url") or upload_details.get("uploadUrl")
    if not upload_url:
        logging.warning("Cloud upload failed because /clips/upload-url did not return upload_url")
        return False

    if not upload_to_blob(upload_url, upload_file_path, content_type):
        logging.warning("Cloud upload aborted because blob upload failed for file: %s", upload_file_path)
        return False

    clip_id = upload_details.get("clip_id") or upload_details.get("clipId")
    if clip_id:
        logging.info("Reporting cloud upload completion for clip_id=%s", clip_id)
        result = complete_clip_upload(config, str(clip_id), upload_file_path, upload_details)
        if result is None:
            logging.warning('Cloud upload completion failed for clip_id=%s', clip_id)
            return False
        if 'whatsapp_alert' in result:
            logging.info('WhatsApp alert result for clip_id=%s: %s', clip_id, result['whatsapp_alert'])
        if metadata.get('scenario') == 'object_detected' and result.get('whatsapp_alert') != 'sent':
            logging.error('WhatsApp object alert not confirmed for clip_id=%s: %s', clip_id, result.get('whatsapp_alert', 'missing result'))
            return False

    logging.info("Cloud upload completed for file: %s", upload_file_path)
    return True


def upload_clip(clip_path: str, metadata: Dict[str, Any], config: Optional[CloudUploadConfig] = None) -> bool:
    return upload_file(clip_path, metadata, config=config)


def build_upload_url_request(
    config: CloudUploadConfig,
    upload_file_path: Path,
    content_type: str,
    metadata: Dict[str, Any],
    container: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {
        "device_id": config.device_id,
        "filename": upload_file_path.name,
        "content_type": content_type,
        "size_bytes": upload_file_path.stat().st_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **metadata,
    }
    if container:
        payload["container"] = container
    return payload


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
    # Stream the video: reading a whole recording into memory can exhaust a Pi.
    parsed = urlsplit(upload_url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        logging.warning('Blob upload URL was invalid')
        return False
    connection = http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=DEFAULT_TIMEOUT_SECONDS)
    try:
        target = parsed.path + ('?' + parsed.query if parsed.query else '')
        connection.putrequest('PUT', target)
        connection.putheader('Content-Length', str(clip_file.stat().st_size))
        connection.putheader('Content-Type', content_type)
        connection.putheader('x-ms-blob-type', 'BlockBlob')
        connection.endheaders()
        with clip_file.open('rb') as source:
            while chunk := source.read(1024 * 1024):
                connection.send(chunk)
        response = connection.getresponse()
        response.read()
        logging.info('Azure Blob upload response: HTTP %s', response.status)
        return 200 <= response.status < 300
    except (OSError, http.client.HTTPException):
        # Provider exceptions may contain the signed URL; do not log them.
        logging.warning('Azure Blob upload failed')
        return False
    finally:
        connection.close()


def complete_clip_upload(
    config: CloudUploadConfig,
    clip_id: str,
    clip_file: Path,
    upload_details: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    payload = {
        "device_id": config.device_id,
        "filename": clip_file.name,
        "size_bytes": clip_file.stat().st_size,
        "blob_url": upload_details.get("blob_url") or upload_details.get("blobUrl"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    return post_json(config, f"/clips/{clip_id}/complete", payload)
