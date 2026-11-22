"""Authenticated access to this device's diagnostic archive."""
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen
from cloud_upload import CloudUploadConfig


def archive_request(path, payload=None, config=None):
    config = config or CloudUploadConfig.from_environment()
    if config is None:
        return None
    request = Request(urljoin(config.api_base_url + '/', path.lstrip('/')),
        data=None if payload is None else json.dumps(payload).encode('utf-8'),
        headers={'Authorization': f'Bearer {config.device_token}', 'X-Device-ID': config.device_id,
                 'Content-Type': 'application/json'}, method='GET' if payload is None else 'POST')
    try:
        with urlopen(request, timeout=5) as response:
            return json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        return None


def list_archive(kind):
    return archive_request('/archive?' + urlencode({'kind': kind}))
