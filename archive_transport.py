"""Streaming transport for the dedicated, non-notifying media archive API."""
import base64
import http.client
import mimetypes
from urllib.parse import urlsplit

from cloud_upload import CloudUploadConfig, post_json


def upload_archive(media_id, path, kind, metadata, digest, size, config=None):
    config = config or CloudUploadConfig.from_environment()
    if config is None:
        return False
    mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
    md5 = base64.b64encode(digest).decode('ascii')
    details = post_json(config, '/archive/uploads', {
        'media_id': media_id, 'kind': kind, 'filename': path.name,
        'content_type': mime, 'size_bytes': size, 'content_md5': md5, 'metadata': metadata,
    })
    if not details:
        return False
    if details.get('status') != 'verified':
        upload_url = details.get('upload_url', '')
        parsed = urlsplit(upload_url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
            return False
        connection = http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=120)
        try:
            target = parsed.path + ('?' + parsed.query if parsed.query else '')
            connection.putrequest('PUT', target)
            connection.putheader('Content-Length', str(size))
            connection.putheader('Content-Type', mime)
            connection.putheader('Content-MD5', md5)
            connection.putheader('x-ms-blob-content-md5', md5)
            connection.putheader('x-ms-blob-type', 'BlockBlob')
            connection.endheaders()
            with path.open('rb') as source:
                while chunk := source.read(1024 * 1024):
                    connection.send(chunk)
            response = connection.getresponse()
            response.read()
            if response.status not in (200, 201):
                return False
        finally:
            connection.close()
    result = post_json(config, f'/archive/{media_id}/verify', {})
    return bool(result and result.get('status') == 'verified')
