import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from archive_transport import upload_archive
from cloud_upload import CloudUploadConfig


class Response:
    status = 201
    def read(self): return b''

class Connection:
    instances = []
    def __init__(self, *args, **kwargs): self.sent = 0; self.headers = {}; self.__class__.instances.append(self)
    def putrequest(self, *args): pass
    def putheader(self, name, value): self.headers[name] = value
    def endheaders(self): pass
    def send(self, data): self.sent += len(data)
    def getresponse(self): return Response()
    def close(self): pass


class ArchiveTransportTests(unittest.TestCase):
    def test_streams_bytes_and_requires_server_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'clip.mp4'
            path.write_bytes(b'x' * (1024 * 1024 + 7))
            config = CloudUploadConfig('https://api.test', 'device', 'token')
            digest = hashlib.md5(path.read_bytes(), usedforsecurity=False).digest()
            with patch('archive_transport.http.client.HTTPSConnection', Connection), \
                 patch('archive_transport.post_json', side_effect=[{'status':'pending', 'upload_url':'https://blob.test/file?sas=secret'}, {'status':'verified'}]) as post:
                self.assertTrue(upload_archive('id', path, 'motion', {}, digest, path.stat().st_size, config))
                self.assertEqual(Connection.instances[-1].sent, path.stat().st_size)
                self.assertIn('Content-MD5', Connection.instances[-1].headers)
                self.assertEqual(post.call_count, 2)
            with patch('archive_transport.post_json', return_value={'status':'pending','upload_url':'http://blob.test'}):
                self.assertFalse(upload_archive('id', path, 'motion', {}, digest, path.stat().st_size, config))
