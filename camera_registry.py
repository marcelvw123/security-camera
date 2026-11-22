"""Private, crash-safe storage for the cameras the operator wants running."""
import json
import os
import tempfile
import threading
from pathlib import Path


class CameraRegistry:
    def __init__(self, path=None):
        self.path = Path(path or (Path.home() / '.local/state/securitycamera/cameras.json'))
        self.lock = threading.RLock()

    def read(self):
        with self.lock:
            if not self.path.exists():
                return {}
            data = json.loads(self.path.read_text())
            if not isinstance(data, dict) or any(not isinstance(v, dict) for v in data.values()):
                raise ValueError('Invalid camera registry')
            return data

    def _write(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        temp = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=self.path.parent, delete=False) as output:
                temp = output.name
                os.fchmod(output.fileno(), 0o600)
                json.dump(data, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temp and os.path.exists(temp):
                os.unlink(temp)

    def put(self, source_id, payload):
        with self.lock:
            data = self.read()
            if source_id not in data and len(data) >= 4:
                raise ValueError('At most four cameras may be selected')
            data[source_id] = payload
            self._write(data)

    def remove(self, source_id):
        with self.lock:
            data = self.read()
            if source_id in data:
                del data[source_id]
                self._write(data)
