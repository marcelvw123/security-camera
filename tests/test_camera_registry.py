import json
import os
import tempfile
import unittest
from pathlib import Path

from camera_registry import CameraRegistry


class CameraRegistryTests(unittest.TestCase):
    def test_persists_privately_and_removes_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'private' / 'cameras.json'
            registry = CameraRegistry(path)
            payload = {'stream': {'stream_id': '701'}, 'settings': {'password': 'private'}}
            registry.put('701', payload)
            self.assertEqual(CameraRegistry(path).read()['701'], payload)
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            self.assertEqual(os.stat(path.parent).st_mode & 0o777, 0o700)
            registry.remove('701')
            self.assertEqual(registry.read(), {})

    def test_limit_and_corrupt_file_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cameras.json'
            registry = CameraRegistry(path)
            for index in range(4):
                registry.put(str(index), {})
            with self.assertRaises(ValueError):
                registry.put('fifth', {})
            registry.put('0', {'updated': True})
            path.write_text('{broken')
            with self.assertRaises(json.JSONDecodeError):
                registry.put('other', {})
            self.assertEqual(path.read_text(), '{broken')
