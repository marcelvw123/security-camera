import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import webapp
from camera_registry import CameraRegistry


class CameraRestoreTests(unittest.TestCase):
    def test_start_restore_stop_and_no_duplicate_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = CameraRegistry(Path(directory) / 'cameras.json')
            state = webapp.WebAppState()
            threads = []
            class FakeThread:
                def __init__(self, **kwargs): self.started = False; threads.append(self)
                def start(self): self.started = True
            payload = {'stream': {'stream_id': '701', 'label': 'Front'},
                       'settings': {'password': 'not-for-logs'}, 'sensitivity': 3}
            with patch.object(webapp, 'CAMERA_REGISTRY', registry), patch.object(webapp, 'APP_STATE', state), \
                 patch.object(webapp.threading, 'Thread', FakeThread), patch.object(webapp, 'OBJECT_DETECTOR', None), \
                 patch.object(webapp, 'load_exclusion_map', return_value={}), patch.object(webapp, 'load_vehicle_areas', return_value={}):
                worker = webapp.start_camera(payload)
                self.assertTrue(threads[0].started)
                self.assertEqual(registry.read()['701']['sensitivity'], 3)
                with self.assertRaises(ValueError): webapp.start_camera(payload)
                state.remove_worker('701')
                restored = webapp.start_camera(registry.read()['701'], restore=True)
                self.assertEqual(restored.sensitivity, 3)
                state.remove_worker('701')
                registry.remove('701')
                with self.assertRaises(ValueError): webapp.start_camera(payload, restore=True)
