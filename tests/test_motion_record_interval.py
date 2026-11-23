import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import webapp
from camera_registry import CameraRegistry


class MotionRecordIntervalTests(unittest.TestCase):
    def test_interval_validation(self):
        self.assertEqual(webapp.parse_motion_record_interval(None), 120)
        self.assertEqual(webapp.parse_motion_record_interval(0), 0)
        self.assertEqual(webapp.parse_motion_record_interval(120), 120)
        for invalid in (True, -1, 19, 3601, 1.5, '120'):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                webapp.parse_motion_record_interval(invalid)

    def test_per_camera_interval_survives_stop_and_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = CameraRegistry(root / 'registry.json')
            state = webapp.WebAppState()
            stream = {'stream': {'stream_id': '701', 'label': 'Front'}, 'settings': {}}
            handler = MagicMock()
            handler.read_json.return_value = {'source_id': '701', 'interval_seconds': 180}
            with patch.object(webapp, 'APP_STATE', state), patch.object(webapp, 'CAMERA_REGISTRY', registry), \
                 patch.object(webapp, 'MOTION_INTERVALS_FILE', root/'intervals.json'), \
                 patch.object(webapp.threading, 'Thread'), patch.object(webapp, 'OBJECT_DETECTOR', None), \
                 patch.object(webapp, 'load_vehicle_areas', return_value={}), \
                 patch.object(webapp, 'load_exclusion_map', return_value={}):
                first = webapp.start_camera(stream)
                self.assertEqual(first.motion_record_interval_seconds,120)
                webapp.SecurityCameraWebHandler.handle_recording_interval(handler)
                self.assertEqual(first.motion_record_interval_seconds,180)
                self.assertEqual(registry.read()['701']['motion_record_interval_seconds'],180)
                state.remove_worker('701'); registry.remove('701')
                restarted = webapp.start_camera(stream)
                self.assertEqual(restarted.motion_record_interval_seconds,180)

    def test_other_camera_interval_is_independent(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(webapp, 'MOTION_INTERVALS_FILE', Path(directory)/'intervals.json'):
            webapp.save_motion_record_interval('701', 180)
            webapp.save_motion_record_interval('801', 60)
            self.assertEqual(webapp.load_motion_record_intervals(), {'701':180, '801':60})
