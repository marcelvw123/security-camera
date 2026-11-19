import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import object_alerts
import security_camera_core as core
from motion_clip_metadata import start_metadata, update_metadata


class ObjectAlertTests(unittest.TestCase):
    def test_only_detected_objects_queue_an_alert(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict('os.environ', {'WHATSAPP_OBJECT_ALERTS_ENABLED': 'true'}), \
             patch('object_alerts.queue_object_alert') as queue_alert:
            empty = Path(directory) / 'empty.mp4'
            start_metadata(empty); update_metadata(empty, [])
            core.finalize_recorded_clip(empty, 'Camera 801', None, upload_on_motion=True)
            queue_alert.assert_not_called()
            detected = Path(directory) / 'detected.mp4'
            start_metadata(detected); update_metadata(detected, ['car'], 'car.jpg')
            core.finalize_recorded_clip(detected, 'Camera 801', None, upload_on_motion=True)
            queue_alert.assert_called_once_with(detected, 'Camera 801', ['car'])

    def test_disabled_alerts_never_queue(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict('os.environ', {'WHATSAPP_OBJECT_ALERTS_ENABLED': 'false'}), \
             patch('object_alerts.queue_object_alert') as queue_alert:
            clip = Path(directory) / 'detected.mp4'
            start_metadata(clip); update_metadata(clip, ['person'])
            core.finalize_recorded_clip(clip, 'Camera 701', None, upload_on_motion=True)
            queue_alert.assert_not_called()

    def test_video_alert_uses_object_scenario(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(object_alerts, 'ffmpeg_executable', return_value='ffmpeg'), \
             patch.object(object_alerts.subprocess, 'run'), \
             patch.object(object_alerts, 'upload_clip', return_value=True) as upload:
            self.assertTrue(object_alerts._send(Path(directory)/'clip.mp4', 'Camera 801', ['car']))
            metadata = upload.call_args.args[1]
            self.assertEqual(metadata['scenario'], 'object_detected')
            self.assertEqual(metadata['detected_objects'], ['car'])
            self.assertFalse(metadata['upload_on_motion'])
