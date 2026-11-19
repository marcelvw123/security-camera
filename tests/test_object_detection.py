import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from object_detection import ObjectGate, ObjectClipRecorder, RoadEntryGate, RemoteVehicleShadow


def detection(label, x, y=10, w=80, h=80):
    return (label, [x, y, x+w, y+h], .9)


class ObjectGateTests(unittest.TestCase):
    def test_discovered_camera_uses_its_channel_verifier(self):
        convert = RemoteVehicleShadow.verifier_camera
        self.assertEqual(convert('probe:/Streaming/Channels/{channel}:801'), 'manual:801')
        self.assertEqual(convert('manual:701'), 'manual:701')
        self.assertIsNone(convert('probe:/other:801'))

    def test_road_entry_requires_recent_empty_observation_and_cooldown(self):
        gate = RoadEntryGate()
        self.assertFalse(gate.update(1, [{'label': 'car'}]))
        self.assertFalse(gate.update(2, []))
        self.assertTrue(gate.update(3, [{'label': 'car'}]))
        self.assertFalse(gate.update(4, []))
        self.assertFalse(gate.update(5, [{'label': 'car'}]))
        self.assertFalse(gate.update(30, []))
        self.assertFalse(gate.update(40, [{'label': 'car'}]))

    def test_person_needs_two_observations_and_does_not_retrigger_until_quiet(self):
        gate = ObjectGate(quiet_seconds=8)
        self.assertEqual(gate.update([detection('person', 100)], 1, 1000), [])
        self.assertEqual(gate.update([detection('person', 105)], 2, 1000), ['person'])
        self.assertEqual(gate.update([detection('person', 110)], 3, 1000), [])
        gate.update([], 12, 1000)
        gate.update([detection('person', 100)], 13, 1000)
        self.assertEqual(gate.update([detection('person', 102)], 14, 1000), ['person'])

    def test_stationary_vehicle_ignored_and_moving_vehicle_accepted(self):
        gate = ObjectGate()
        self.assertEqual(gate.update([detection('car', 100)], 1, 1000), [])
        self.assertEqual(gate.update([detection('car', 103)], 2, 1000), [])
        self.assertEqual(gate.update([detection('car', 140)], 3, 1000), ['vehicle'])

    def test_large_box_change_not_mistaken_for_vehicle_motion(self):
        gate = ObjectGate()
        gate.update([detection('car', 100, w=100)], 1, 1000)
        self.assertEqual(gate.update([detection('bus', 100, w=400)], 2, 1000), [])


class ObjectClipTests(unittest.TestCase):
    def test_remote_entry_starts_separate_object_clip(self):
        detector, callback = MagicMock(), MagicMock()
        detector.take.return_value = None
        detector.take_remote.side_effect = [(1, []), (2, [{'label': 'car'}]), None]
        frame = np.zeros((100, 100, 3), np.uint8)
        with tempfile.TemporaryDirectory() as directory, \
             patch('shutil.disk_usage', return_value=SimpleNamespace(free=1000)), \
             patch('motion_preroll.AsyncPreRollWriter') as writer_class:
            writer_class.return_value.path = Path(directory) / 'object.mp4'
            factory = MagicMock()
            recorder = ObjectClipRecorder(detector, 'manual:801', directory, factory,
                                          callback, MagicMock(), min_free_bytes=10, seconds=20)
            for t in (1, 2, 22):
                recorder.observe(frame, t, True)
            self.assertEqual(factory.call_count, 1)
            self.assertEqual([c.args[0] for c in callback.call_args_list], [True, False])

    def test_single_clip_for_confirmed_person_and_close(self):
        detector = MagicMock(); callback = MagicMock(); raw = MagicMock()
        detector.take_remote.return_value = None
        frame = np.zeros((100, 100, 3), np.uint8)
        detector.take.side_effect = [
            (1, 100, [detection('person', 10, w=30, h=30)]),
            (2, 100, [detection('person', 11, w=30, h=30)]),
            (3, 100, [detection('person', 12, w=30, h=30)]),
            None,
        ]
        with tempfile.TemporaryDirectory() as directory, \
             patch('shutil.disk_usage', return_value=SimpleNamespace(free=1000)), \
             patch('motion_preroll.AsyncPreRollWriter') as writer_class:
            writer = writer_class.return_value
            writer.path = Path(directory) / 'object.mp4'
            factory = MagicMock(return_value=raw)
            recorder = ObjectClipRecorder(detector, 'Camera 701', directory, factory,
                                          callback, MagicMock(), min_free_bytes=10, seconds=20)
            for t in (1, 2, 3, 22):
                recorder.observe(frame, t, True)
            self.assertEqual(factory.call_count, 1)
            writer.release.assert_called_once()
            self.assertEqual([c.args[0] for c in callback.call_args_list], [True, False])
            recorder.stop()
            detector.remove.assert_called_once_with('Camera 701')


if __name__ == '__main__':
    unittest.main()
