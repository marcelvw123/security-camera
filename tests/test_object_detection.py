import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from object_detection import ObjectGate, ObjectClipRecorder, VehicleDetectionGate, RemoteVehicleShadow, LocalVehicleVerifier


def detection(label, x, y=10, w=80, h=80):
    return (label, [x, y, x+w, y+h], .9)


class ObjectGateTests(unittest.TestCase):
    def test_local_verifier_uses_custom_camera_and_invalidates_old_results(self):
        import threading
        from collections import OrderedDict
        verifier = LocalVehicleVerifier.__new__(LocalVehicleVerifier)
        verifier.condition = threading.Condition()
        verifier.crops, verifier.results, verifier.generations = {}, {'new-camera': ('old', [])}, {}
        verifier.pending, verifier.last_submitted = OrderedDict({'new-camera': 'old'}), {'new-camera': 1}
        self.assertEqual(verifier.default_crop('new-camera'), [0, 0, 1, 1])
        self.assertEqual(verifier.default_crop('probe:/Streaming/Channels/{channel}:801'), [0, .0868, .4375, .3125])
        verifier.set_crop('new-camera', [.1, .2, .3, .4])
        self.assertEqual(verifier.crops['new-camera'], [.1, .2, .3, .4])
        self.assertNotIn('new-camera', verifier.results)
        self.assertNotIn('new-camera', verifier.pending)
        self.assertEqual(verifier.crop_revision('new-camera'), 1)
        self.assertEqual(verifier.verifier_camera('new-camera'), 'new-camera')

    def test_discovered_camera_uses_its_channel_verifier(self):
        convert = RemoteVehicleShadow.verifier_camera
        self.assertEqual(convert('probe:/Streaming/Channels/{channel}:801'), 'manual:801')
        self.assertEqual(convert('manual:701'), 'manual:701')
        self.assertIsNone(convert('probe:/other:801'))

    def test_vehicle_detection_starts_immediately_and_rearms_after_quiet(self):
        gate = VehicleDetectionGate()
        self.assertTrue(gate.update(1, ['car']))
        self.assertFalse(gate.update(2, ['car']))
        self.assertFalse(gate.update(3, []))
        self.assertFalse(gate.update(10, []))
        self.assertTrue(gate.update(11, ['car']))
        self.assertFalse(gate.update(9, []))

    def test_late_positive_still_triggers_after_newer_empty_sample(self):
        gate = VehicleDetectionGate()
        self.assertFalse(gate.update(5, []))
        self.assertTrue(gate.update(4, ['car']))

    def test_person_needs_two_observations_and_does_not_retrigger_until_quiet(self):
        gate = ObjectGate(quiet_seconds=8)
        self.assertEqual(gate.update([detection('person', 100)], 1, 1000), [])
        self.assertEqual(gate.update([detection('person', 105)], 2, 1000), ['person'])
        self.assertEqual(gate.update([detection('person', 110)], 3, 1000), [])
        gate.update([], 12, 1000)
        gate.update([detection('person', 100)], 13, 1000)
        self.assertEqual(gate.update([detection('person', 102)], 14, 1000), ['person'])

class ObjectClipTests(unittest.TestCase):
    def test_evidence_is_saved_as_an_annotated_source_frame(self):
        import cv2
        detector = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            recorder = ObjectClipRecorder(detector, 'camera:1', directory, MagicMock(), MagicMock(), MagicMock(), min_free_bytes=0)
            frame = np.zeros((100, 120, 3), np.uint8)
            name = recorder.save_evidence(frame, [{'label': 'car', 'box': [20, 25, 60, 70], 'confidence': .8}])
            saved = cv2.imread(str(Path(directory) / 'detection_frames' / name))
            self.assertEqual(saved.shape, frame.shape)
            self.assertGreater(int(saved[25, 20, 1]), 100)

    def test_first_vehicle_detection_starts_separate_object_clip(self):
        detector, callback = MagicMock(), MagicMock()
        detector.take.return_value = None
        detector.take_remote.side_effect = [(1, [{'label': 'car'}]), (2, [{'label': 'car'}]), None]
        frame = np.zeros((100, 100, 3), np.uint8)
        with tempfile.TemporaryDirectory() as directory, \
             patch('shutil.disk_usage', return_value=SimpleNamespace(free=1000)), \
             patch('motion_preroll.AsyncPreRollWriter') as writer_class:
            writer_class.return_value.path = Path(directory) / 'object.mp4'
            factory = MagicMock()
            recorder = ObjectClipRecorder(detector, 'manual:801', directory, factory,
                                          callback, MagicMock(), min_free_bytes=10, seconds=20)
            for t in (1, 2, 21):
                recorder.observe(frame, t, True)
            self.assertEqual(factory.call_count, 1)
            self.assertEqual([c.args[0] for c in callback.call_args_list], [True, False])

    def test_first_full_frame_vehicle_detection_also_starts_clip(self):
        from motion_clip_metadata import load_metadata
        detector, callback = MagicMock(), MagicMock()
        detected_frame = np.zeros((100, 100, 3), np.uint8)
        detected_frame[50, 50] = [80, 130, 180]
        detector.take.return_value = (1, 100, [detection('car', 10, w=30, h=30)], detected_frame)
        detector.take_remote.return_value = None
        with tempfile.TemporaryDirectory() as directory, \
             patch('shutil.disk_usage', return_value=SimpleNamespace(free=1000)), \
             patch('motion_preroll.AsyncPreRollWriter') as writer_class:
            writer_class.return_value.path = Path(directory) / 'object.mp4'
            factory = MagicMock()
            recorder = ObjectClipRecorder(detector, 'manual:701', directory, factory,
                                          callback, MagicMock(), min_free_bytes=10)
            checks = recorder.observe(np.zeros((100,100,3), np.uint8), 1, True)
            factory.assert_called_once()
            self.assertTrue(callback.call_args.args[0])
            self.assertTrue((Path(directory) / 'detection_frames' / checks[0][2]).is_file())
            self.assertEqual(load_metadata(Path(directory) / 'object.mp4')['evidence'], [checks[0][2]])
            recorder.stop()

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
