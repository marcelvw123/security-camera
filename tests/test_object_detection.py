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
    def test_idle_baseline_does_not_write_detection_images_or_start_clips(self):
        detector = MagicMock()
        frame = np.zeros((100,100,3), np.uint8)
        detector.take.return_value = (1,100,[detection('car',10,w=30,h=30)],frame)
        detector.take_remote.return_value = None
        with tempfile.TemporaryDirectory() as directory:
            recorder=ObjectClipRecorder(detector,'camera',directory,MagicMock(),MagicMock(),MagicMock(),min_free_bytes=0)
            with patch.object(recorder,'save_evidence') as save:
                recorder.observe(frame,1,False)
            save.assert_not_called()
            self.assertIsNone(recorder.writer)

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


if __name__ == '__main__':
    unittest.main()
