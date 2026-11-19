import unittest
import numpy as np
from motion_difference import FrameDifferenceMotion


class FrameDifferenceTests(unittest.TestCase):
    def test_stable_frame_quiet_and_large_movement_is_detected(self):
        detector = FrameDifferenceMotion()
        frame = np.zeros((270, 480, 3), np.uint8)
        self.assertEqual(detector.observe(frame, 0), [])
        self.assertEqual(detector.observe(frame, .5), [])
        changed = frame.copy(); changed[100:145, 200:250] = 255
        boxes = detector.observe(changed, 1)
        self.assertTrue(any(x <= 200 < x+w and y <= 100 < y+h for x,y,w,h in boxes))

    def test_reconnect_gap_primes_baseline_instead_of_triggering(self):
        detector = FrameDifferenceMotion(max_gap_seconds=2)
        frame = np.zeros((270,480,3), np.uint8)
        detector.observe(frame, 0)
        self.assertEqual(detector.observe(np.full_like(frame, 255), 3), [])

    def test_initial_and_nonmonotonic_timestamps(self):
        detector = FrameDifferenceMotion()
        frame = np.zeros((270,480,3), np.uint8)
        detector.observe(frame, 1)
        with self.assertRaises(ValueError): detector.observe(frame, 1)
        detector.reset()
        self.assertEqual(detector.observe(frame, 0), [])

    def test_low_contrast_is_not_silently_counted_as_detected(self):
        detector = FrameDifferenceMotion(pixel_threshold=25, min_area=120)
        base = np.zeros((270,480,3), np.uint8)
        changed = base.copy(); changed[100:160, 200:300] = 20
        detector.observe(base, 0)
        self.assertEqual(detector.observe(changed, .5), [])

    def test_more_sensitive_pixels_can_detect_low_contrast_change(self):
        detector = FrameDifferenceMotion(pixel_threshold=15, min_area=120)
        base = np.zeros((270,480,3), np.uint8)
        changed = base.copy(); changed[100:160, 200:300] = 20
        detector.observe(base, 0)
        self.assertTrue(detector.observe(changed, .5))

    def test_exclusion_suppresses_only_that_region(self):
        detector = FrameDifferenceMotion(pixel_threshold=15, min_area=120, exclusions=[[0, 0, .5, 1]])
        base = np.zeros((270,480,3), np.uint8)
        changed = base.copy(); changed[100:160, 80:150] = 255; changed[100:160, 300:370] = 255
        detector.observe(base, 0)
        boxes = detector.observe(changed, .5)
        self.assertTrue(any(x >= 240 for x, y, w, h in boxes))
        self.assertFalse(any(x < 240 for x, y, w, h in boxes))

    def test_invalid_exclusions_are_rejected(self):
        for zones in ([[0, 0, 2, 1]], [[0, 0, -1, 1]], [[0, 0, 1]], 'bad'):
            with self.subTest(zones=zones), self.assertRaises(ValueError):
                FrameDifferenceMotion(exclusions=zones)


if __name__ == '__main__':
    unittest.main()
