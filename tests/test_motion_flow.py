import unittest
import cv2
import numpy as np
from motion_flow import motion_flow_evidence


class MotionFlowTests(unittest.TestCase):
    def test_static_and_translated_texture(self):
        rng = np.random.default_rng(5)
        original = rng.integers(0, 256, (100, 100), dtype=np.uint8)
        moved = cv2.warpAffine(original, np.float32([[1, 0, 4], [0, 1, 0]]), (100, 100))
        box = [[15, 15, 85, 85]]
        static = motion_flow_evidence(original, original.copy(), .5, box, 100, 100)[0]
        moving = motion_flow_evidence(original, moved, .5, box, 100, 100)[0]
        self.assertLess(static, .1)
        self.assertGreater(moving, 1.)

    def test_invalid_pair_fails_closed(self):
        img = np.zeros((50, 50), np.uint8)
        self.assertEqual(motion_flow_evidence(None, img, .5, [[0, 0, 10, 10]], 50, 50), [None])
        self.assertEqual(motion_flow_evidence(img, img, 3, [[0, 0, 10, 10]], 50, 50), [None])


if __name__ == '__main__':
    unittest.main()
