import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from motion_shadow import MotionShadowComparison


class MotionShadowTests(unittest.TestCase):
    def test_reports_disagreement_without_returning_recording_decision(self):
        shadow=MotionShadowComparison('camera', report_interval=.4)
        frame=np.zeros((270,480,3),np.uint8)
        with patch('motion_shadow.logging.info') as log:
            self.assertIsNone(shadow.observe(frame,0,True))
            self.assertIsNone(shadow.observe(frame,.5,True))
            self.assertIsNone(shadow.observe(frame,1,True))
        self.assertEqual(log.call_count,1)
        self.assertEqual(log.call_args.args[2:], (2,2,0,2,0))

    def test_comparison_failure_does_not_escape(self):
        shadow=MotionShadowComparison('camera')
        shadow.filter.observe=MagicMock(side_effect=RuntimeError('test error'))
        with patch('motion_shadow.logging.exception') as log:
            self.assertIsNone(shadow.observe(np.zeros((5,5,3),np.uint8),1,True))
        log.assert_called_once()


if __name__=='__main__':
    unittest.main()
