import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import security_camera_core as core


class MotionCaptureTests(unittest.TestCase):
    def test_continuous_motion_stays_in_one_clip_until_quiet(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cap, writer = MagicMock(), MagicMock()
        cap.get.return_value = 20
        clock = [0.0]
        def read():
            clock[0] += 1
            return True, frame
        cap.read.side_effect = read
        with tempfile.TemporaryDirectory() as directory:
            writer.path = Path(directory) / 'motion_test.mp4'
            # Motion continues past the minimum duration, then stops.
            def boxes(*args):
                return [(2, 2, 20, 20)] if clock[0] <= 8 else []
            with patch.object(core, 'OUTPUT_DIR', Path(directory)), \
                 patch.object(core, 'CLIP_SECONDS', 2), \
                 patch.object(core, 'MOTION_QUIET_SECONDS', 3), \
                 patch.object(core, 'MOTION_WARMUP_SECONDS', 0), \
                 patch.object(core, 'MOTION_MIN_FREE_MB', 0), \
                 patch.object(core, 'open_rtsp_capture', return_value=cap), \
                 patch.object(core, 'detect_motion', side_effect=boxes), \
                 patch.object(core, 'create_video_writer', return_value=writer) as create, \
                 patch.object(core, 'emit_recording_progress', return_value=0), \
                 patch.object(core, 'write_recording_frames_until', return_value=0), \
                 patch.object(core, 'finalize_recorded_clip') as finalize, \
                 patch.object(core.time, 'monotonic', side_effect=lambda: clock[0]):
                core.run_detection('rtsp://test', 'camera', set(), lambda: clock[0] >= 13)
            self.assertEqual(create.call_count, 1)
            self.assertEqual(finalize.call_count, 1)
            self.assertNotEqual(finalize.call_args.kwargs.get('reason'), 'stream_stopped')

    def test_motion_clip_finalization_is_local_only(self):
        with patch.object(core, 'upload_clip_async') as upload, patch.object(core, 'log_trigger_step') as log:
            status, analysis = MagicMock(), MagicMock()
            core.finalize_recorded_clip('/tmp/motion_test.mp4', 'camera', None,
                                        upload_on_motion=True, status_callback=status,
                                        analysis_callback=analysis)
        upload.assert_not_called()
        self.assertEqual(log.call_args.args[1], 'MOTION_CLIP_SAVED')
        status.assert_called_once()
        self.assertIn('locally', analysis.call_args.args[1])

    def test_camera_forces_motion_and_continues_detecting_while_recording(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cap = MagicMock()
        cap.get.return_value = 20
        cap.read.return_value = (True, frame)
        writer = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            writer.path = Path(directory) / 'motion_test.mp4'
            ticks = iter(range(1000))
            stop = MagicMock(side_effect=[False, False, False, True])
            with patch.object(core, 'OUTPUT_DIR', Path(directory)), \
                 patch.object(core, 'MOTION_WARMUP_SECONDS', 0), \
                 patch.object(core, 'MOTION_MIN_FREE_MB', 0), \
                 patch.object(core, 'open_rtsp_capture', return_value=cap), \
                 patch.object(core, 'load_yolo_model') as model, \
                 patch.object(core, 'detect_motion', return_value=[(2, 2, 20, 20)]) as detect, \
                 patch.object(core, 'create_video_writer', return_value=writer), \
                 patch.object(core, 'emit_recording_progress', return_value=0), \
                 patch.object(core, 'write_recording_frames_until', return_value=0), \
                 patch.object(core, 'finalize_recorded_clip') as finalize, \
                 patch.object(core.time, 'monotonic', side_effect=lambda: next(ticks)):
                core.run_detection('rtsp://test', 'camera', set(), stop, upload_on_motion=False)
            model.assert_not_called()
            self.assertGreaterEqual(detect.call_count, 2)
            self.assertTrue(finalize.call_args.kwargs['upload_on_motion'])
            writer.release.assert_called_once()


    def test_shadow_receives_same_motion_decision_without_affecting_capture(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cap, writer, shadow = MagicMock(), MagicMock(), MagicMock()
        cap.get.return_value = 20
        cap.read.return_value = (True, frame)
        with tempfile.TemporaryDirectory() as directory:
            writer.path = Path(directory) / 'motion_test.mp4'
            ticks = iter(range(1000))
            stop = MagicMock(side_effect=[False, False, True])
            with patch.object(core, 'OUTPUT_DIR', Path(directory)), \
                 patch.object(core, 'MOTION_WARMUP_SECONDS', 0), \
                 patch.object(core, 'MOTION_MIN_FREE_MB', 0), \
                 patch.object(core, 'MOTION_SHADOW_ENABLED', True), \
                 patch.object(core, 'MotionShadowComparison', return_value=shadow), \
                 patch.object(core, 'open_rtsp_capture', return_value=cap), \
                 patch.object(core, 'detect_motion', return_value=[(2, 2, 20, 20)]), \
                 patch.object(core, 'create_video_writer', return_value=writer), \
                 patch.object(core, 'emit_recording_progress', return_value=0), \
                 patch.object(core, 'write_recording_frames_until', return_value=0), \
                 patch.object(core, 'finalize_recorded_clip') as finalize, \
                 patch.object(core.time, 'monotonic', side_effect=lambda: next(ticks)):
                core.run_detection('rtsp://test', 'camera', set(), stop)
            self.assertTrue(shadow.observe.call_args.args[2])
            self.assertTrue(finalize.call_args.kwargs['upload_on_motion'])

    def test_difference_controls_recording_and_failure_falls_back(self):
        for fails in (False, True):
            with self.subTest(fails=fails), tempfile.TemporaryDirectory() as directory:
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                cap, writer, candidate = MagicMock(), MagicMock(), MagicMock()
                cap.get.return_value = 20
                cap.read.return_value = (True, frame)
                writer.path = Path(directory) / 'motion_test.mp4'
                candidate.previous_time = 0
                candidate.max_gap_seconds = 1000
                candidate.observe.side_effect = RuntimeError('bad candidate') if fails else None
                candidate.observe.return_value = []
                ticks = iter(range(1000))
                stop = MagicMock(side_effect=[False, False, True])
                with patch.object(core, 'OUTPUT_DIR', Path(directory)), \
                     patch.object(core, 'MOTION_WARMUP_SECONDS', 0), \
                     patch.object(core, 'MOTION_MIN_FREE_MB', 0), \
                     patch.object(core, 'FrameDifferenceMotion') as difference_class, \
                     patch.object(core, 'open_rtsp_capture', return_value=cap), \
                     patch.object(core, 'detect_motion', return_value=[(2, 2, 20, 20)]), \
                     patch.object(core, 'create_video_writer', return_value=writer) as create, \
                     patch.object(core, 'emit_recording_progress', return_value=0), \
                     patch.object(core, 'write_recording_frames_until', return_value=0), \
                     patch.object(core, 'finalize_recorded_clip'), \
                     patch.object(core.time, 'monotonic', side_effect=lambda: next(ticks)):
                    difference_class.return_value = candidate
                    core.run_detection('rtsp://test', 'camera', set(), stop, motion_algorithm='difference')
                self.assertEqual(create.called, fails)

    def test_lowest_sensitivity_uses_higher_pixel_threshold(self):
        with patch.object(core, 'FrameDifferenceMotion') as difference:
            core.run_detection('rtsp://test', 'camera', set(), lambda: True,
                               motion_sensitivity=1, motion_algorithm='difference')
        self.assertEqual(difference.call_args.kwargs['pixel_threshold'], 20)


if __name__ == '__main__':
    unittest.main()
