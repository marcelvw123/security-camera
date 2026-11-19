import unittest
import threading
from unittest.mock import MagicMock

import numpy as np
from motion_preroll import AsyncPreRollWriter, MotionPreRoll


class PreRollTests(unittest.TestCase):
    def test_frame_order_duration_and_trigger_excluded(self):
        buffer = MotionPreRoll(seconds=2, sample_fps=2, max_bytes=1000000)
        for t, value in [(0, 10), (.5, 40), (1, 80), (1.5, 120), (2, 160)]:
            buffer.add(t, np.full((30, 30, 3), value, dtype=np.uint8))
        writer = MagicMock()
        duration = buffer.write_before(writer, 2, 10)
        self.assertEqual(duration, 2)
        self.assertEqual(writer.write.call_count, 20)
        levels = [int(call.args[0].mean()) for call in writer.write.call_args_list]
        self.assertLess(levels[0], levels[5])
        self.assertLess(levels[5], levels[10])
        self.assertLess(levels[10], levels[15])
        self.assertLess(max(levels), 150)

    def test_buffer_is_bounded_and_clear_resets_it(self):
        frame = np.zeros((60, 60, 3), dtype=np.uint8)
        buffer = MotionPreRoll(seconds=2, sample_fps=4, max_bytes=2000)
        for i in range(40): buffer.add(i / 4, frame)
        self.assertLessEqual(buffer.total_bytes, 2000)
        self.assertTrue(all(t >= 9.75-2 for t, _ in buffer.frames))
        buffer.clear()
        self.assertEqual(buffer.total_bytes, 0)
        self.assertFalse(buffer.frames)
        self.assertIsNone(buffer.next_sample)

    def test_disabled_buffer_does_not_write(self):
        buffer = MotionPreRoll(seconds=0)
        buffer.add(1, np.zeros((5, 5, 3), dtype=np.uint8))
        writer = MagicMock()
        self.assertEqual(buffer.write_before(writer, 2, 10), 0)
        writer.write.assert_not_called()


class AsyncWriterTests(unittest.TestCase):
    def test_capture_can_queue_live_frame_while_preroll_is_written(self):
        started, unblock = threading.Event(), threading.Event()
        target = MagicMock()
        target.path = '/tmp/test.mp4'
        values = []
        def write(frame):
            values.append(int(frame.mean()))
            if len(values) == 1:
                started.set()
                self.assertTrue(unblock.wait(2))
        target.write.side_effect = write
        preroll = MotionPreRoll(seconds=1, sample_fps=1)
        preroll.add(0, np.full((10, 10, 3), 20, np.uint8))
        writer = AsyncPreRollWriter(target, preroll, 1, 1)
        self.assertTrue(started.wait(2))
        writer.write(np.full((10, 10, 3), 200, np.uint8))
        self.assertEqual(len(values), 1)
        unblock.set()
        writer.release()
        self.assertEqual(values, [20, 200])
        target.release.assert_called_once()

    def test_writer_failure_is_propagated_and_target_released(self):
        target = MagicMock()
        target.path = '/tmp/test.mp4'
        target.write.side_effect = OSError('write failed')
        preroll = MotionPreRoll(seconds=0)
        writer = AsyncPreRollWriter(target, preroll, 1, 1)
        writer.write(np.zeros((10, 10, 3), np.uint8))
        with self.assertRaises(RuntimeError): writer.release()
        target.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
