import tempfile
import unittest
from pathlib import Path
from motion_clip_metadata import start_metadata, update_metadata, finish_metadata, load_metadata, title_for_metadata


class MotionMetadataTests(unittest.TestCase):
    def test_old_and_unchecked_clips_are_not_reported_as_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory) / 'motion.mp4'
            self.assertEqual(title_for_metadata(load_metadata(clip)), 'Object detection not recorded')
            start_metadata(clip)
            self.assertEqual(title_for_metadata(load_metadata(clip)), 'Recording; object detection in progress')
            finish_metadata(clip)
            self.assertEqual(title_for_metadata(load_metadata(clip)), 'Object detection not checked')

    def test_samples_and_detections_persist_through_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory) / 'motion.mp4'
            start_metadata(clip)
            update_metadata(clip, [])
            update_metadata(clip, ['person', 'car'], 'detection_1.jpg')
            update_metadata(clip, ['person'], 'detection_2.jpg')
            finish_metadata(clip)
            result = load_metadata(clip)
            self.assertEqual(result['samples'], 3)
            self.assertEqual(title_for_metadata(result), 'Object detected: car, person')
            self.assertEqual(result['evidence'], ['detection_1.jpg', 'detection_2.jpg'])
            update_metadata(clip, ['truck'])
            self.assertEqual(load_metadata(clip)['objects'], ['car', 'person'])

    def test_negative_means_sampled_frames_only(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory) / 'motion.mp4'
            start_metadata(clip); update_metadata(clip, []); finish_metadata(clip)
            self.assertEqual(title_for_metadata(load_metadata(clip)), 'No objects detected in sampled frames')

    def test_missing_evidence_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory) / 'motion.mp4'
            start_metadata(clip); update_metadata(clip, ['car']); finish_metadata(clip)
            self.assertIn('images unavailable', title_for_metadata(load_metadata(clip)))

if __name__ == '__main__': unittest.main()
