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

    def test_diagnostic_association_never_creates_an_alert(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory) / 'motion.mp4'
            start_metadata(clip)
            update_metadata(clip,['car'],'frame.jpg',associations=[{'label':'car','status':'associated','matching_samples':2,'motion_samples':3}])
            finish_metadata(clip)
            data=load_metadata(clip)
            self.assertEqual(data['motion_associations'][0]['status'],'associated')
            self.assertNotIn('alert_objects',data)

    def test_crop_comparison_is_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as directory:
            clip=Path(directory)/'motion.mp4'
            start_metadata(clip)
            update_metadata(clip,[],crop_comparison={'crop_box':[0,0,100,100],'detections':[{'label':'person'}],'full_frame_labels':[]})
            finish_metadata(clip)
            result=load_metadata(clip)
            self.assertEqual(result['crop_comparisons'][0]['detections'][0]['label'],'person')
            self.assertEqual(result['objects'],[])
            self.assertNotIn('alert_objects',result)

if __name__ == '__main__': unittest.main()
