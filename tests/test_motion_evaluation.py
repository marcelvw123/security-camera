import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from scripts.evaluate_motion import evaluate, score_events
from scripts.evaluate_difference import evaluate as evaluate_difference


class EventScoringTests(unittest.TestCase):
    def setUp(self):
        self.result = {'duration_seconds': 10, 'samples': [
            {'seconds': 0, 'warmup': True, 'boxes': [[0, 0, 100, 100]]},
            {'seconds': 2, 'warmup': False, 'boxes': [[10, 10, 20, 20]]},
            {'seconds': 4, 'warmup': False, 'boxes': [[100, 100, 10, 10]]},
            {'seconds': 6, 'warmup': False, 'boxes': []},
        ]}

    def test_hit_miss_and_unsampled_are_distinct(self):
        annotations = {'events': [
            {'start': 1, 'end': 3, 'roi': [0, 0, 40, 40]},
            {'start': 3, 'end': 5, 'roi': [0, 0, 40, 40]},
            {'start': 0, 'end': 1, 'roi': [0, 0, 40, 40]},
            {'start': 7, 'end': 8, 'roi': [0, 0, 40, 40]},
        ]}
        score = score_events(self.result, annotations)
        self.assertEqual(score['counts'], {'hit': 1, 'miss': 1, 'unsampled': 2})
        self.assertEqual(score['events'][0]['latency_seconds'], 1)

    def test_event_end_is_exclusive_and_edge_touch_is_not_overlap(self):
        annotations = {'events': [
            {'start': 1, 'end': 2, 'roi': [0, 0, 100, 100]},
            {'start': 2, 'end': 3, 'roi': [30, 10, 10, 10]},
        ]}
        self.assertEqual([e['status'] for e in score_events(self.result, annotations)['events']],
                         ['unsampled', 'miss'])

    def test_rejects_invalid_intervals_and_regions(self):
        for event in ({'start': 3, 'end': 2, 'roi': [0, 0, 1, 1]},
                      {'start': 0, 'end': 11, 'roi': [0, 0, 1, 1]},
                      {'start': 0, 'end': 1, 'roi': [0, 0, 0, 1]},
                      {'start': 0, 'end': float('nan'), 'roi': [0, 0, 1, 1]}):
            with self.subTest(event=event), self.assertRaises(ValueError):
                score_events(self.result, {'events': [event]})


class ReplayTests(unittest.TestCase):
    def test_difference_replay_applies_camera_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (320, 240))
            if not writer.isOpened():
                self.skipTest('MJPEG video writer unavailable')
            for index in range(20):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                if index >= 10:
                    cv2.rectangle(frame, (20, 50), (100, 150), (255, 255, 255), -1)
                writer.write(frame)
            writer.release()
            unmasked = evaluate_difference(path, interval=.1, min_area=100, warmup=0, pixel_threshold=15)
            masked = evaluate_difference(path, interval=.1, min_area=100, warmup=0, pixel_threshold=15,
                                         exclusions=[[0, 0, .5, 1]])
            self.assertTrue(any(sample['boxes'] for sample in unmasked['samples']))
            self.assertFalse(any(sample['boxes'] for sample in masked['samples']))

    def test_continuous_video_detects_motion_after_stable_background(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (320, 240))
            if not writer.isOpened():
                self.skipTest('MJPEG video writer unavailable')
            try:
                for index in range(100):
                    frame = np.zeros((240, 320, 3), dtype=np.uint8)
                    if 50 <= index < 80:
                        x = 30 + (index - 50) * 4
                        cv2.rectangle(frame, (x, 80), (x + 50, 140), (255, 255, 255), -1)
                    writer.write(frame)
            finally:
                writer.release()
            result = evaluate(path, .2, 5, 2)
            score = score_events(result, {'events': [
                {'start': 5, 'end': 8, 'roi': [20, 70, 200, 90]},
                {'start': 2, 'end': 5, 'roi': [20, 70, 200, 90]},
            ]})
            self.assertEqual(len(result['samples']), 50)
            self.assertEqual([event['status'] for event in score['events']], ['hit', 'miss'])


if __name__ == '__main__':
    unittest.main()
