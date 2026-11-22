import tempfile
import unittest
from pathlib import Path
from alert_outbox import AlertOutbox

class AlertOutboxTests(unittest.TestCase):
    def test_pending_survives_restart_and_interrupted_send_becomes_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'alerts.db'
            queue = AlertOutbox(path)
            queue.enqueue('/tmp/example-clip.mp4', '701', ['person'])
            queue = AlertOutbox(path)
            self.assertEqual(queue.claim()[1:], ('701', ['person']))
            queue = AlertOutbox(path)
            self.assertEqual(queue.status('/tmp/example-clip.mp4'), 'unknown')
            self.assertIsNone(queue.claim())

    def test_deduplicates_clip_and_records_provider_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = AlertOutbox(Path(directory) / 'alerts.db')
            queue.enqueue('/tmp/a.mp4', '701', ['person'])
            queue.enqueue('/tmp/a.mp4', '701', ['person'])
            self.assertIsNotNone(queue.claim())
            queue.finish('/tmp/a.mp4', True)
            self.assertEqual(queue.counts(), {'accepted': 1})
            self.assertIsNone(queue.claim())
