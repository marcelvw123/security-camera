import tempfile
import unittest
from pathlib import Path
from archive_outbox import ArchiveOutbox


class ArchiveOutboxTests(unittest.TestCase):
    def test_retry_survives_restart_and_deletes_only_after_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / 'clip.mp4'
            clip.write_bytes(b'video')
            queue = ArchiveOutbox(root / 'queue.db')
            job = queue.enqueue(clip, 'motion', {'camera': '701'})
            self.assertEqual(queue.process_one(lambda *args: False, now=0), 'retry')
            self.assertTrue(clip.exists())
            queue = ArchiveOutbox(root / 'queue.db')
            self.assertEqual(queue.enqueue(clip, 'motion', {'camera': '701'}), job)
            self.assertEqual(queue.process_one(lambda *args: True, now=11), 'verified')
            self.assertFalse(clip.exists())

    def test_changed_file_and_unverified_never_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / 'clip.mp4'
            clip.write_bytes(b'video')
            queue = ArchiveOutbox(root / 'queue.db')
            queue.enqueue(clip, 'object', {})
            def changed(*args):
                clip.write_bytes(b'other video')
                return True
            self.assertEqual(queue.process_one(changed, now=0), 'retry')
            self.assertEqual(clip.read_bytes(), b'other video')

    def test_pending_alert_can_retain_verified_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'image.jpg'
            image.write_bytes(b'image')
            queue = ArchiveOutbox(root / 'queue.db')
            queue.enqueue(image, 'image', {}, retain_local=True)
            self.assertEqual(queue.process_one(lambda *args: True, now=0), 'verified')
            self.assertTrue(image.exists())

    def test_sidecar_preserved_if_changed_and_deleted_after_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / 'clip.mp4'
            sidecar = Path(str(clip) + '.detections.json')
            clip.write_bytes(b'video')
            sidecar.write_text('{"recording": false}')
            queue = ArchiveOutbox(root / 'queue.db')
            queue.enqueue(clip, 'motion', {'detections': {'recording': False}})
            def changed(*args):
                sidecar.write_text('{"recording": false, "objects": ["person"]}')
                return True
            self.assertEqual(queue.process_one(changed, now=0), 'retry')
            self.assertTrue(clip.exists())
            self.assertTrue(sidecar.exists())

    def test_recent_media_is_processed_before_backlog(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, recent = root/'old.mp4', root/'recent.mp4'
            old.write_bytes(b'old'); recent.write_bytes(b'new')
            os.utime(old, (1, 1)); os.utime(recent, (100, 100))
            queue = ArchiveOutbox(root/'queue.db')
            queue.enqueue(old, 'motion', {}); queue.enqueue(recent, 'motion', {})
            seen = []
            queue.process_one(lambda _id, path, *_: seen.append(path.name) or True, now=0)
            self.assertEqual(seen, ['recent.mp4'])
