import json
import tempfile
import unittest
from pathlib import Path
from archive_manager import ArchiveManager
from motion_clip_metadata import start_metadata, finish_metadata, update_metadata

class ArchiveManagerTests(unittest.TestCase):
    def test_active_clips_are_not_archived_and_interrupted_are_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / 'motion_one.mp4'
            clip.write_bytes(b'video')
            start_metadata(clip)
            manager = ArchiveManager(root/'queue.db', {'motion':root}, transport=lambda *args: True)
            manager.scan()
            self.assertIsNone(manager.outbox.process_one(manager.transport))
            manager.recover_interrupted()
            manager.scan()
            self.assertEqual(manager.outbox.process_one(manager.transport), 'verified')
            self.assertFalse(clip.exists())
            self.assertFalse(Path(str(clip)+'.detections.json').exists())

    def test_alert_clip_is_retained_until_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root/'object_one.mp4'
            clip.write_bytes(b'video')
            start_metadata(clip)
            update_metadata(clip, ['person'], alerts=[{'label':'person'}])
            finish_metadata(clip)
            status = ['pending']
            manager = ArchiveManager(root/'queue.db', {'object':root}, transport=lambda *args: True,
                                     alert_status=lambda _path: status[0])
            manager.scan()
            self.assertEqual(manager.outbox.process_one(manager.transport), 'verified')
            self.assertTrue(clip.exists())
            status[0] = 'accepted'
            manager.scan()
            self.assertEqual(manager.outbox.process_one(manager.transport), 'verified')
            self.assertFalse(clip.exists())
