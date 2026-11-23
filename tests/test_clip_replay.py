import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from clip_replay import ClipReplay
import security_camera_core as core


class ClipReplayTests(unittest.TestCase):
    def test_recording_time_and_camera_settings_are_used_without_alerts(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ClipReplay(directory)
            worker = MagicMock(label='Front', sensitivity=2)
            worker.current_exclusions.return_value = [[0,0,.2,.2]]
            detector = MagicMock()
            detector.remote_shadow = None
            path, recorded_at = manager.reserve(worker, '2026-11-20T22:15:00', '.mp4')
            self.assertEqual(recorded_at.utcoffset().total_seconds(), 7200)
            path.write_bytes(b'video')
            with patch('clip_replay.run_detection') as run:
                manager.start(path, recorded_at, worker, detector, 'difference')
                for _ in range(100):
                    if manager.snapshot()['status'] == 'complete': break
                    time.sleep(.01)
                self.assertEqual(manager.snapshot()['status'], 'complete')
                self.assertEqual(run.call_args.kwargs['motion_sensitivity'], 2)
                self.assertEqual(run.call_args.kwargs['excluded_areas'], [[0,0,.2,.2]])
                self.assertEqual(run.call_args.kwargs['replay']['recorded_at'], recorded_at)

    def test_records_unsuccessful_object_checks_for_diagnosis(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ClipReplay(directory)
            worker = MagicMock(label='Front'); detector = MagicMock(); detector.remote_shadow = None
            path, recorded_at = manager.reserve(worker, '2026-11-20T22:15', '.mp4')
            path.write_bytes(b'video')
            def replay(*args, **kwargs):
                kwargs['replay_check_callback']([(1, [], None, [], ['motion crop did not confirm a moving object'], 'full', [], {'captured_at':'2026-11-20T22:15:01+02:00'})])
            with patch('clip_replay.run_detection', side_effect=replay):
                manager.start(path, recorded_at, worker, detector, 'difference')
                manager.thread.join(timeout=2)
            job=manager.snapshot()
            self.assertEqual(job['object_checks'],1)
            self.assertEqual(job['events'][0]['decisions'], ['motion crop did not confirm a moving object'])

    def test_persists_and_recovers_completed_job(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ClipReplay(directory)
            worker = MagicMock(label='Front')
            path, recorded_at = manager.reserve(worker, '2026-11-20T22:15', '.mp4')
            path.write_bytes(b'video')
            detector = MagicMock(); detector.remote_shadow = None
            with patch('clip_replay.run_detection'):
                manager.start(path, recorded_at, worker, detector, 'difference')
                manager.thread.join(timeout=2)
            restored = ClipReplay(directory).snapshot()
            self.assertEqual(restored['status'], 'complete')
            self.assertEqual(restored['recorded_at'], '2026-11-20T22:15:00+02:00')

    def test_replay_object_recording_never_queues_whatsapp(self):
        from motion_clip_metadata import start_metadata, update_metadata
        from object_detection import ObjectClipRecorder
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'object.mp4'
            start_metadata(path)
            update_metadata(path, ['person'], alerts=[{'label':'person'}])
            recorder = ObjectClipRecorder(MagicMock(), 'camera', directory, MagicMock(), MagicMock(),
                                          MagicMock(), 0, send_alerts=False)
            recorder.writer = MagicMock(); recorder.writer.path = path
            with patch.dict('os.environ', {'WHATSAPP_OBJECT_ALERTS_ENABLED':'true'}), \
                 patch('object_alerts.queue_object_alert') as send:
                recorder.close()
                send.assert_not_called()

    def test_validation_and_single_running_job(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ClipReplay(directory)
            worker = MagicMock(label='Front')
            with self.assertRaises(ValueError): manager.reserve(worker, 'bad time', '.mp4')
            with self.assertRaises(ValueError): manager.reserve(worker, '2026-11-20T22:00', '.exe')
            manager.reserve(worker, '2026-11-20T22:00', '.mp4')
            with self.assertRaises(ValueError): manager.reserve(worker, '2026-11-20T22:00', '.mp4')

    def test_video_replay_uses_live_loop_with_alerts_disabled_and_clip_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); video = root/'test.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64,64))
            self.assertTrue(writer.isOpened())
            for _ in range(8): writer.write(np.zeros((64,64,3), dtype=np.uint8))
            writer.release()
            recorder = MagicMock(); recorder.observe.return_value=[]
            detector = MagicMock(); detector.has_pending.return_value=False
            callbacks=[]
            with patch('object_detection.ObjectClipRecorder', return_value=recorder) as cls, \
                 patch.object(core, 'MOTION_WARMUP_SECONDS', 0), \
                 patch.object(core, 'MOTION_MIN_FREE_MB', 0), \
                 patch.object(core, 'detect_motion', return_value=[(2,2,20,20)]), \
                 patch.object(core, 'create_video_writer') as video_writer:
                video_writer.return_value.path = root/'motion'/'result.mp4'
                core.run_detection(str(video), 'test', set(), lambda:False, motion_algorithm='background',
                    object_detector=detector, object_source_id='replay:test',
                    replay={'path':video, 'recorded_at':datetime.fromisoformat('2026-11-20T22:15:00+02:00'),
                            'motion_dir':root/'motion', 'object_dir':root/'object'},
                    replay_check_callback=callbacks.append)
            self.assertFalse(cls.call_args.kwargs['send_alerts'])
            observed=[call.kwargs.get('captured_at') for call in recorder.observe.call_args_list if call.kwargs.get('captured_at')]
            self.assertTrue(observed)
            self.assertTrue(all(value.startswith('2026-11-20T22:15:00') for value in observed))
            self.assertTrue(any(isinstance(group[0],dict) and 'motion_regions' in group[0] for group in callbacks))

class ClipTestUploadTests(unittest.TestCase):
    def test_upload_streams_bytes_and_starts_with_selected_camera(self):
        import io
        from urllib.parse import urlparse
        import webapp
        with tempfile.TemporaryDirectory() as directory:
            manager = ClipReplay(directory)
            worker = MagicMock(label='Front', sensitivity=3, motion_algorithm='difference')
            detector = MagicMock(); detector.health.return_value = 'ready'; detector.remote_shadow = None
            handler = MagicMock(); handler.headers={'Content-Length':'5'}; handler.rfile=io.BytesIO(b'video')
            with patch.object(webapp,'CLIP_REPLAY',manager), patch.object(webapp,'OBJECT_DETECTOR',detector), \
                 patch.object(webapp.APP_STATE,'get_worker',return_value=worker), \
                 patch('shutil.disk_usage',return_value=MagicMock(free=10_000_000_000)), \
                 patch.object(manager,'start') as start:
                webapp.SecurityCameraWebHandler.handle_clip_test(handler, urlparse('/api/clip-test?camera=x&filename=clip.mp4&recorded_at=2026-11-20T22%3A15'))
                path=start.call_args.args[0]
                self.assertEqual(path.read_bytes(),b'video')
                self.assertEqual(start.call_args.args[2],worker)
                self.assertEqual(handler.send_json.call_args.args[1],202)

    def test_upload_uses_neutral_defaults_without_camera_selection(self):
        import io
        from urllib.parse import urlparse
        import webapp
        with tempfile.TemporaryDirectory() as directory:
            manager=ClipReplay(directory)
            detector=MagicMock();detector.health.return_value='ready';detector.remote_shadow=None
            handler=MagicMock();handler.headers={'Content-Length':'5'};handler.rfile=io.BytesIO(b'video')
            with patch.object(webapp,'CLIP_REPLAY',manager), patch.object(webapp,'OBJECT_DETECTOR',detector), \
                 patch.object(webapp.APP_STATE,'get_worker') as get_worker, \
                 patch('shutil.disk_usage',return_value=MagicMock(free=10_000_000_000)), \
                 patch.object(manager,'start') as start:
                webapp.SecurityCameraWebHandler.handle_clip_test(handler,urlparse('/api/clip-test?filename=clip.mp4&recorded_at=2026-11-20T22%3A15'))
                get_worker.assert_not_called()
                worker=start.call_args.args[2]
                self.assertEqual(worker.sensitivity,5)
                self.assertEqual(worker.motion_algorithm,'difference')
                self.assertEqual(worker.current_exclusions(),[])
                self.assertEqual(worker.current_vehicle_area(),[0,0,1,1])
                self.assertEqual(handler.send_json.call_args.args[1],202)

    def test_rejects_large_or_unknown_file_without_starting_job(self):
        from urllib.parse import urlparse
        import webapp
        handler=MagicMock(); handler.headers={'Content-Length':str(251*1024*1024)}
        detector=MagicMock(); detector.health.return_value='ready'
        with patch.object(webapp,'OBJECT_DETECTOR',detector), patch.object(webapp.APP_STATE,'get_worker',return_value=MagicMock()):
            webapp.SecurityCameraWebHandler.handle_clip_test(handler,urlparse('/api/clip-test?camera=x&filename=clip.mp4'))
        self.assertEqual(handler.send_json.call_args.args[1],400)

class StageSelectionTests(unittest.TestCase):
    def test_image_motion_rejected_and_valid_stages_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            manager=ClipReplay(directory); worker=MagicMock(label='Camera')
            with self.assertRaisesRegex(ValueError, 'Motion detection needs'):
                manager.reserve(worker, '2026-11-20T22:00', '.jpg', 'motion')
            source, _ = manager.reserve(worker, '2026-11-20T22:00', '.jpg', 'scenario')
            self.assertEqual(source.suffix,'.jpg')
            self.assertEqual(manager.snapshot()['stage'],'scenario')

    def test_full_frame_scenario_uses_person_verification(self):
        from object_detection import process_scene_frame
        from types import SimpleNamespace
        model=MagicMock(); model.names={0:'person'}
        model.predict.return_value=[SimpleNamespace(boxes=SimpleNamespace(
            xyxy=[[0,0,20,40],[30,0,50,40]],conf=[.9,.8],cls=[0,0]))]
        verified=[{'label':'person','box':[0.,0.,20.,40.],'confidence':.9,'verified':True,'verification_box':[0.,0.,20.,40.]},
                  {'label':'person','box':[30.,0.,50.,40.],'confidence':.8,'verified':True,'verification_box':[30.,0.,50.,40.]}]
        with patch('person_verifier.verify_scene_people',return_value=verified):
            detections, details=process_scene_frame(model,np.zeros((64,64,3),dtype=np.uint8),'2026-11-20T14:00:00+02:00',MagicMock())
        self.assertEqual(len(detections),2)
        self.assertEqual(details['scenarios'][0]['scenario'],'two_or_more_people')
        self.assertTrue(details['motion_skipped'])

    def test_image_scene_test_outputs_annotated_frame_without_motion_or_whatsapp(self):
        class Detector:
            remote_shadow=None
            def __init__(self): self.result=None
            def submit_scene(self, camera, frame, timestamp, captured_at):
                details={'captured_at':captured_at,'scenarios':[{'scenario':'one_person_overnight','reason':'1 person detected between 21:00 and 05:00'}]}
                self.result=(timestamp,frame.shape[1],[('person',[2,2,20,40],.9)],frame,details);return True
            def take(self,camera): result,self.result=self.result,None;return result
            def has_pending(self,camera):return self.result is not None
            def remove(self,camera):pass
        with tempfile.TemporaryDirectory() as directory:
            manager=ClipReplay(directory);worker=MagicMock(label='Camera');detector=Detector()
            path,recorded=manager.reserve(worker,'2026-11-20T22:15','.jpg','object')
            self.assertTrue(cv2.imwrite(str(path),np.zeros((64,64,3),dtype=np.uint8)))
            with patch('clip_replay.run_detection') as motion, patch('object_alerts.queue_object_alert') as notify, \
                 patch('shutil.disk_usage',return_value=MagicMock(free=10_000_000_000)):
                manager.start(path,recorded,worker,detector,'difference');manager.thread.join(timeout=3)
                motion.assert_not_called();notify.assert_not_called()
            result=manager.snapshot()
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['motion_checks'],0)
            self.assertEqual(result['object_checks'],1)
            self.assertEqual(result['events'][0]['scenarios'][0]['scenario'],'one_person_overnight')
            self.assertTrue(any(f['kind']=='image' for f in result['files']))
