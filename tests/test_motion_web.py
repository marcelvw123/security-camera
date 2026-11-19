import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from http import HTTPStatus
import webapp


class MotionWebTests(unittest.TestCase):
    def test_vehicle_area_is_saved_per_camera_and_applied_to_worker(self):
        worker = webapp.WorkerState('camera:1', 'Camera 1', MagicMock(), MagicMock(), 'camera', vehicle_area=[0,0,1,1])
        handler = MagicMock()
        handler.read_json.return_value = {'source_id': 'camera:1', 'vehicle_area': [.1,.2,.3,.4]}
        with tempfile.TemporaryDirectory() as directory, patch.object(webapp, 'VEHICLE_AREAS_FILE', Path(directory)/'areas.json'), \
             patch.object(webapp, 'APP_STATE') as state, patch.object(webapp, 'OBJECT_DETECTOR', None):
            state.get_worker.return_value = worker
            webapp.SecurityCameraWebHandler.handle_stream_vehicle_area(handler)
            self.assertEqual(webapp.load_vehicle_areas()['camera:1'], [.1,.2,.3,.4])
            self.assertEqual(worker.current_vehicle_area(), [.1,.2,.3,.4])
            handler.read_json.return_value['vehicle_area'] = [.9, 0, .3, .2]
            webapp.SecurityCameraWebHandler.handle_stream_vehicle_area(handler)
            self.assertEqual(handler.send_json.call_args.args[1], HTTPStatus.BAD_REQUEST)
            self.assertEqual(worker.current_vehicle_area(), [.1,.2,.3,.4])

    def test_new_worker_loads_vehicle_area(self):
        handler = MagicMock()
        handler.read_json.return_value = {'stream': {'stream_id': 'camera:1'}}
        with tempfile.TemporaryDirectory() as directory, patch.object(webapp, 'VEHICLE_AREAS_FILE', Path(directory)/'areas.json'):
            webapp.save_vehicle_area('camera:1', [.1,.2,.3,.4])
            with patch.object(webapp, 'APP_STATE'), patch.object(webapp.threading, 'Thread') as thread, patch.object(webapp, 'OBJECT_DETECTOR', None):
                webapp.SecurityCameraWebHandler.handle_stream_start(handler)
            self.assertEqual(thread.call_args.kwargs['args'][0].current_vehicle_area(), [.1,.2,.3,.4])

    def test_media_list_displays_detection_summary_and_unknown_for_old_clips(self):
        from motion_clip_metadata import start_metadata, update_metadata, finish_metadata
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root / 'motion_old.mp4', root / 'motion_new.mp4'
            old.touch(); new.touch()
            start_metadata(new); update_metadata(new, ['person'], 'evidence.jpg'); finish_metadata(new)
            (root / 'detection_frames').mkdir()
            (root / 'detection_frames' / 'evidence.jpg').touch()
            folders = {'clips': {'path': root, 'extensions': {'.mp4'}, 'type': 'video'}}
            with patch.object(webapp, 'MEDIA_FOLDERS', folders), patch.object(webapp, 'OBJECT_OUTPUT_DIR', root):
                items = {item['name']: item for item in webapp.list_saved_media('clips')}
            self.assertEqual(items[new.name]['detection_title'], 'Object detected: person')
            self.assertEqual(items[old.name]['detection_title'], 'Object detection not recorded')
            self.assertEqual(items[new.name]['evidence'][0]['url'], '/media/detection_frames/evidence.jpg')

    def test_worker_summary_reports_active_recording(self):
        state = webapp.WebAppState()
        worker = webapp.WorkerState('camera:1', 'Camera 1', MagicMock(), MagicMock(), 'camera')
        state.add_worker(worker)
        worker.sensitivity = 3
        self.assertEqual(state.worker_summaries()[0]['sensitivity'], 3)
        worker.status = 'Recording clip: motion_example.mp4'
        self.assertTrue(state.worker_summaries()[0]['recording'])
        worker.status = 'Motion clip saved: motion_example.mp4'
        self.assertFalse(state.worker_summaries()[0]['recording'])

    def test_recent_detected_objects_are_per_camera_and_expire(self):
        state = webapp.WebAppState()
        first = webapp.WorkerState('camera:1', 'Camera 1', MagicMock(), MagicMock(), 'camera')
        second = webapp.WorkerState('camera:2', 'Camera 2', MagicMock(), MagicMock(), 'camera')
        state.add_worker(first); state.add_worker(second)
        with patch.object(webapp, 'OBJECT_DETECTOR', None), patch.object(webapp.time, 'monotonic', return_value=100):
            first.record_objects(['person', 'car', 'car'])
            summaries = state.worker_summaries()
            self.assertEqual(summaries[0]['detected_objects'], ['car', 'person'])
            self.assertEqual(summaries[1]['detected_objects'], [])
        with patch.object(webapp, 'OBJECT_DETECTOR', None), patch.object(webapp.time, 'monotonic', return_value=109):
            self.assertEqual(state.worker_summaries()[0]['detected_objects'], [])

    def test_camera_settings_reach_worker(self):
        handler=MagicMock()
        handler.read_json.return_value={'stream':{'stream_id':'camera:1','label':'Camera 1'},
            'sensitivity':4,'motion_algorithm':'difference','excluded_areas':[[.1,.2,.3,.4]]}
        with patch.object(webapp, 'APP_STATE'), patch.object(webapp.threading, 'Thread') as thread:
            webapp.SecurityCameraWebHandler.handle_stream_start(handler)
        args=thread.call_args.kwargs['args']
        self.assertEqual(args[-2],[[.1,.2,.3,.4]])
        self.assertEqual(args[-1],'difference')
        thread.return_value.start.assert_called_once()

    def test_invalid_exclusion_does_not_start_worker(self):
        handler=MagicMock()
        handler.read_json.return_value={'stream':{'stream_id':'camera:1'},'excluded_areas':[[.9,0,.3,.5]]}
        with patch.object(webapp.threading, 'Thread') as thread:
            webapp.SecurityCameraWebHandler.handle_stream_start(handler)
        thread.assert_not_called()
        self.assertEqual(handler.send_json.call_args.args[1],HTTPStatus.BAD_REQUEST)

    def test_saved_camera_areas_survive_new_worker_and_can_clear(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(webapp, 'EXCLUSIONS_FILE', Path(directory)/'areas.json'):
            worker=webapp.WorkerState('camera:1','Camera 1',MagicMock(),MagicMock(),'camera')
            handler=MagicMock()
            handler.read_json.return_value={'source_id':'camera:1','excluded_areas':[[.1,.2,.3,.4]]}
            with patch.object(webapp, 'APP_STATE') as state:
                state.get_worker.return_value=worker
                webapp.SecurityCameraWebHandler.handle_stream_exclusions(handler)
                self.assertEqual(worker.current_exclusions(),[[.1,.2,.3,.4]])
                self.assertEqual(webapp.load_exclusion_map()['camera:1'],[[.1,.2,.3,.4]])
                handler.read_json.return_value={'source_id':'camera:1','excluded_areas':[]}
                webapp.SecurityCameraWebHandler.handle_stream_exclusions(handler)
            self.assertEqual(worker.current_exclusions(),[])
            self.assertEqual(webapp.load_exclusion_map()['camera:1'],[])

    def test_failed_save_does_not_change_live_areas(self):
        worker=webapp.WorkerState('camera:1','Camera 1',MagicMock(),MagicMock(),'camera',excluded_areas=[[0,0,.1,.1]])
        handler=MagicMock()
        handler.read_json.return_value={'source_id':'camera:1','excluded_areas':[[.5,.5,.2,.2]]}
        with patch.object(webapp, 'APP_STATE') as state, patch.object(webapp, 'save_camera_exclusions',side_effect=OSError('disk')):
            state.get_worker.return_value=worker
            webapp.SecurityCameraWebHandler.handle_stream_exclusions(handler)
        self.assertEqual(worker.current_exclusions(),[[0,0,.1,.1]])
        self.assertEqual(handler.send_json.call_args.args[1],HTTPStatus.BAD_REQUEST)

    def test_new_worker_loads_saved_exclusions(self):
        handler=MagicMock()
        handler.read_json.return_value={'stream':{'stream_id':'camera:1'}}
        with tempfile.TemporaryDirectory() as directory, patch.object(webapp, 'EXCLUSIONS_FILE',Path(directory)/'areas.json'):
            webapp.save_camera_exclusions('camera:1',[[.1,.1,.2,.2]])
            with patch.object(webapp, 'APP_STATE'), patch.object(webapp.threading, 'Thread') as thread:
                webapp.SecurityCameraWebHandler.handle_stream_start(handler)
            self.assertEqual(thread.call_args.kwargs['args'][-2],[[.1,.1,.2,.2]])


if __name__=='__main__':
    unittest.main()
