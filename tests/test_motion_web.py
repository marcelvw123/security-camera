import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from http import HTTPStatus
import webapp


class MotionWebTests(unittest.TestCase):
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
