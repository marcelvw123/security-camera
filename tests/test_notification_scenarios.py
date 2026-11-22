import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
from notification_scenarios import count_people, crop_person_supported, evaluate_person_scenario
from object_detection import ObjectClipRecorder


def person(box):
    return {'label':'person','box':box,'confidence':.8}


class NotificationScenarioTests(unittest.TestCase):
    def test_one_person_at_overnight_boundaries(self):
        detections=[person([0,0,10,20])]
        for time,expected in [('2026-11-20T20:59:59+02:00',False),('2026-11-20T21:00:00+02:00',True),
                              ('2026-11-21T04:59:59+02:00',True),('2026-11-21T05:00:00+02:00',False)]:
            self.assertEqual(bool(evaluate_person_scenario(detections,time)),expected,time)
        self.assertTrue(evaluate_person_scenario(detections,'2026-11-20T19:00:00+00:00'))

    def test_two_people_any_time_and_no_vehicle_only_alert(self):
        two=[person([0,0,10,20]),person([20,0,30,20])]
        for time in ('2026-11-20T12:00:00+02:00','2026-11-20T22:00:00+02:00'):
            self.assertEqual(evaluate_person_scenario(two,time)['scenario'],'two_or_more_people')
            self.assertIsNone(evaluate_person_scenario([{'label':'car','box':[0,0,5,5]}],time))

    def test_duplicate_crop_does_not_count_as_second_person(self):
        detections=[person([0,0,20,40]),person([1,2,18,38])]
        self.assertEqual(count_people(detections),1)
        self.assertIsNone(evaluate_person_scenario(detections,'2026-11-20T12:00:00+02:00'))

    def test_small_crop_needs_full_frame_corroboration(self):
        bird=person([540,440,548,458]);full=person([539,438,550,460])
        self.assertFalse(crop_person_supported(bird,[],720))
        self.assertTrue(crop_person_supported(bird,[full],720))
        self.assertTrue(crop_person_supported(person([100,100,125,150]),[],720))

    def test_daytime_two_person_alert_requires_two_full_frame_people(self):
        frame=np.zeros((100,100,3),np.uint8);detector=MagicMock()
        detector.take.return_value=(1,100,[('person',[5,5,20,50],.9),('person',[25,5,40,50],.8)],frame,
          {'moving_gate':[{'label':'person','qualified':True}], 'full_frame_ran':True,
           'captured_at':'2026-11-20T12:00:00+02:00', 'full_frame_detections':[person([5,5,20,50])]})
        with tempfile.TemporaryDirectory() as folder, patch('shutil.disk_usage',return_value=SimpleNamespace(free=1000)), patch('motion_preroll.AsyncPreRollWriter') as writer:
            writer.return_value.path=Path(folder)/'object.mp4'
            recorder=ObjectClipRecorder(detector,'camera',folder,MagicMock(),MagicMock(),MagicMock(),0)
            checks=recorder.observe(frame,1,True,[(5,5,15,45)])
            self.assertEqual(checks[0][3],[])
            recorder.stop()

    def test_unavailable_group_verifier_suppresses_group_alert(self):
        frame=np.zeros((100,100,3),np.uint8);detector=MagicMock()
        people=[person([5,5,20,50]),person([25,5,40,50])]
        detector.take.return_value=(1,100,[('person',p['box'],p['confidence']) for p in people],frame,
          {'moving_gate':[{'label':'person','qualified':True}], 'full_frame_ran':True,
           'captured_at':'2026-11-20T12:00:00+02:00', 'full_frame_detections':people,
           'person_verification_error':'person verifier unavailable'})
        with tempfile.TemporaryDirectory() as folder, patch('shutil.disk_usage',return_value=SimpleNamespace(free=1000)), patch('motion_preroll.AsyncPreRollWriter') as writer:
            writer.return_value.path=Path(folder)/'object.mp4'
            recorder=ObjectClipRecorder(detector,'camera',folder,MagicMock(),MagicMock(),MagicMock(),0)
            checks=recorder.observe(frame,1,True,[(5,5,15,45)])
            self.assertEqual(checks[0][3],[])
            self.assertIn('person verification unavailable; group alert suppressed',checks[0][4])
            recorder.stop()

    def test_missing_or_naive_time_does_not_trigger(self):
        self.assertIsNone(evaluate_person_scenario([person([0,0,10,20])],None))
        self.assertIsNone(evaluate_person_scenario([person([0,0,10,20])],datetime(2026,11,20,23)))

    def test_scene_rules_use_people_even_when_moving_vehicle_triggered_scene(self):
        frame=np.zeros((100,100,3),np.uint8);detector=MagicMock()
        detector.take.return_value=(1,100,[('person',[5,5,20,50],.9),('person',[25,5,40,50],.8),('car',[45,10,90,50],.9)],frame,
          {'moving_gate':[{'label':'car','qualified':True}], 'full_frame_ran':True, 'captured_at':'2026-11-20T12:00:00+02:00',
           'full_frame_detections':[person([5,5,20,50]),person([25,5,40,50])],
           'verified_full_people':[person([5,5,20,50]),person([25,5,40,50])]})
        with tempfile.TemporaryDirectory() as folder, patch('shutil.disk_usage',return_value=SimpleNamespace(free=1000)), patch('motion_preroll.AsyncPreRollWriter') as writer:
            writer.return_value.path=Path(folder)/'object.mp4'
            recorder=ObjectClipRecorder(detector,'camera',folder,MagicMock(),MagicMock(),MagicMock(),0)
            checks=recorder.observe(frame,1,True,[(45,10,30,40)])
            self.assertEqual(checks[0][3][0]['scenario'],'two_or_more_people')
            self.assertEqual(checks[0][3][0]['person_count'],2)
            recorder.stop()

    def test_single_person_day_still_records_without_alert(self):
        frame=np.zeros((100,100,3),np.uint8);detector=MagicMock()
        detector.take.return_value=(1,100,[('person',[5,5,20,50],.9)],frame,
          {'moving_gate':[{'label':'person','qualified':True}], 'full_frame_ran':True, 'captured_at':'2026-11-20T12:00:00+02:00'})
        with tempfile.TemporaryDirectory() as folder, patch('shutil.disk_usage',return_value=SimpleNamespace(free=1000)), patch('motion_preroll.AsyncPreRollWriter') as writer:
            writer.return_value.path=Path(folder)/'object.mp4';factory=MagicMock()
            recorder=ObjectClipRecorder(detector,'camera',folder,factory,MagicMock(),MagicMock(),0)
            checks=recorder.observe(frame,1,True,[(5,5,15,45)])
            factory.assert_called_once();self.assertEqual(checks[0][3],[])
            recorder.stop()

    def test_scenario_changes_are_not_blocked_by_recording_cooldown(self):
        frame=np.zeros((100,100,3),np.uint8);detector=MagicMock()
        crop={'moving_gate':[{'label':'person','qualified':True}], 'full_frame_ran':True,
              'captured_at':'2026-11-20T22:00:00+02:00'}
        second_crop={**crop,'full_frame_detections':[person([5,5,20,50]),person([25,5,40,50])],
                     'verified_full_people':[person([5,5,20,50]),person([25,5,40,50])]}
        detector.take.side_effect=[(1,100,[('person',[5,5,20,50],.9)],frame,crop),
                                   (2,100,[('person',[5,5,20,50],.9),('person',[25,5,40,50],.8)],frame,second_crop)]
        with tempfile.TemporaryDirectory() as folder, patch('shutil.disk_usage',return_value=SimpleNamespace(free=1000)), patch('motion_preroll.AsyncPreRollWriter') as writer:
            writer.return_value.path=Path(folder)/'object.mp4'
            recorder=ObjectClipRecorder(detector,'camera',folder,MagicMock(),MagicMock(),MagicMock(),0)
            first=recorder.observe(frame,1,True,[(5,5,15,45)])
            second=recorder.observe(frame,2,True,[(5,5,15,45)])
            self.assertEqual(first[0][3][0]['scenario'],'one_person_overnight')
            self.assertEqual(second[0][3][0]['scenario'],'two_or_more_people')
            recorder.stop()

    def test_only_matching_scenario_queues_short_object_clip(self):
        frame=np.zeros((100,100,3),np.uint8)
        cases=[('day_single','2026-11-20T12:00:00+02:00',1,False),
               ('night_single','2026-11-20T22:00:00+02:00',1,True),
               ('day_two','2026-11-20T12:00:00+02:00',2,True)]
        for name,time,count,should_send in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder, \
                 patch.dict('os.environ',{'WHATSAPP_OBJECT_ALERTS_ENABLED':'true'}), \
                 patch('shutil.disk_usage',return_value=SimpleNamespace(free=1000)), \
                 patch('motion_preroll.AsyncPreRollWriter') as writer, \
                 patch('object_alerts.queue_object_alert') as queue:
                path=Path(folder)/'object.mp4';writer.return_value.path=path
                detections=[('person',[5,5,20,50],.9)]
                if count==2:detections.append(('person',[25,5,40,50],.8))
                detector=MagicMock()
                detector.take.return_value=(1,100,detections,frame,{'moving_gate':[{'label':'person','qualified':True}],
                                              'full_frame_ran':True,'captured_at':time,
                                              'full_frame_detections':[person(d[1]) for d in detections],
                                              'verified_full_people':[person(d[1]) for d in detections]})
                recorder=ObjectClipRecorder(detector,'camera',folder,MagicMock(),MagicMock(),MagicMock(),0,
                                            alert_camera='Front camera')
                recorder.observe(frame,1,True,[(5,5,15,45)])
                recorder.close()
                if should_send:queue.assert_called_once_with(path,'Front camera',['person'])
                else:queue.assert_not_called()

if __name__ == '__main__':unittest.main()
