import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
from object_detection import ObjectClipRecorder, process_motion_frame


class ActiveMotionGateTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict('os.environ', {'WHATSAPP_OBJECT_ALERTS_ENABLED': 'false'})
        env.start()
        self.addCleanup(env.stop)

    def test_no_full_frame_inference_without_confirmed_crop(self):
        frame=np.zeros((100,100,3),np.uint8);model=MagicMock();gate=MagicMock()
        gate.observe.return_value=[{'label':'bus','qualified':False}]
        with patch('motion_crop.detect_motion_crop',return_value=[{'label':'bus','confidence':.8,'box':[10,10,30,30]}]):
            detections,result=process_motion_frame(model,frame,(0,0,50,50),[(10,10,10,10)],None,gate,1)
        self.assertEqual(detections,[])
        self.assertFalse(result['full_frame_ran'])
        model.predict.assert_not_called()

    def test_full_frame_runs_after_moving_crop(self):
        frame=np.zeros((100,100,3),np.uint8);model=MagicMock();gate=MagicMock()
        gate.observe.return_value=[{'label':'person','box':[10,10,30,70],'qualified':True}]
        model.names={5:'bus'}
        model.predict.return_value=[SimpleNamespace(boxes=SimpleNamespace(xyxy=[[50,20,90,80]],conf=[.8],cls=[5]))]
        crop=[{'label':'person','confidence':.9,'box':[10,10,30,70]}]
        with patch('motion_crop.detect_motion_crop',return_value=crop):
            detections,result=process_motion_frame(model,frame,(0,0,50,80),[(10,10,10,50)],None,gate,1)
        self.assertTrue(result['full_frame_ran'])
        self.assertEqual({d[0] for d in detections},{'person','bus'})

    def test_static_full_frame_label_cannot_start_clip_or_alert(self):
        detector=MagicMock();detector.take.return_value=(1,100,[('bus',[10,10,50,50],.8)],None,
            {'moving_gate':[{'label':'bus','qualified':False}]})
        with tempfile.TemporaryDirectory() as directory:
            factory=MagicMock();recorder=ObjectClipRecorder(detector,'camera',directory,factory,MagicMock(),MagicMock(),0)
            checks=recorder.observe(np.zeros((100,100,3),np.uint8),1,True,[(10,10,20,20)])
            factory.assert_not_called();self.assertEqual(checks[0][3],[])

    def test_moving_person_triggers_but_stationary_bus_is_only_context(self):
        frame=np.zeros((100,100,3),np.uint8);detector=MagicMock()
        detector.take.return_value=(1,100,[('person',[10,10,30,70],.9),('bus',[50,10,90,80],.8)],frame,
            {'moving_gate':[{'label':'person','qualified':True},{'label':'bus','qualified':False}],
             'full_frame_ran':True, 'captured_at':'2026-11-20T22:00:00+02:00'})
        with tempfile.TemporaryDirectory() as directory, patch('shutil.disk_usage',return_value=SimpleNamespace(free=1000)), patch('motion_preroll.AsyncPreRollWriter') as writer:
            writer.return_value.path=Path(directory)/'object.mp4';factory=MagicMock()
            recorder=ObjectClipRecorder(detector,'camera',directory,factory,MagicMock(),MagicMock(),0)
            checks=recorder.observe(frame,1,True,[(10,10,20,60)])
            factory.assert_called_once()
            self.assertEqual([e['label'] for e in checks[0][3]],['person'])
            recorder.stop()

    def test_no_object_inference_without_motion(self):
        detector=MagicMock();detector.take.return_value=None
        with tempfile.TemporaryDirectory() as directory:
            recorder=ObjectClipRecorder(detector,'camera',directory,MagicMock(),MagicMock(),MagicMock(),0)
            recorder.observe(np.zeros((100,100,3),np.uint8),1,False)
            detector.submit.assert_not_called()

    def test_small_unconfirmed_person_does_not_pass_gate(self):
        frame=np.zeros((720,1280,3),np.uint8);model=MagicMock();gate=MagicMock()
        bird={'label':'person','confidence':.61,'box':[540,440,548,458]}
        pole={'label':'person','confidence':.65,'box':[588,354,608,389]}
        gate.observe.return_value=[{**bird,'qualified':True},{**pole,'qualified':False}]
        model.predict.return_value=[SimpleNamespace(boxes=SimpleNamespace(xyxy=[],conf=[],cls=[]))]
        with patch('motion_crop.detect_motion_crop',return_value=[bird,pole]):
            detections,details=process_motion_frame(model,frame,(400,300,650,500),[(540,440,8,18)],None,gate,1)
        self.assertEqual(detections,[])
        self.assertFalse(any(item['qualified'] for item in details['moving_gate']))
        self.assertEqual(details['moving_gate'][0]['reason'],'small person not confirmed in full frame')
