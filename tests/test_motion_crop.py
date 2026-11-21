import threading
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from motion_crop import detect_motion_crop, select_motion_crop, merge_scene_detections
from object_detection import SharedObjectDetector


class MotionCropTests(unittest.TestCase):
    def test_crop_is_padded_clamped_and_uses_largest_region(self):
        crop=select_motion_crop([(5,10,60,40),(500,250,180,100)],960,576)
        self.assertEqual(crop,(410,200,770,400))
        self.assertEqual(select_motion_crop([(0,0,20,20)],960,576),(0,0,90,90))

    def test_invalid_and_giant_regions_are_ignored(self):
        self.assertIsNone(select_motion_crop([('x',1,2,3),(0,0,-1,2),(0,0,960,576)],960,576))

    def test_crop_prediction_maps_back_to_full_frame_coordinates(self):
        frame=np.zeros((500,600,3),np.uint8)
        model=MagicMock();model.names={0:'person'}
        model.predict.return_value=[SimpleNamespace(boxes=SimpleNamespace(xyxy=[[2,3,25,35]],conf=[.8],cls=[0]))]
        result=detect_motion_crop(model,frame,(100,200,300,400))
        self.assertEqual(model.predict.call_args.args[0].shape,(200,200,3))
        self.assertEqual(result[0]['box'],[102,203,125,235])
        self.assertEqual(result[0]['label'],'person')

    def test_same_frame_results_merge_duplicates_and_keep_distinct_objects(self):
        full=[{'label':'car','confidence':.7,'box':[100,100,200,200]},
              {'label':'person','confidence':.8,'box':[300,100,350,200]}]
        crop=[{'label':'bus','confidence':.9,'box':[102,102,198,198]},
              {'label':'person','confidence':.9,'box':[400,100,450,200]}]
        result=merge_scene_detections(full,crop)
        self.assertEqual(len(result),3)
        self.assertEqual(sum(item['label']=='person' for item in result),2)
        self.assertEqual(sum(item['label'] in ('car','bus') for item in result),1)

    def test_motion_crop_is_required_and_rate_limited_per_camera(self):
        detector=SharedObjectDetector.__new__(SharedObjectDetector)
        detector.remote_shadow=None;detector.condition=threading.Condition()
        detector.status='ready';detector.interval=1;detector.last_submitted={}
        detector.last_crop_submitted={};detector.generations={};detector.pending=OrderedDict()
        frame=np.zeros((576,960,3),np.uint8)
        detector.submit('a',frame,1,[])
        self.assertNotIn('a',detector.pending)
        detector.submit('a',frame,1,[(10,10,100,100)])
        self.assertIsNotNone(detector.pending['a'][3])
        detector.submit('b',frame,1,[(10,10,100,100)])
        self.assertIsNotNone(detector.pending['b'][3])
