import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np
from person_verifier import verify_scene_people
from notification_scenarios import count_people, scenario_people


def candidate(box, confidence=.8):
    return {'label':'person','box':box,'confidence':confidence}


class PersonVerifierTests(unittest.TestCase):
    def test_verification_matches_the_candidate_in_source_coordinates(self):
        frame=np.zeros((200,200,3),np.uint8)
        model=MagicMock()
        # For [50,50,90,130], the padded crop starts at [30,10].
        model.predict.return_value=[SimpleNamespace(boxes=SimpleNamespace(xyxy=[[20,40,60,120]],conf=[.8]))]
        result=verify_scene_people(model,frame,[candidate([50,50,90,130])])
        self.assertTrue(result[0]['verified'])
        self.assertEqual(result[0]['verification_confidence'],.8)

    def test_nonmatching_detection_does_not_verify_candidate(self):
        frame=np.zeros((200,200,3),np.uint8);model=MagicMock()
        model.predict.return_value=[SimpleNamespace(boxes=SimpleNamespace(xyxy=[[0,0,10,10]],conf=[.9]))]
        result=verify_scene_people(model,frame,[candidate([50,50,90,130])])
        self.assertFalse(result[0]['verified'])

    def test_group_requires_two_verified_independent_people(self):
        full=[candidate([0,0,20,50]),candidate([40,0,60,50])]
        self.assertEqual(count_people(scenario_people(full,full,[])),1)
        self.assertEqual(count_people(scenario_people(full,full,full[:1])),1)
        self.assertEqual(count_people(scenario_people(full,full,full)),2)

    def test_candidate_work_is_bounded(self):
        frame=np.zeros((200,200,3),np.uint8);model=MagicMock()
        model.predict.return_value=[SimpleNamespace(boxes=SimpleNamespace(xyxy=[],conf=[]))]
        verify_scene_people(model,frame,[candidate([10*i,10,10*i+5,50]) for i in range(10)])
        self.assertEqual(model.predict.call_count,4)

    def test_two_candidates_verifying_as_the_same_person_count_once(self):
        frame=np.zeros((200,200,3),np.uint8);model=MagicMock()
        # Both padded crops refer to the same full-frame person.
        model.predict.side_effect=[
            [SimpleNamespace(boxes=SimpleNamespace(xyxy=[[20,40,60,120]],conf=[.8]))],
            [SimpleNamespace(boxes=SimpleNamespace(xyxy=[[22,30,62,110]],conf=[.9]))],
        ]
        checked=verify_scene_people(model,frame,[candidate([50,50,90,130]),candidate([48,60,88,140],.7)])
        verified=[{**item,'box':item['verification_box']} for item in checked if item['verified']]
        self.assertEqual(count_people(verified),1)

if __name__=='__main__':unittest.main()
