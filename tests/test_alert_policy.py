import unittest
from alert_policy import AlertPolicy


def det(label, x=100, y=100, w=100, h=60, confidence=.8):
    return {'label':label, 'box':[x,y,x+w,y+h], 'confidence':confidence}


class AlertPolicyTests(unittest.TestCase):
    def test_stationary_baseline_vehicle_does_not_alert(self):
        policy=AlertPolicy()
        for t in (0,5,10,15,20,25,30):
            events,_=policy.observe('full',[det('bus',100+(t%2))],t,1000,600)
            self.assertEqual(events,[])

    def test_new_vehicle_alerts_once_after_confirmation(self):
        policy=AlertPolicy()
        for t in (0,5,10,15): policy.observe('full',[],t,1000,600)
        self.assertEqual(policy.observe('full',[det('car',20)],20,1000,600)[0],[])
        events,_=policy.observe('full',[det('car',45)],22,1000,600)
        self.assertEqual(events[0]['reason'],'new vehicle confirmed')
        self.assertEqual(policy.observe('full',[det('car',70)],24,1000,600)[0],[])

    def test_baseline_vehicle_that_moves_alerts(self):
        policy=AlertPolicy()
        for t in (0,5,10,15):policy.observe('full',[det('car',100)],t,1000,600)
        events,_=policy.observe('full',[det('car',145)],20,1000,600)
        self.assertEqual(events[0]['reason'],'vehicle moved')

    def test_person_confirmed_even_during_baseline_and_stationary(self):
        policy=AlertPolicy()
        self.assertEqual(policy.observe('full',[det('person')],1,1000,600)[0],[])
        events,_=policy.observe('full',[det('person')],2,1000,600)
        self.assertEqual(events[0]['label'],'person')
        for t in (4,10,22,40):self.assertEqual(policy.observe('full',[det('person')],t,1000,600)[0],[])

    def test_reconnect_rebaselines_and_old_detections_ignored(self):
        policy=AlertPolicy()
        for t in (0,5,10,15):policy.observe('full',[],t,1000,600)
        policy.observe('full',[det('car')],50,1000,600)
        self.assertEqual(policy.observe('full',[det('car')],55,1000,600)[0],[])
        self.assertEqual(policy.observe('full',[det('car')],51,1000,600)[1],['stale sample ignored'])

    def test_low_confidence_and_large_vehicle_box_are_rejected(self):
        policy=AlertPolicy()
        events,decisions=policy.observe('vehicle',[det('car',confidence=.12),det('bus',0,0,900,500)],1,1000,600)
        self.assertEqual(events,[])
        self.assertIn('vehicle rejected: low confidence',decisions)
        self.assertIn('vehicle rejected: oversized box',decisions)

    def test_vehicle_area_does_not_filter_people(self):
        policy=AlertPolicy()
        vehicle=det('truck',100,100);person=det('person',100,100)
        events,decisions=policy.observe('full',[vehicle,person],1,1000,600,vehicle_area=[.5,0,.5,1])
        self.assertIn('vehicle outside configured vehicle area',decisions)
        self.assertEqual(policy.observe('full',[person],2,1000,600,vehicle_area=[.5,0,.5,1])[0][0]['label'],'person')

    def test_separate_model_trackers_group_same_vehicle(self):
        policy=AlertPolicy()
        for t in (0,5,10,15):
            policy.observe('full',[],t,1000,600);policy.observe('vehicle',[],t,1000,600)
        policy.observe('full',[det('car',20)],20,1000,600)
        self.assertTrue(policy.observe('full',[det('car',30)],22,1000,600)[0])
        policy.observe('vehicle',[det('car',20)],20,1000,600)
        self.assertFalse(policy.observe('vehicle',[det('car',30)],22,1000,600)[0])
