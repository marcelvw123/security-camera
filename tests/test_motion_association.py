import unittest
from motion_association import MotionAssociation


def car(x=100):
    return {'label': 'car', 'box': [x,100,x+100,200], 'confidence': .9}


class MotionAssociationTests(unittest.TestCase):
    def test_repeated_motion_inside_vehicle_is_associated(self):
        history=MotionAssociation()
        history.add(9.0,[(110,110,30,30)],1000,600)
        history.add(9.5,[(115,115,30,30)],1000,600)
        result=history.check([car()],10.0,1000,600)[0]
        self.assertEqual(result['status'],'associated')
        self.assertEqual(result['matching_samples'],2)

    def test_separate_motion_does_not_make_parked_car_move(self):
        history=MotionAssociation()
        history.add(1,[(500,400,80,80)],1000,600)
        history.add(1.5,[(500,400,80,80)],1000,600)
        self.assertEqual(history.check([car()],2,1000,600)[0]['status'],'no overlap')

    def test_single_matching_sample_is_only_possible(self):
        history=MotionAssociation()
        history.add(1,[(110,110,30,30)],1000,600)
        history.add(1.5,[],1000,600)
        self.assertEqual(history.check([car()],2,1000,600)[0]['status'],'possible')

    def test_no_history_stale_and_future_are_unknown(self):
        history=MotionAssociation()
        history.add(5,[(100,100,100,100)],1000,600)
        self.assertEqual(history.check([car()],4,1000,600)[0]['status'],'unknown')
        self.assertEqual(history.check([car()],8,1000,600)[0]['status'],'unknown')
        self.assertEqual(history.check([car()],5,800,600)[0]['status'],'unknown')

    def test_empty_motion_sample_is_not_unknown(self):
        history=MotionAssociation();history.add(1,[],1000,600)
        self.assertEqual(history.check([car()],1,1000,600)[0]['status'],'no overlap')

    def test_clear_and_resolution_change_remove_old_samples(self):
        history=MotionAssociation();history.add(1,[(100,100,100,100)],1000,600)
        history.add(2,[],800,600)
        self.assertEqual(history.check([car()],2,1000,600)[0]['status'],'unknown')
        history.clear()
        self.assertEqual(history.check([car()],2,800,600)[0]['status'],'unknown')

    def test_history_is_bounded(self):
        history=MotionAssociation(max_samples=3)
        for t in range(10):history.add(t,[],1000,600)
        self.assertEqual(len(history.samples),3)

    def test_multiple_people_and_vehicles_are_evaluated_independently(self):
        history=MotionAssociation();history.add(1,[(110,110,30,30)],1000,600)
        results=history.check([car(),{'label':'person','box':[500,100,600,200]}],1,1000,600)
        self.assertEqual([r['status'] for r in results],['possible','no overlap'])
