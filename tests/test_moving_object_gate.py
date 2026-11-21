import unittest
from moving_object_gate import MovingObjectGate


def vehicle(x=100, label='bus'):
    return {'label': label, 'box': [x, 100, x+100, 200], 'confidence': .8}


class MovingObjectGateTests(unittest.TestCase):
    def test_motion_beside_parked_bus_does_not_qualify(self):
        gate = MovingObjectGate()
        for t in (1, 2, 3):
            result = gate.observe([vehicle()], [(250, 100, 50, 80)], t, 1000, 600)
            self.assertFalse(result[0]['qualified'])
            self.assertEqual(result[0]['reason'], 'insufficient motion overlap')

    def test_motion_on_stationary_bus_does_not_qualify(self):
        gate = MovingObjectGate()
        for t in (1, 2, 3):
            result = gate.observe([vehicle()], [(100, 100, 70, 80)], t, 1000, 600)
            self.assertFalse(result[0]['qualified'])
        self.assertEqual(result[0]['reason'], 'vehicle displacement below threshold')

    def test_vehicle_requires_two_overlapping_observations_and_displacement(self):
        gate = MovingObjectGate()
        self.assertFalse(gate.observe([vehicle()], [(100, 100, 100, 100)], 1, 1000, 600)[0]['qualified'])
        result = gate.observe([vehicle(125, 'car')], [(125, 100, 100, 100)], 2, 1000, 600)
        self.assertTrue(result[0]['qualified'])

    def test_person_requires_two_motion_observations(self):
        gate = MovingObjectGate()
        person = {'label': 'person', 'box': [100, 100, 150, 200]}
        self.assertFalse(gate.observe([person], [(100, 100, 50, 100)], 1, 1000, 600)[0]['qualified'])
        self.assertTrue(gate.observe([person], [(100, 100, 50, 100)], 2, 1000, 600)[0]['qualified'])

    def test_stale_samples_and_long_gaps_reset_tracks(self):
        gate = MovingObjectGate()
        gate.observe([vehicle()], [(100, 100, 100, 100)], 1, 1000, 600)
        self.assertFalse(gate.observe([vehicle(125)], [(125, 100, 100, 100)], 10, 1000, 600)[0]['qualified'])
        self.assertFalse(gate.observe([vehicle(150)], [(150, 100, 100, 100)], 9, 1000, 600)[0]['qualified'])

    def test_multiple_detections_do_not_share_a_track(self):
        gate = MovingObjectGate()
        result = gate.observe([vehicle(), vehicle(120)], [(100, 100, 120, 100)], 1, 1000, 600)
        self.assertTrue(all(not item['qualified'] for item in result))

    def test_single_detection_can_use_coherent_image_motion(self):
        gate = MovingObjectGate()
        result = gate.observe([vehicle()], [(100, 100, 100, 100)], 1, 1000, 600, [1.2])
        self.assertTrue(result[0]['qualified'])
        self.assertIn('image motion', result[0]['reason'])

    def test_image_motion_cannot_bypass_overlap(self):
        gate = MovingObjectGate()
        result = gate.observe([vehicle()], [(300, 100, 100, 100)], 1, 1000, 600, [2.0])
        self.assertFalse(result[0]['qualified'])


if __name__ == '__main__':
    unittest.main()
