import unittest
from weak_motion_vehicle import WeakMotionVehicle


def candidate(x, label='car', confidence=.08):
    return {'label': label, 'confidence': confidence, 'box': [x, 100, x+70, 150]}


class WeakVehicleTests(unittest.TestCase):
    def test_two_motion_backed_observations_with_travel(self):
        gate = WeakMotionVehicle()
        self.assertFalse(gate.observe([candidate(100)], [[100,100,70,50]], [.8], 1, 1000, 600)[0]['qualified'])
        self.assertTrue(gate.observe([candidate(125)], [[125,100,70,50]], [1.4], 1.5, 1000, 600)[0]['qualified'])

    def test_stationary_and_single_image_cannot_qualify(self):
        gate = WeakMotionVehicle()
        for t in (1, 1.5, 2):
            self.assertEqual(gate.observe([candidate(100)], [[100,100,70,50]], [0.0], t, 1000, 600), [])
        result = gate.observe([candidate(100), candidate(100,'bus')], [[100,100,70,50]], [1,1], 3, 1000, 600)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]['qualified'])

    def test_motion_must_overlap_and_samples_expire(self):
        gate = WeakMotionVehicle()
        self.assertEqual(gate.observe([candidate(100)], [[300,100,50,50]], [2], 1, 1000, 600), [])
        gate.observe([candidate(100)], [[100,100,70,50]], [2], 2, 1000, 600)
        self.assertFalse(gate.observe([candidate(125)], [[125,100,70,50]], [2], 5, 1000, 600)[0]['qualified'])

    def test_invalid_inputs_are_ignored(self):
        gate=WeakMotionVehicle()
        self.assertEqual(gate.observe([candidate(100)], [[100,100,70,50]], [None], 1, 1000, 600), [])
        self.assertEqual(gate.observe([candidate(100, confidence=.01)], [[100,100,70,50]], [2], 2, 1000, 600), [])


if __name__ == '__main__':
    unittest.main()
