import unittest

import numpy as np

from motion_decomposition.goal_count_selection import chord_knee,detection_speed,speed_valley_knots,valley_initializations
from motion_decomposition.synthetic import generate_motion


class GoalCountSelectionTests(unittest.TestCase):
    def test_known_knee_and_affine_scale_invariance(self):
        result=chord_knee([1,2,3,4],[10,3,2,1])
        self.assertEqual(result['selected_count'],2)
        self.assertEqual(chord_knee([1,2,3,4],[35,14,11,8])['selected_count'],2)

    def test_no_manufactured_knee_for_linear_flat_or_nonmonotone_curve(self):
        for values in ([4,3,2,1],[1,1,1,1],[4,2,3,1]):
            self.assertIsNone(chord_knee([1,2,3,4],values)['selected_count'])
        with self.assertRaises(ValueError):
            chord_knee([1,2.5,3],[5,2,1])

    def test_speed_lobes_have_valley_boundary_not_peak_boundary(self):
        t=np.arange(9)*.1
        result=speed_valley_knots(t,[0,1,3,1,.2,1,4,1,0],.1,.15)
        self.assertEqual(result['component_count'],2)
        np.testing.assert_array_equal(result['peak_frames'],[2,6])
        np.testing.assert_array_equal(result['boundary_frames'],[0,4,8])

    def test_constant_speed_fallback_and_invalid_settings(self):
        t=np.arange(9)*.1
        result=speed_valley_knots(t,np.ones(9))
        self.assertEqual(result['component_count'],1)
        self.assertTrue(result['no_peak_fallback'])
        with self.assertRaises(ValueError):
            speed_valley_knots(t,np.ones(9),0)

    def test_detection_derivative_and_reproducible_bounded_starts(self):
        t=np.arange(121)/120
        positions=np.column_stack([2*t,np.zeros_like(t),np.zeros_like(t)])
        np.testing.assert_allclose(detection_speed(t,positions,17),2,atol=1e-10)
        one,two=valley_initializations([0,.3,1]),valley_initializations([0,.3,1])
        for a,b in zip(one,two):
            np.testing.assert_array_equal(a['activation'],b['activation'])
            self.assertTrue(np.all(a['activation']>=0) and np.all(a['completion']<=1))
            self.assertTrue(np.all(a['completion']>a['activation']))

    def test_separated_pulses_detected_but_identical_overlapping_pulses_merge(self):
        t=np.arange(481)/120
        for activation,completion,expected in [([.2,2.2],[1.8,3.8],2),([.2,.2],[3.8,3.8],1)]:
            positions=generate_motion(t,np.zeros(3),[[.3,0,0],[.2,0,0]],activation,completion)
            detected=speed_valley_knots(t,detection_speed(t,positions,17),.1,.15)
            self.assertEqual(detected['component_count'],expected)
