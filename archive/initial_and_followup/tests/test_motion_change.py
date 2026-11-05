from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.followup.motion_change import local_peaks, motion_scores, select_motion_goals


class MotionChangeChecks(unittest.TestCase):
    def test_quadratic_has_known_acceleration(self):
        times = np.linspace(0, 2, 21)
        positions = 0.5 * times[:, None]**2 * np.array([3, 4, 0])
        scores, _, acceleration = motion_scores(positions, times)
        np.testing.assert_allclose(acceleration, np.tile([3, 4, 0], (21, 1)), atol=1e-10)
        np.testing.assert_allclose(scores, 5, atol=1e-10)

    def test_constant_velocity_has_negligible_score(self):
        times = np.linspace(0, 2, 21)
        positions = times[:, None] * np.array([1, -2, 3])
        scores, _, _ = motion_scores(positions, times)
        np.testing.assert_allclose(scores, 0, atol=1e-10)

    def test_plateau_ties_and_endpoints(self):
        np.testing.assert_array_equal(local_peaks([9, 0, 3, 3, 1, 2, 0, 9]), [2, 5])
        self.assertEqual(len(local_peaks([0, 0, 0, 0])), 0)

    def test_score_order_and_minimum_gap(self):
        times = np.arange(9, dtype=float)
        scores = np.array([0, 0, 8, 0, 9, 0, 8, 0, 0])
        selected, _, order = select_motion_goals(scores, times, max_goals=4, min_gap_s=1)
        np.testing.assert_array_equal(order, [4, 2, 6])
        np.testing.assert_array_equal(selected, [0, 2, 4, 6, 8])
        selected, _, order = select_motion_goals(scores, times, max_goals=4, min_gap_s=3)
        np.testing.assert_array_equal(order, [4])
        np.testing.assert_array_equal(selected, [0, 4, 8])

    def test_no_peaks_does_not_fill_with_arbitrary_frames(self):
        selected, ranked, order = select_motion_goals(np.zeros(5), np.arange(5.), max_goals=4)
        np.testing.assert_array_equal(selected, [0, 4])
        self.assertEqual(len(ranked), 0)
        self.assertEqual(len(order), 0)


if __name__ == '__main__':
    unittest.main()
