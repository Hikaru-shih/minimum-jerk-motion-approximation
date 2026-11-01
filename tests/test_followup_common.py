from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.followup.common import (
    overlap_times, reconstruct_samples, select_error_candidate, trajectory_errors,
)


class FollowupChecks(unittest.TestCase):
    def test_constant_offset_uses_vector_distance(self):
        times = np.arange(5, dtype=float)
        original = np.zeros((5, 3))
        approximated = np.tile([0.03, 0.04, 0], (5, 1))
        metrics, errors = trajectory_errors(original, approximated, times)
        np.testing.assert_allclose(errors, 0.05, rtol=0, atol=1e-12)
        self.assertAlmostEqual(metrics['position_rmse_m'], 0.05)
        self.assertAlmostEqual(metrics['max_position_error_m'], 0.05)
        self.assertAlmostEqual(metrics['velocity_rmse_m_s'], 0)

    def test_velocity_direction_error_is_not_speed_error(self):
        times = np.arange(5, dtype=float)
        original = np.column_stack([times, times * 0, times * 0])
        metrics, _ = trajectory_errors(original, -original, times)
        self.assertAlmostEqual(metrics['velocity_rmse_m_s'], 2)

    def test_unequal_intervals_use_shorter_neighbor(self):
        activation, completion = overlap_times(np.array([0, 1, 1.2]))
        np.testing.assert_allclose(activation, [0, 0.95])
        np.testing.assert_allclose(completion, [1.05, 1.2])

    def test_equal_intervals_reproduce_regular_overlap(self):
        activation, completion = overlap_times(np.array([0, 0.5, 1]))
        np.testing.assert_allclose(activation, [0, 0.375])
        np.testing.assert_allclose(completion, [0.625, 1])

    def test_reconstruction_retains_endpoints(self):
        times = np.linspace(0, 1, 11)
        original = np.column_stack([times, times**2, -times])
        positions, _, _ = reconstruct_samples(original, times, np.array([0, 4, 10]))
        np.testing.assert_allclose(positions[[0, -1]], original[[0, -1]], atol=1e-12)

    def test_candidate_excludes_nearby_frames_and_selected_points(self):
        times = np.arange(5, dtype=float)
        errors = np.array([100, 9, 8, 7, 100])
        self.assertEqual(select_error_candidate(errors, times, [0, 4], 1.5), 2)
        with self.assertRaises(ValueError):
            select_error_candidate(errors, times, [0, 4], 3)

    def test_invalid_times_are_rejected(self):
        with self.assertRaises(ValueError):
            trajectory_errors(np.zeros((3, 3)), np.zeros((3, 3)), [0, 0, 1])
        with self.assertRaises(ValueError):
            overlap_times([0, 1], overlap_fraction=0.5)


if __name__ == '__main__':
    unittest.main()
