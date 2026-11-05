import unittest

import numpy as np

from motion_decomposition.fixed_times import fit_fixed_endpoints
from motion_decomposition.synthetic import generate_motion


class FixedEndpointTests(unittest.TestCase):
    def test_hand_calculated_imperfect_fit_retains_both_endpoints(self):
        positions = np.array([[2, 0, 0], [5, 0, 0], [7, 0, 0], [8, 0, 0]])
        result = fit_fixed_endpoints([0, 1, 2, 3], positions, [2, 0, 0], [8, 0, 0], [0, 1], [2, 3])
        np.testing.assert_allclose(result['displacements'], [[5.0, 0, 0], [1.0, 0, 0]], atol=1e-12)
        np.testing.assert_allclose(result['positions'][:, 0], [2, 4.5, 7.5, 8], atol=1e-12)
        self.assertGreater(result['position_rmse_m'], 0)
        self.assertLess(result['final_error_m'], 1e-12)
        np.testing.assert_allclose(result['free_basis'].T @ (result['positions'] - positions), 0, atol=1e-12)

    def test_known_overlapping_motion(self):
        times = np.linspace(0, 4, 401)
        initial = np.array([0.2, -0.1, 0.4])
        displacement = np.array([[0.3, 0.1, -0.05], [-0.12, 0.2, 0.08]])
        final = initial + displacement.sum(axis=0)
        positions = generate_motion(times, initial, displacement, [0.2, 1.2], [2.2, 3.2])
        result = fit_fixed_endpoints(times, positions, initial, final, [0.2, 1.2], [2.2, 3.2])
        np.testing.assert_allclose(result['displacements'], displacement, rtol=0, atol=1e-12)
        self.assertTrue(result['full_column_rank'])

    def test_one_component_has_no_free_displacement(self):
        result = fit_fixed_endpoints([0, 1, 2], [[2, 0, 0], [6, 0, 0], [8, 0, 0]],
                                     [2, 0, 0], [8, 0, 0], [0], [2])
        np.testing.assert_allclose(result['displacements'], [[6, 0, 0]], atol=1e-12)
        self.assertEqual(result['free_columns'], 0)
        self.assertIsNone(result['condition_number'])

    def test_same_times_remain_nonidentifiable_with_fixed_endpoints(self):
        positions = np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0]])
        result = fit_fixed_endpoints([0, 1, 2], positions, [0, 0, 0], [1, 0, 0], [0, 0], [2, 2])
        self.assertFalse(result['full_column_rank'])
        self.assertEqual(result['free_rank'], 0)
        np.testing.assert_allclose(result['positions'], positions, atol=1e-12)

    def test_endpoint_targets_are_not_overwritten_by_observations(self):
        result = fit_fixed_endpoints([0, 1, 2], [[0.1, 0, 0], [0.6, 0, 0], [1.1, 0, 0]],
                                     [0, 0, 0], [1, 0, 0], [0], [2])
        np.testing.assert_allclose(result['positions'][[0, -1]], [[0, 0, 0], [1, 0, 0]], atol=1e-12)

    def test_invalid_boundaries_and_positions_rejected(self):
        for activation, completion in [([-1], [2]), ([0], [3])]:
            with self.assertRaises(ValueError):
                fit_fixed_endpoints([0, 1, 2], np.zeros((3, 3)), [0, 0, 0], [1, 0, 0], activation, completion)
        with self.assertRaises(ValueError):
            fit_fixed_endpoints([0, 1, 2], np.zeros((3, 3)), [0, 0, 0], [np.nan, 0, 0], [0], [2])


if __name__ == '__main__':
    unittest.main()
