import unittest

import numpy as np

from motion_decomposition.fixed_times import fit_fixed_endpoints
from motion_decomposition.opposition import fit_opposition_fixed_times, opposition_residuals
from motion_decomposition.step1_known_motion import generate_motion


class OppositionTests(unittest.TestCase):
    def test_simultaneous_opposite_penalized_but_sequential_return_allowed(self):
        times = np.linspace(0, 3, 301)
        d = np.array([[1., 0, 0], [-1., 0, 0]])
        self.assertGreater(np.sum(opposition_residuals(times, [0, 0], [2, 2], d, 1)**2), 0)
        np.testing.assert_array_equal(opposition_residuals(times, [0, 1.5], [1.5, 3], d, 1), 0)

    def test_same_direction_orthogonal_and_rotation(self):
        times = np.linspace(0, 3, 301)
        for second in ([1, 0, 0], [0, 1, 0]):
            np.testing.assert_array_equal(opposition_residuals(times, [0, 0], [2, 2], [[1,0,0], second], 1), 0)
        d = np.array([[1., 2, 0], [-2., -1, 3]])
        q, _ = np.linalg.qr(np.array([[1., 2, 3], [2., -1, 1], [0., 1, 2]]))
        np.testing.assert_allclose(opposition_residuals(times, [0, 0], [2, 2], d, 1),
                                   opposition_residuals(times, [0, 0], [2, 2], d @ q, 1), atol=1e-12)

    def test_zero_weight_matches_linear_fit_and_positive_weight_keeps_endpoints(self):
        t = np.linspace(0, 3, 301)
        p = generate_motion(t, np.zeros(3), [[0.4,0,0], [-0.1,0,0]], [0,0.2], [2,2.2])
        base = fit_fixed_endpoints(t, p, p[0], p[-1], [0,0.2], [2,2.2])
        zero = fit_opposition_fixed_times(t, p, p[0], p[-1], [0,0.2], [2,2.2], 0, 0.3)
        np.testing.assert_array_equal(zero['displacements'], base['displacements'])
        positive = fit_opposition_fixed_times(t, p, p[0], p[-1], [0,0.2], [2,2.2], 1, 0.3)
        self.assertLess(positive['opposition_cost_m2'], zero['opposition_cost_m2'])
        self.assertGreater(positive['position_rmse_m'], zero['position_rmse_m'])
        np.testing.assert_allclose(positive['positions'][[0,-1]], p[[0,-1]], atol=1e-12)

    def test_invalid_weight_or_scale_rejected(self):
        for weight, scale in [(-1, 1), (np.nan, 1), (1, 0), (1, np.inf)]:
            with self.assertRaises(ValueError):
                fit_opposition_fixed_times([0,1,2], np.zeros((3,3)), np.zeros(3), np.zeros(3), [0], [2], weight, scale)
