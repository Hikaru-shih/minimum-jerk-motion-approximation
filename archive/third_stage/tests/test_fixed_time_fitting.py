import unittest

import numpy as np

from motion_decomposition.fixed_times import design_matrix, fit_displacements
from motion_decomposition.step1_known_motion import generate_motion


class FixedTimeFittingTests(unittest.TestCase):
    def test_hand_calculated_basis(self):
        actual = design_matrix([-1, 0, 0.25, 0.5, 0.75, 1, 2], [0], [1])
        expected = [[0], [0], [0.103515625], [0.5], [0.896484375], [1], [1]]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)

    def test_overlapping_displacements_recovered(self):
        times = np.linspace(0, 4, 401)
        initial = np.array([0.2, -0.1, 0.4])
        displacements = np.array([[0.3, 0.1, -0.05], [-0.12, 0.2, 0.08]])
        original = generate_motion(times, initial, displacements, [0.2, 1.2], [2.2, 3.2])
        result = fit_displacements(times, original, initial, [0.2, 1.2], [2.2, 3.2])
        self.assertEqual(result['rank'], 2)
        np.testing.assert_allclose(result['displacements'], displacements, rtol=0, atol=1e-12)
        np.testing.assert_allclose(result['goals'], [[0.5, 0, 0.35], [0.38, 0.2, 0.43]], rtol=0, atol=1e-12)

    def test_identical_times_only_identify_displacement_sum(self):
        times = np.linspace(0, 3, 301)
        initial = np.zeros(3)
        displacements = np.array([[0.3, 0, 0], [-0.1, 0.2, 0]])
        positions = generate_motion(times, initial, displacements, [0, 0], [2, 2])
        result = fit_displacements(times, positions, initial, [0, 0], [2, 2])
        self.assertEqual(result['rank'], 1)
        self.assertFalse(result['full_column_rank'])
        np.testing.assert_allclose(result['displacements'], [[0.1, 0.1, 0], [0.1, 0.1, 0]], atol=1e-12)
        np.testing.assert_allclose(result['positions'], positions, rtol=0, atol=1e-12)

    def test_least_squares_on_off_model_data(self):
        times = np.linspace(0, 3, 301)
        initial = np.array([0.1, 0.2, 0.3])
        truth = np.array([[0.2, 0.1, 0], [0, -0.1, 0.3]])
        positions = generate_motion(times, initial, truth, [0, 1], [2, 3])
        positions += np.random.default_rng(27).normal(0, 0.001, positions.shape)
        result = fit_displacements(times, positions, initial, [0, 1], [2, 3])
        residual = result['positions'] - positions
        np.testing.assert_allclose(result['basis'].T @ residual, 0, rtol=0, atol=1e-12)
        original_residual = initial + result['basis'] @ truth - positions
        self.assertLessEqual(np.sum(residual**2), np.sum(original_residual**2) + 1e-14)

    def test_invalid_inputs_rejected(self):
        for times, activation, completion in [
            ([0, 0], [0], [1]), ([0, 1], [1], [1]),
            ([0, 1], [0], [np.nan]), ([0, 1], [], []),
            ([0, 1], [0, 1], [2]),
        ]:
            with self.subTest(times=times, activation=activation):
                with self.assertRaises(ValueError):
                    design_matrix(times, activation, completion)
        with self.assertRaises(ValueError):
            fit_displacements([0, 1], [[0, 0], [1, 1]], [0, 0, 0], [0], [1])

if __name__ == '__main__':
    unittest.main()
