import unittest

import numpy as np

from motion_decomposition.step1_known_motion import generate_motion
from motion_decomposition.timing_fit import decode_times, encode_times, fit_timing


class TimingFitTests(unittest.TestCase):
    def test_timing_parameter_round_trip_and_bounds(self):
        parameters = encode_times([0.2, 1.2], [2.2, 3.2], 0, 3.4, 0.02)
        start, end = decode_times(parameters, 0, 3.4, 0.02)
        np.testing.assert_allclose(start, [0.2, 1.2], atol=1e-12)
        np.testing.assert_allclose(end, [2.2, 3.2], atol=1e-12)
        start, end = decode_times([[0, 0], [1, 1]], 0, 3.4, 0.02)
        self.assertTrue(np.all(start >= 0) and np.all(end <= 3.4))
        self.assertTrue(np.all(end - start >= 0.02 - 1e-12))

    def test_known_times_recovered_from_reversed_labels(self):
        times = np.linspace(0, 3.4, 205)
        initial = np.array([0.2, -0.1, 0.4])
        displacements = np.array([[0.3, 0.1, -0.05], [-0.12, 0.2, 0.08]])
        final = initial + displacements.sum(axis=0)
        positions = generate_motion(times, initial, displacements, [0.2, 1.2], [2.2, 3.2])
        result = fit_timing(times, positions, initial, final, [0.8, 0], [3.4, 1.8], 0.02)
        self.assertTrue(result['success'])
        np.testing.assert_allclose(result['activation'], [0.2, 1.2], atol=1e-5)
        np.testing.assert_allclose(result['completion'], [2.2, 3.2], atol=1e-5)
        np.testing.assert_allclose(result['displacements'], displacements, atol=1e-7)
        self.assertLess(result['initial_error_m'], 1e-12)
        self.assertLess(result['final_error_m'], 1e-12)
        self.assertLess(result['position_rmse_m'], result['initial_position_rmse_m'])

    def test_no_motion_does_not_identify_timing(self):
        times = np.linspace(0, 3.4, 101)
        initial = np.array([0.2, 0.1, 0.3])
        positions = np.tile(initial, (len(times), 1))
        result = fit_timing(times, positions, initial, initial, [0.1, 1.1], [2, 3.1], 0.02)
        self.assertLess(result['position_rmse_m'], 1e-12)
        self.assertEqual(result['timing_jacobian_rank'], 0)

    def test_invalid_duration_and_initial_times(self):
        with self.assertRaises(ValueError):
            decode_times([[0.5, 0.5]], 0, 1, 0)
        with self.assertRaises(ValueError):
            decode_times([[0.5, 0.5]], 0, 1, 2)
        with self.assertRaises(ValueError):
            encode_times([0], [0.01], 0, 1, 0.02)
        with self.assertRaises(ValueError):
            decode_times([[1.2, 0.5]], 0, 1, 0.02)


if __name__ == '__main__':
    unittest.main()
