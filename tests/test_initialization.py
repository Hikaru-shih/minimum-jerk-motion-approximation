import unittest

import numpy as np

from motion_decomposition.initialization import choose_best_result, make_initializations, speed_seed_knots, times_from_knots


class InitializationTests(unittest.TestCase):
    def test_known_overlap_times(self):
        start, end = times_from_knots([0, 1, 2])
        np.testing.assert_allclose(start, [0, 0.75])
        np.testing.assert_allclose(end, [1.25, 2])

    def test_no_interior_speed_minimum_falls_back_to_uniform(self):
        times = np.arange(5, dtype=float)
        positions = times[:, None] * np.array([1, 0, 0])
        uniform, seed, details = speed_seed_knots(times, positions, 2)
        np.testing.assert_array_equal(seed, uniform)
        self.assertTrue(details[0]['fallback_uniform'])

    def test_generated_starts_are_reproducible_and_in_bounds(self):
        times = np.linspace(0, 4, 401)
        positions = np.column_stack([np.sin(times), times, np.zeros(len(times))])
        a, _ = make_initializations(times, positions, 3, 0.02)
        b, _ = make_initializations(times, positions, 3, 0.02)
        self.assertEqual(len(a), 10)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left['activation'], right['activation'])
            self.assertTrue(left['valid'])
            self.assertTrue(np.all(np.diff(left['knots']) > 0))
            self.assertGreaterEqual(left['activation'].min(), times[0])
            self.assertLessEqual(left['completion'].max(), times[-1])

    def test_selection_uses_error_after_success_and_endpoint_checks(self):
        def result(error, success=True, final_error=0):
            return {'success': success, 'position_rmse_m': error,
                    'initial_error_m': 0, 'final_error_m': final_error,
                    'positions': np.zeros((3, 3)), 'activation': np.array([0]),
                    'completion': np.array([1]), 'displacements': np.zeros((1, 3))}
        rows = [result(0.01), result(0, success=False), result(0.001, final_error=0.1), result(0.005)]
        self.assertEqual(choose_best_result(rows), 3)
        self.assertEqual(choose_best_result([result(0.01), result(0.01)]), 0)
        with self.assertRaises(ValueError):
            choose_best_result([None, result(float('nan')), result(0, success=False)])

    def test_invalid_initialization_settings(self):
        with self.assertRaises(ValueError):
            times_from_knots([0, 0, 1])
        with self.assertRaises(ValueError):
            speed_seed_knots([0, 1, 2], np.zeros((3, 3)), 0)


if __name__ == '__main__':
    unittest.main()
