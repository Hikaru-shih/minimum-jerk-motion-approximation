import unittest

import numpy as np

from motion_decomposition.step3_speed_minima import local_minimum_runs, speed_profile


class SpeedMinimaTests(unittest.TestCase):
    def test_speed_is_vector_norm(self):
        times = np.linspace(0, 2, 21)
        positions = 0.5 * times[:, None]**2 * np.array([3, -4, 0])
        velocity, speed = speed_profile(positions, times)
        np.testing.assert_allclose(velocity, times[:, None] * [3, -4, 0], atol=1e-12)
        np.testing.assert_allclose(speed, 5 * times, atol=1e-12)

    def test_plateaus_zero_minima_and_endpoints(self):
        actual = local_minimum_runs([0, 3, 1, 1, 2, 4, 0, 3, 0])
        np.testing.assert_array_equal(actual, [[2, 3], [6, 6]])

    def test_constant_and_monotone_arrays_have_no_interior_minima(self):
        for speed in ([0, 0, 0, 0], [2, 2, 2], [0, 1, 2, 3], [3, 2, 1, 0]):
            with self.subTest(speed=speed):
                self.assertEqual(local_minimum_runs(speed).shape, (0, 2))

    def test_single_minimum_jerk_speed_has_no_interior_minimum(self):
        u = np.linspace(0, 1, 101)
        speed = 30 * u**2 * (1 - u)**2
        self.assertEqual(local_minimum_runs(speed).shape, (0, 2))

    def test_two_movements_with_exact_pause_have_one_minimum_run(self):
        times = np.arange(-50, 351) / 100
        speed = np.zeros(len(times))
        for start in (0, 2):
            u = np.clip(times - start, 0, 1)
            speed += 30 * u**2 * (1 - u)**2
        runs = local_minimum_runs(speed)
        self.assertEqual(runs.shape, (1, 2))
        np.testing.assert_allclose(times[runs[0]], [1, 2], atol=1e-12)

    def test_invalid_data_rejected(self):
        for speed in ([0, -1, 0], [0, np.nan, 0], [0, 1], [[0, 1, 0]]):
            with self.subTest(speed=speed):
                with self.assertRaises(ValueError):
                    local_minimum_runs(speed)
        for positions, times in [
            (np.zeros((3, 3)), [0, 0, 1]),
            (np.zeros((3, 2)), [0, 1, 2]),
            (np.full((3, 3), np.nan), [0, 1, 2]),
        ]:
            with self.assertRaises(ValueError):
                speed_profile(positions, times)


if __name__ == '__main__':
    unittest.main()
