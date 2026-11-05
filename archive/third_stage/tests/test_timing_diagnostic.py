import unittest

import numpy as np

from motion_decomposition.step4_diagnose_initialization import component_velocities, composition_metrics


class TimingDiagnosticTests(unittest.TestCase):
    def test_known_midpoint_velocity_and_zero_outside(self):
        actual = component_velocities([-1, 0, 1, 2, 3], [0], [2], [[2, 0, 0]])
        np.testing.assert_allclose(actual[:, 0, 0], [0, 0, 1.875, 0, 0], atol=1e-12)

    def test_exact_opposites_cancel(self):
        times = np.linspace(0, 1, 1001)
        velocity = component_velocities(times, [0, 0], [1, 1], [[1, 0, 0], [-1, 0, 0]])
        metrics = composition_metrics(times, velocity)
        np.testing.assert_allclose(velocity.sum(axis=1), 0, atol=1e-12)
        self.assertAlmostEqual(metrics['sum_component_path_lengths_m'], 2, places=9)
        self.assertAlmostEqual(metrics['relative_component_length_excess'], 1)

    def test_length_excess_does_not_imply_opposing_directions(self):
        times = np.linspace(0, 1, 1001)
        velocity = component_velocities(times, [0, 0], [1, 1], [[1, 0, 0], [0, 1, 0]])
        metrics = composition_metrics(times, velocity)
        self.assertGreater(metrics['relative_component_length_excess'], 0)
        self.assertEqual(metrics['opposing_velocity_sample_count'], 0)


if __name__ == '__main__':
    unittest.main()
