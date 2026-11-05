import unittest

import numpy as np

from motion_decomposition.joint_fit import fit_joint
from motion_decomposition.opposition import opposition_residuals
from motion_decomposition.step1_known_motion import generate_motion


class JointFitTests(unittest.TestCase):
    def test_known_nonopposing_motion_recovered_with_and_without_penalty(self):
        t = np.linspace(0, 3.4, 205)
        d = np.array([[0.3, 0.1, 0], [0.1, 0.2, 0.05]])
        p = generate_motion(t, np.zeros(3), d, [0.2, 1.2], [2.2, 3.2])
        for weight in [0, 1e-4]:
            fit = fit_joint(t, p, p[0], p[-1], [0.1, 1.3], [2.1, 3.3], 0.02, weight, 0.3)
            self.assertTrue(fit['success'])
            self.assertLess(fit['position_rmse_m'], 1e-8)
            np.testing.assert_allclose(fit['activation'], [0.2, 1.2], atol=1e-5)
            np.testing.assert_allclose(fit['completion'], [2.2, 3.2], atol=1e-5)
            np.testing.assert_allclose(fit['displacements'], d, atol=1e-7)

    def test_objective_endpoints_bounds_and_parameter_count(self):
        t = np.linspace(0, 3, 151)
        p = generate_motion(t, np.zeros(3), [[0.4,0,0],[-0.1,0,0]], [0.2,0.4], [2.2,2.4])
        weight, scale = 1e-4, 0.3
        fit = fit_joint(t, p, p[0], p[-1], [0.1,0.5], [2.0,2.6], 0.02, weight, scale)
        penalty = opposition_residuals(t, fit['activation'], fit['completion'], fit['displacements'], scale)
        expected = np.mean(np.sum((fit['positions']-p)**2, axis=1))+weight*np.sum(penalty**2)/len(t)
        self.assertAlmostEqual(fit['objective_m2'], expected, places=14)
        self.assertLessEqual(fit['objective_m2'], fit['initial_objective_m2']+1e-12)
        np.testing.assert_allclose(fit['positions'][[0,-1]], p[[0,-1]], atol=1e-12)
        self.assertTrue(np.all(fit['activation'] >= t[0]))
        self.assertTrue(np.all(fit['completion'] <= t[-1]))
        self.assertTrue(np.all(fit['completion']-fit['activation'] >= 0.02-1e-12))
        self.assertEqual(fit['parameter_count'], 7)

    def test_one_component_still_fits_time_with_fixed_displacement(self):
        t = np.linspace(0, 3, 151)
        p = generate_motion(t, np.zeros(3), [[0.3,0,0]], [0.3], [2.5])
        fit = fit_joint(t, p, p[0], p[-1], [0.1], [2.8], 0.02, 1, 0.3)
        self.assertLess(fit['position_rmse_m'], 1e-8)
        self.assertEqual(fit['parameter_count'], 2)
        self.assertEqual(fit['opposition_cost_m2'], 0)

    def test_invalid_weight_scale_and_times_rejected(self):
        for weight, scale in [(-1,0.3), (np.nan,0.3), (1,0), (1,np.inf)]:
            with self.assertRaises(ValueError):
                fit_joint([0,1,2], np.zeros((3,3)), [0,0,0], [0,0,0], [0], [2], .02, weight, scale)
        with self.assertRaises(ValueError):
            fit_joint([0,1,2], np.zeros((3,3)), [0,0,0], [0,0,0], [0], [2], 3, 0, .3)
