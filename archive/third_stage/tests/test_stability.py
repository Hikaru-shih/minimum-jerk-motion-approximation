import unittest

import numpy as np

from motion_decomposition.fixed_times import design_matrix
from motion_decomposition.stability import align_components, compare_aligned


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.times = np.linspace(0,3,301)
        self.reference = {'activation':np.array([0.1,1.2]),'completion':np.array([1.8,2.9]),
                          'displacements':np.array([[.3,.1,0],[-.1,.2,.05]])}
        self.reference['positions'] = design_matrix(self.times,self.reference['activation'],self.reference['completion']) @ self.reference['displacements']

    def test_label_swap_is_not_instability(self):
        candidate = {k:v[::-1].copy() if k != 'positions' else v.copy() for k,v in self.reference.items()}
        result = compare_aligned(self.times,self.reference,candidate)
        np.testing.assert_array_equal(result['alignment'],[1,0])
        for key,value in result.items():
            if key != 'alignment':
                self.assertAlmostEqual(value,0,places=12)

    def test_changed_components_can_have_identical_total_trajectory(self):
        reference = {'activation':np.array([.1,.1]),'completion':np.array([2.9,2.9]),
                     'displacements':np.array([[.3,0,0],[.1,0,0]])}
        reference['positions'] = design_matrix(self.times,reference['activation'],reference['completion']) @ reference['displacements']
        candidate = {k:v.copy() for k,v in reference.items()}
        candidate['displacements'] = np.array([[.5,0,0],[-.1,0,0]])
        result = compare_aligned(self.times,reference,candidate)
        self.assertEqual(result['trajectory_difference_rmse_m'],0)
        self.assertGreater(result['velocity_signature_rmse_m_s'],0)
        self.assertGreater(result['maximum_displacement_vector_difference_m'],.19)

    def test_component_count_mismatch_rejected(self):
        candidate = {k:v[:1] for k,v in self.reference.items()}
        with self.assertRaises(ValueError):
            align_components(self.times,self.reference,candidate)
