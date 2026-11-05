import unittest

import numpy as np

from motion_decomposition.opposition import opposition_residuals


class PenaltyScalingTests(unittest.TestCase):
    def setUp(self):
        self.t = np.linspace(0,3,301)
        self.a = np.array([.2,.4])
        self.b = np.array([2.2,2.4])
        self.d = np.array([[.4,.1,0],[-.1,0,.01]])

    def test_length_scale_and_weight_are_coupled(self):
        r = opposition_residuals(self.t,self.a,self.b,self.d,.3)
        changed = opposition_residuals(self.t,self.a,self.b,self.d,.6)
        np.testing.assert_allclose(4*changed**2,r**2,atol=1e-12)

    def test_change_of_length_units_scales_residual_like_position(self):
        r = opposition_residuals(self.t,self.a,self.b,self.d,.3)
        changed = opposition_residuals(self.t,self.a,self.b,100*self.d,30)
        np.testing.assert_allclose(changed,100*r,atol=1e-10)

    def test_time_units_and_time_origin_do_not_change_penalty(self):
        r = opposition_residuals(self.t,self.a,self.b,self.d,.3)
        changed = opposition_residuals(1000*self.t+500,1000*self.a+500,1000*self.b+500,self.d,.3)
        np.testing.assert_allclose(changed,r,atol=1e-12)
