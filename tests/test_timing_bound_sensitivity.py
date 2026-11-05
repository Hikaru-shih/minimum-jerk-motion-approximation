import unittest
import numpy as np
from motion_decomposition.fixed_times import design_matrix
from motion_decomposition.timing_bound_sensitivity import fit_variant
from motion_decomposition.valley_fit import fit_valley_units


class TimingBoundSensitivityTests(unittest.TestCase):
    def test_nominal_matches_existing_fitter(self):
        t=np.linspace(0,3,121)
        p=design_matrix(t,[.25],[2.75])@np.array([[.3,.1,.05]])
        previous=fit_valley_units(t,p,[0,3])
        current=fit_variant(t,p,[0,3],.45,.20)
        np.testing.assert_allclose(current['best']['positions'],previous['best']['positions'],atol=1e-12)
        self.assertLess(current['best']['position_rmse_m'],1e-8)

    def test_invalid_inward_fraction_rejected(self):
        t=np.linspace(0,3,121)
        with self.assertRaises(ValueError):fit_variant(t,np.zeros((121,3)),[0,3],.45,.6)


if __name__=='__main__':unittest.main()
