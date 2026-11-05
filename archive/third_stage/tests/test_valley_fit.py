import unittest
import numpy as np

from motion_decomposition.fixed_times import design_matrix
from motion_decomposition.goal_count_selection import detection_speed, speed_valley_knots
from motion_decomposition.valley_fit import fit_valley_units


class ValleyFitTests(unittest.TestCase):
    def test_single_component_count_and_recovery(self):
        t=np.linspace(0,3,361)
        d=np.array([[.3,.1,.05]])
        p=design_matrix(t,[.25],[2.75])@d
        detected=speed_valley_knots(t,detection_speed(t,p,17))
        self.assertEqual(detected['component_count'],1)
        fit=fit_valley_units(t,p,detected['knots'],starts=2)['best']
        self.assertIsNotNone(fit)
        np.testing.assert_allclose(fit['positions'],p,atol=1e-8)
        np.testing.assert_allclose(fit['activation'],[.25],atol=1e-6)
        np.testing.assert_allclose(fit['completion'],[2.75],atol=1e-6)

    def test_associated_overlapping_components(self):
        t=np.linspace(0,3,361)
        a,b=np.array([.1,1.3]),np.array([1.7,2.9])
        d=np.array([[.3,.1,-.05],[-.12,.2,.08]])
        p=design_matrix(t,a,b)@d
        result=fit_valley_units(t,p,[0,1.5,3],starts=2)
        fit=result['best']
        self.assertIsNotNone(fit)
        np.testing.assert_allclose(fit['positions'],p,atol=1e-7)
        np.testing.assert_allclose(fit['displacements'],d,atol=1e-6)
        self.assertLess(fit['activation'][1],fit['completion'][0])
        np.testing.assert_allclose(fit['positions'][[0,-1]],p[[0,-1]],atol=1e-12)

    def test_invalid_partition_rejected(self):
        t=np.linspace(0,3,361)
        with self.assertRaises(ValueError):fit_valley_units(t,np.zeros((361,3)),[0,2,1,3])


if __name__=='__main__':unittest.main()
