import unittest
import numpy as np

from motion_decomposition.fixed_times import design_matrix, fit_fixed_endpoints
from motion_decomposition.velocity_objective import fit_joint_fixed, joint_residual, scales


class VelocityObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.t=np.linspace(0,3,91)
        self.a,self.b=np.array([.1,1.2]),np.array([1.8,2.9])
        self.d=np.array([[.3,.1,-.05],[-.12,.2,.08]])
        self.p=design_matrix(self.t,self.a,self.b)@self.d

    def test_exact_recovery_and_endpoints(self):
        length,speed=scales(self.t,self.p)
        f=fit_joint_fixed(self.t,self.p,self.a,self.b,1,length,speed)
        np.testing.assert_allclose(f['displacements'],self.d,atol=1e-12)
        np.testing.assert_allclose(f['positions'][[0,-1]],self.p[[0,-1]],atol=1e-12)

    def test_zero_weight_matches_previous_linear_fit(self):
        f=fit_joint_fixed(self.t,self.p,self.a,self.b,0,*scales(self.t,self.p))
        old=fit_fixed_endpoints(self.t,self.p,self.p[0],self.p[-1],self.a,self.b)
        np.testing.assert_array_equal(f['positions'],old['positions'])

    def test_joint_solution_minimizes_actual_joint_residual(self):
        p=self.p.copy();p[:,0]+=.02*np.sin(8*self.t)
        length,speed=scales(self.t,p)
        a,b=np.array([0,1.5]),np.array([1.5,3])
        old=fit_joint_fixed(self.t,p,a,b,0,length,speed)
        new=fit_joint_fixed(self.t,p,a,b,1,length,speed)
        cost=lambda f:np.sum(joint_residual(self.t,p,f['positions'],1,length,speed)**2)
        self.assertLessEqual(cost(new),cost(old)+1e-12)
        basis=design_matrix(self.t,a,b)
        for step in (-.001,.001):
            d=new['displacements'].copy();d[0,0]+=step;d[1,0]-=step
            self.assertGreater(cost({'positions':p[0]+basis@d}),cost(new))

    def test_single_component_and_invalid_weight(self):
        f=fit_joint_fixed(self.t,self.p,[0],[3],1,*scales(self.t,self.p))
        np.testing.assert_allclose(f['positions'][[0,-1]],self.p[[0,-1]],atol=1e-12)
        with self.assertRaises(ValueError):
            fit_joint_fixed(self.t,self.p,self.a,self.b,-1,*scales(self.t,self.p))


if __name__=='__main__':unittest.main()
