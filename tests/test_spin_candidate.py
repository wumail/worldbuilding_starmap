from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'design'))
import terrax_spin_candidate as model

class SpinCandidateTests(unittest.TestCase):
    def test_moon_inclinations_are_relative_to_the_initial_equator(self):
        y=model.initial();p=y[:30].reshape(10,3);v=y[30:60].reshape(10,3);spin=y[60:]
        for i,expected in [(8,.4),(9,1.1)]:
            normal=np.cross(p[i]-p[model.HOME],v[i]-v[model.HOME])
            inclination=np.degrees(np.arctan2(np.linalg.norm(np.cross(normal,spin)),normal@spin))
            self.assertAlmostEqual(inclination,expected,places=9)

    def test_total_force_and_orbital_plus_spin_torque_cancel(self):
        y=model.initial();dy=model.rhs(0,y);p=y[:30].reshape(10,3);a=dy[30:60].reshape(10,3)
        force=a*model.MASS[:,None]
        self.assertLess(np.linalg.norm(force.sum(axis=0))/np.linalg.norm(force),1e-14)
        torques=np.cross(p,force);total=torques.sum(axis=0)+model.SPIN*dy[60:]
        self.assertLess(np.linalg.norm(total)/np.linalg.norm(torques),1e-12)

    def test_quadrupole_acceleration_is_negative_potential_gradient(self):
        spin=np.array([.2,-.3,.8]);spin/=np.linalg.norm(spin)
        for vector in [np.array([.01,.02,.03]),np.array([.002,-.001,.0001])]:
            p=np.tile([1.,0.,0.],(10,1));p[model.HOME]=0;p[0]=vector;_,aq=model.quadrupole(p,spin)
            def potential(r):
                distance=np.linalg.norm(r)
                return model.G*model.MASS[model.HOME]*model.J2*(model.R/model.AU)**2/(2*distance**3)*(3*(r@spin/distance)**2-1)
            step=np.linalg.norm(vector)*1e-5
            gradient=np.array([(potential(vector+np.eye(3)[i]*step)-potential(vector-np.eye(3)[i]*step))/(2*step) for i in range(3)])
            self.assertLess(np.linalg.norm(aq[0]+gradient)/np.linalg.norm(aq[0]),1e-8)

if __name__=='__main__':unittest.main()
