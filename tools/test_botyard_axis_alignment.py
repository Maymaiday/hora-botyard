"""Offline kinematics/regressions; NOT a physical grip test."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from botyard_axis_alignment import axis_error,alignment_plan,alignment_pose,vertical_check
from botyard_two_stage_core import transform

class AxisAlignmentTests(unittest.TestCase):
    def test_uploaded_endpoint_axis_corrected_without_moving_pivot(self):
        # Same final object tilt recorded in the uploaded 000346 run.
        q=[-.009456632658839226,-.0005542945000343025,.010027263313531876,.9999049305915833]
        object_rotation=Rotation.from_euler('x',np.pi)*Rotation.from_euler('y',np.deg2rad(8.8))
        row=dict(object7=[.01,.02,.4]+object_rotation.as_quat().tolist(),
                 hand7=[-.01,.12,.31]+Rotation.from_euler('xyz',[1.05,.01,.02]).as_quat().tolist())
        plan=alignment_plan(row,-1);base=Rotation.from_quat(row['hand7'][3:]).as_matrix()
        relative_rotation=base.T@object_rotation.as_matrix()
        for u in np.linspace(0,1,21):
            position,rpy=alignment_pose(plan,u);r=Rotation.from_euler('xyz',rpy).as_matrix()
            np.testing.assert_allclose(position+r@plan['relative'],row['object7'][:3],atol=1e-12)
        expected=r@relative_rotation
        np.testing.assert_allclose(expected[:,2],[0,0,-1],atol=1e-12)

    def test_actual_uploaded_quaternion_prediction(self):
        q=[-.01216938,-.15249059,-.98823] # measured object axis, not quaternion
        axis=np.array(q);axis/=np.linalg.norm(axis)
        rot,_=Rotation.align_vectors([axis],[[0,0,1]])
        row=dict(object7=[0,0,.4]+rot.as_quat().tolist(),hand7=[0,.1,.3,0,0,0,1])
        plan=alignment_plan(row,-1);p,r=alignment_pose(plan,1.)
        aligned=Rotation.from_euler('xyz',r)*rot
        self.assertLess(axis_error([0,0,.4]+aligned.as_quat().tolist(),-1),1e-5)

    def test_step_budget_and_world_positive_axis(self):
        rot=Rotation.from_euler('x',np.deg2rad(25))
        row=dict(object7=[0,0,.4]+rot.as_quat().tolist(),hand7=[0,.1,.3,0,0,0,1])
        plan=alignment_plan(row,1);self.assertAlmostEqual(plan['requestedAngleDeg'],10.)
        p,r=alignment_pose(plan,1.);actual=Rotation.from_euler('xyz',r)*rot
        self.assertAlmostEqual(axis_error([0,0,.4]+actual.as_quat().tolist(),1),15.)

    def test_final_hold_checks_tolerance_and_axis_sign(self):
        row=dict(simTime=1.,object7=[0,0,.4]+Rotation.from_euler('x',np.pi).as_quat().tolist())
        self.assertTrue(vertical_check([row],-1)['passed'])
        self.assertFalse(vertical_check([row],1)['passed'])
        row['object7'][3:]=(Rotation.from_euler('x',np.pi)*Rotation.from_euler('y',np.deg2rad(3))).as_quat().tolist()
        self.assertFalse(vertical_check([row],-1)['passed'])

    def test_cold_start_preserves_measured_aligned_orientation(self):
        from pathlib import Path
        s=(Path(__file__).with_name('botyard_two_stage_worker.py')).read_text()
        self.assertIn("root7=list(seed['handPosition'])+list(final['hand7'][3:])",s)
        self.assertIn('vertical=vertical_check(rows,axis_sign)',s)

if __name__=='__main__':unittest.main(verbosity=2)
