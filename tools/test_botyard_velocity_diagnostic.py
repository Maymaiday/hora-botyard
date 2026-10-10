"""Synthetic signal checks, not Isaac Gym validation."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from botyard_velocity_diagnostic import analyze_velocity

class VelocityDiagnosticTests(unittest.TestCase):
    def rows(self,speed=.5,moving=False,alternate=False,dt=1/240):
        rows=[];angle=0.
        for i in range(300):
            omega=speed*(-1 if alternate and i%2 else 1)
            if moving:angle+=omega*dt
            rows.append(dict(simTime=i*dt,object7=[0,0,.5]+Rotation.from_rotvec([0,0,angle]).as_quat().tolist(),
                             objectVelocity6=[0,0,0,0,0,omega]))
        return rows
    def test_constant_reported_velocity_with_fixed_pose(self):
        d=analyze_velocity(self.rows(),1/240)['summary']
        self.assertEqual(d['classification'],'reported_velocity_pose_mismatch_at_physics_step_resolution')
        self.assertAlmostEqual(d['poseAngularRmsRadS'],0.)
        self.assertFalse(d['acceptanceThresholdsChanged'])
    def test_real_rotation_detected(self):
        d=analyze_velocity(self.rows(moving=True),1/240)['summary']
        self.assertEqual(d['classification'],'pose_changes_show_motion_above_angular_limit')
        self.assertAlmostEqual(d['poseAngularRmsRadS'],.5)
    def test_alternating_motion_not_mislabeled_stationary(self):
        d=analyze_velocity(self.rows(moving=True,alternate=True),1/240)['summary']
        self.assertEqual(d['classification'],'pose_changes_show_oscillatory_motion')
    def test_sparse_logs_cannot_resolve_aliasing(self):
        d=analyze_velocity(self.rows(dt=.05),1/240)['summary']
        self.assertFalse(d['perPhysicsStepSamples'])
        self.assertEqual(d['classification'],'sparse_samples_cannot_identify_between_sample_motion')
    def test_quaternion_sign_changes_not_false_rotation(self):
        rows=self.rows(speed=0.)
        for i,row in enumerate(rows):
            if i%2:row['object7'][3:]=(-np.array(row['object7'][3:])).tolist()
        self.assertAlmostEqual(analyze_velocity(rows,1/240)['summary']['poseAngularMaxRadS'],0.)

if __name__=='__main__':unittest.main(verbosity=2)
