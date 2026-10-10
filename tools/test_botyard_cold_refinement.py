"""Geometry only. This test does not execute Isaac Gym or prove stable grasping."""
import copy,json,unittest,zipfile,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from refine_botyard_cold_grasp import correction,proposal,eligible

class Geometry(unittest.TestCase):
    def test_axis_signs_and_limit(self):
        for sign in [-1,1]:
            axis=Rotation.from_euler('x',6,degrees=True).apply([0,0,sign]);r,angle=correction(axis)
            np.testing.assert_allclose(r.apply(axis),[0,0,sign],atol=1e-10);self.assertAlmostEqual(angle,6)
        axis=Rotation.from_euler('x',12,degrees=True).apply([0,0,1]);r,angle=correction(axis)
        self.assertAlmostEqual(angle,8)
        self.assertAlmostEqual(np.rad2deg(np.arccos(r.apply(axis)[2])),4)
    def test_proposal_preserves_contact_geometry_and_actual_angles(self):
        root=[0,0,.5,0,0,0,1];object_q=Rotation.from_euler('x',6,degrees=True).as_quat().tolist()
        row=dict(q20=list(range(20)),relativeObject7=[.01,.02,.2]+object_q,object7=[.01,.02,.7]+object_q)
        data=dict(completed=True,stage2=dict(passed=True),coldStart=dict(handRoot7=root,frames=[row],reasons=['cylinder_tilt']),
                  poseColdStartCheck=dict(reasons=['cylinder_tilt'],metrics=dict(maxTiltDeg=6,maxDriftM=.001,
                    maxPoseLinearSpeed=.001,maxPoseAngularSpeed=.01,tailPositionDriftM=.0001,tailOrientationDriftDeg=.5)),candidate=dict(close16=list(range(16))))
        self.assertTrue(eligible(data));out,angle=proposal(data)
        self.assertEqual(out['final']['q20'],row['q20']);self.assertEqual(out['candidate'],data['candidate'])
        self.assertEqual(out['final']['relativeObject7'],row['relativeObject7'])
        np.testing.assert_allclose(Rotation.from_quat(out['final']['object7'][3:]).apply([0,0,1]),[0,0,1],atol=1e-10)
        self.assertFalse(out['accepted']);self.assertFalse(out['completed'])
        bad=copy.deepcopy(data);bad['coldStart']['reasons'].append('missing_finger_side_contact');self.assertFalse(eligible(bad))
    def test_user_residual_drift_can_continue_search_without_becoming_accepted(self):
        data=dict(completed=True,accepted=False,stage2=dict(passed=True),coldStart=dict(frames=[{}],reasons=['cylinder_tilt','tail_pose_drift']),
            poseColdStartCheck=dict(passed=False,reasons=['cylinder_tilt','tail_pose_drift'],metrics=dict(
                maxTiltDeg=4.0603685075354665,maxDriftM=.001676315622064278,
                maxPoseLinearSpeed=.0011865045699864874,maxPoseAngularSpeed=.031077143223164537,
                tailPositionDriftM=.0006848489756285664,tailOrientationDriftDeg=1.2977183799732182)))
        self.assertTrue(eligible(data));self.assertFalse(data['accepted']);self.assertFalse(data['poseColdStartCheck']['passed'])
        bad=copy.deepcopy(data);bad['poseColdStartCheck']['metrics']['tailOrientationDriftDeg']=3.;self.assertFalse(eligible(bad))
        bad=copy.deepcopy(data);bad['coldStart']['reasons'].append('support_contact');self.assertFalse(eligible(bad))
        bad=copy.deepcopy(data);bad['poseColdStartCheck']['metrics']['maxPoseAngularSpeed']=float('nan');self.assertFalse(eligible(bad))

if __name__=='__main__':unittest.main(verbosity=2)
