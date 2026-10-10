"""Measured cylinder axis correction using wrist motion about object center.

No simulator state edits. A symmetric cylinder's axis line may point along
world +Z or -Z; keep the post-flip sign to preserve the palm-up grasp.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def axis_error(object7, sign):
    axis=Rotation.from_quat(object7[3:]).as_matrix()[:,2]
    return float(np.degrees(np.arccos(np.clip(sign*axis[2],-1,1))))


def alignment_plan(row, sign, maximum_step_deg=10.):
    axis=Rotation.from_quat(row['object7'][3:]).as_matrix()[:,2]
    goal=np.array([0.,0.,float(sign)])
    angle=np.arccos(np.clip(axis@goal,-1,1))
    cross=np.cross(axis,goal);norm=np.linalg.norm(cross)
    if norm<1e-10:
        if angle>1e-6:raise ValueError('Antiparallel correction would change grasp orientation by 180 degrees')
        vector=np.zeros(3)
    else:vector=cross/norm*min(angle,np.deg2rad(maximum_step_deg))
    base_rotation=Rotation.from_quat(row['hand7'][3:]).as_matrix()
    pivot=np.array(row['object7'][:3]);base_position=np.array(row['hand7'][:3])
    relative=base_rotation.T@(pivot-base_position)
    return dict(vector=vector,pivot=pivot,baseRotation=base_rotation,relative=relative,
                requestedAngleDeg=float(np.degrees(np.linalg.norm(vector))))


def alignment_pose(plan,u):
    rotation=Rotation.from_rotvec(plan['vector']*u).as_matrix()@plan['baseRotation']
    position=plan['pivot']-rotation@plan['relative']
    return position,Rotation.from_matrix(rotation).as_euler('xyz')


def vertical_check(rows,sign,tolerance_deg=2.):
    if not rows:return dict(passed=False,reasons=['no_vertical_hold_samples'])
    tail=[r for r in rows if r['simTime']>=rows[-1]['simTime']-1.]
    errors=[axis_error(r['object7'],sign) for r in tail]
    return dict(passed=max(errors)<=tolerance_deg,reasons=[] if max(errors)<=tolerance_deg else ['final_cylinder_axis_not_vertical'],
                axisSign=sign,toleranceDeg=tolerance_deg,maxTailErrorDeg=max(errors),finalErrorDeg=errors[-1],
                convention='axis line aligned to world Z; post-flip local +Z usually points toward world -Z')
