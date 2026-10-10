"""CPU-only grasp validation; intentionally independent of Isaac Gym/torch.

v3 uses measured pose motion to accept settled grasps; reported SDK velocities
are retained as diagnostics. This is a changed acceptance protocol, not a claim
that earlier SDK angular-velocity thresholds passed.
"""
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ACTUAL = [f+j for f in ('FF','LF','MF','RF') for j in ('J4','J3','J2','J1')] + ['THJ4','THJ3','THJ2','THJ1']
POLICY = [n for n in ACTUAL if n not in ('FFJ1','LFJ1','MFJ1','RFJ1')]
FINGERS = ('ff','lf','mf','rf','th')
VALIDATION_VERSION = 'five_side_pose_motion_v3'

def expand_delta(delta, names):
    values = dict(zip(POLICY, delta))
    for f in ('FF','LF','MF','RF'):
        values[f+'J1'] = values[f+'J2']
    return np.asarray([values[n] for n in names])

def world_object(root7, relative7):
    r = Rotation.from_quat(root7[3:])
    return np.r_[np.asarray(root7[:3])+r.apply(relative7[:3]),
                 (r*Rotation.from_quat(relative7[3:])).as_quat()]

def collision_masks(names, pairs):
    if len(pairs)>30: raise ValueError('Too many named collision pairs')
    masks=dict.fromkeys(names,0); requested=set()
    for i,(a,b) in enumerate(pairs):
        pair=frozenset((a,b))
        if a==b or a not in masks or b not in masks or pair in requested:
            raise ValueError('Invalid collision pair: '+str((a,b)))
        requested.add(pair);masks[a]|=1<<i;masks[b]|=1<<i
    actual={frozenset((a,b)) for a,b in itertools.combinations(names,2) if masks[a]&masks[b]}
    if actual!=requested: raise ValueError('Unintended self-collision exclusion')
    return masks

def apply_collision_masks(gym, env, actor, config):
    names=list(gym.get_actor_rigid_body_names(env,actor))
    masks=collision_masks(names,config['pairs'])
    spans=gym.get_actor_rigid_body_shape_indices(env,actor)
    shapes=gym.get_actor_rigid_shape_properties(env,actor);expected=[None]*len(shapes)
    if len(spans)!=len(names): raise RuntimeError('Body/shape map mismatch')
    for name,span in zip(names,spans):
        for i in range(span.start,span.start+span.count):
            if i>=len(shapes) or expected[i] is not None: raise RuntimeError('Invalid shape range')
            shapes[i].filter=masks[name];expected[i]=masks[name]
    if any(v is None for v in expected): raise RuntimeError('Unmapped collision shape')
    gym.set_actor_rigid_shape_properties(env,actor,shapes)
    if [int(s.filter) for s in gym.get_actor_rigid_shape_properties(env,actor)]!=expected:
        raise RuntimeError('Collision mask readback failed')
    return expected

def field(record,name):
    norm=lambda s:s.replace('_','').lower()
    for key in record.dtype.names or ():
        if norm(key)==norm(name): return record[key]
    raise ValueError('Unknown contact record field '+name)

def side_contacts(contacts, object_body, finger_bodies, scale):
    touching=set();support=[];caps=0
    for c in contacts:
        b0,b1=int(field(c,'body0')),int(field(c,'body1'))
        if object_body not in (b0,b1) or float(field(c,'lambda'))<=0: continue
        other=b1 if b0==object_body else b0
        if other not in finger_bodies:
            support.append(other);continue
        p=field(c,'localPos0' if b0==object_body else 'localPos1')
        x,y,z=[float(p[k]) for k in ('x','y','z')]
        if abs(z)<.04*scale-.002 and abs(np.hypot(x,y)-.04*scale)<.005:
            touching.add(finger_bodies[other])
        else: caps+=1
    return dict(fingers=sorted(touching),support=sorted(set(support)),capContacts=caps)

def frame(task, index=0, refresh=True):
    """Call only AFTER simulate + fetch_results + refresh, once per physics step."""
    if refresh:task._refresh_gym()
    pose=task.object_pose[index].detach().cpu().numpy().astype(float)
    pose[:3]-=task.state_offsets[index].cpu().numpy()
    contact=side_contacts(task.gym.get_env_rigid_contacts(task.envs[index]),
                          task.object_body_handle,task.finger_body_groups,float(task.env_scales_np[index]))
    q=task.allegro_hand_dof_pos[index].cpu().numpy()
    lo=task.allegro_hand_dof_lower_limits.cpu().numpy();hi=task.allegro_hand_dof_upper_limits.cpu().numpy()
    return dict(object7=pose.tolist(),contacts=contact,
                sdkVelocity6=np.r_[task.object_linvel[index].cpu().numpy(),task.object_angvel[index].cpu().numpy()].tolist(),
                jointViolation=float(np.maximum(lo-q,q-hi).clip(0).max()) if np.isfinite(q).all() else float('inf'))

def evaluate(rows, start7, scale, dt, warmup_seconds=.4, max_tilt=2., max_drift=.008, max_drop=.005):
    if not rows: return dict(passed=False,reasons=['no_physics_samples'])
    poses=np.asarray([r['object7'] for r in rows]);start=np.asarray(start7)
    if not np.isfinite(poses).all(): return dict(passed=False,reasons=['nonfinite_pose'])
    if not np.allclose(np.linalg.norm(poses[:,3:],axis=1),1.,atol=1e-3):
        return dict(passed=False,reasons=['invalid_object_quaternion'])
    times=np.arange(1,len(rows)+1)*dt;after=times>warmup_seconds+1e-9
    axis=Rotation.from_quat(poses[:,3:]).apply([0,0,1])
    tilts=np.degrees(np.arccos(np.clip(np.abs(axis[:,2]),0,1)))
    drift=np.linalg.norm(poses[:,:3]-start[:3],axis=1)
    bottom=poses[:,2]-(.04*scale*np.abs(axis[:,2])+.04*scale*np.linalg.norm(axis[:,:2],axis=1))
    reasons=[]
    if tilts.max()>max_tilt: reasons.append('cylinder_tilt')
    if drift.max()>max_drift: reasons.append('object_drift')
    if np.max(start[2]-poses[:,2])>max_drop or bottom.min()<.03: reasons.append('object_drop_or_ground')
    if any(r['jointViolation']>.01 for r in rows): reasons.append('joint_limit')
    check=[r for r,active in zip(rows,after) if active]
    if not check: reasons.append('no_samples_after_warmup')
    if any(set(r['contacts']['fingers'])!=set(FINGERS) for r in check): reasons.append('missing_finger_side_contact')
    if any(r['contacts']['support'] or r['contacts']['capContacts'] for r in check): reasons.append('palm_ground_or_cap_support')
    metrics=dict(maxTiltDeg=float(tilts.max()),maxDriftM=float(drift.max()),minBottomZ=float(bottom.min()))
    if len(rows)>1:
        lin=np.linalg.norm(np.diff(poses[:,:3],axis=0)/dt,axis=1)
        rotations=Rotation.from_quat(poses[:,3:]);ang=(rotations[1:]*rotations[:-1].inv()).magnitude()/dt
        tail=(times[1:]>=times[-1]-1.) & (times[1:]>warmup_seconds)
        if not tail.any(): reasons.append('no_settle_samples')
        else:
            if lin[tail].max()>.03 or ang[tail].max()>.3: reasons.append('pose_motion_not_settled')
            pt=poses[1:][tail];rot=Rotation.from_quat(pt[:,3:])
            pd=np.linalg.norm(pt[:,:3]-pt[0,:3],axis=1).max()
            rd=np.degrees((rot*rot[0].inv()).magnitude()).max()
            if pd>.001 or rd>1.: reasons.append('tail_pose_drift')
            sdk=np.asarray([r['sdkVelocity6'] for r in rows])[1:][tail]
            metrics.update(maxPoseLinearSpeed=float(lin[tail].max()),maxPoseAngularSpeed=float(ang[tail].max()),
                           tailPositionDriftM=float(pd),tailOrientationDriftDeg=float(rd),
                           maxSDKLinearSpeed=float(np.linalg.norm(sdk[:,:3],axis=1).max()),
                           maxSDKAngularSpeed=float(np.linalg.norm(sdk[:,3:],axis=1).max()))
    else: reasons.append('insufficient_motion_samples')
    return dict(passed=not reasons,reasons=reasons,metrics=metrics,validationVersion=VALIDATION_VERSION,
                sdkVelocityIsDiagnosticOnly=True,physicsSteps=len(rows))

def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)
