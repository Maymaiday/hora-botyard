#!/usr/bin/env python3
"""Isolated Isaac Gym Preview 4 worker. Only launch through the supervisor."""
print('[PHASE] importing Isaac Gym',flush=True)
from isaacgym import gymapi
import argparse,json,os,sys,time,traceback,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from botyard_two_stage_core import (ACTUAL,POLICY,WRIST,Geometry,transform,pose7,expand,smooth,make_wrist_urdf,sample_candidate,contact_summary,hold_check,atomic_json,wrist_coordinates,upright_start_check)

from botyard_grip_control import collision_masks, tracking_check, ContactCloser, closed_grip_check, forefinger_target

from botyard_axis_alignment import axis_error, alignment_plan, alignment_pose, vertical_check
from botyard_hold_recovery import HoldRecovery

def xyz(v):return [float(v[n]) for n in ('x','y','z')]
def quat(v):return [float(v[n]) for n in ('x','y','z','w')]
def sdk_pose(p):
    t=gymapi.Transform();t.p=gymapi.Vec3(*p[:3]);t.r=gymapi.Quat(*p[3:]);return t

def single_env_state_offset(origin):
    # This standalone tool creates exactly ONE environment at the global origin.
    # ENV_SPACE and GLOBAL_SPACE translations therefore coincide. Do not infer a
    # coordinate transform from an unstepped dynamic object's body-state buffer.
    origin=np.asarray(origin,dtype=float)
    if origin.shape!=(3,) or not np.isfinite(origin).all() or np.linalg.norm(origin)>1e-6:
        raise RuntimeError('Single-environment tool requires origin [0,0,0], got '+str(origin))
    return np.zeros(3)

REQUIRED_GYM_APIS = ('add_ground', 'apply_actor_dof_efforts', 'create_actor', 'create_env', 'create_sim', 'create_viewer', 'draw_viewer', 'fetch_results', 'find_actor_rigid_body_handle', 'get_actor_dof_names', 'get_actor_dof_properties', 'get_actor_dof_states', 'get_actor_rigid_body_names', 'get_actor_rigid_body_properties', 'get_actor_rigid_body_shape_indices', 'get_actor_rigid_body_states', 'get_actor_rigid_shape_properties', 'get_env_origin', 'get_env_rigid_contacts', 'load_asset', 'prepare_sim', 'query_viewer_has_closed', 'set_actor_dof_position_targets', 'set_actor_dof_properties', 'set_actor_dof_states', 'set_actor_rigid_body_properties', 'set_actor_rigid_shape_properties', 'set_actor_scale', 'simulate', 'step_graphics', 'sync_frame_time', 'viewer_camera_look_at')

def check_gym_api(gym):
    missing=[name for name in REQUIRED_GYM_APIS if not callable(getattr(gym,name,None))]
    if missing:raise RuntimeError("Isaac Gym API missing: "+str(missing))
    print("[API CHECK] "+str(len(REQUIRED_GYM_APIS))+" required Gym methods available",flush=True)

class Scene:
    def __init__(self,a,hand_path,fixed,root7,object7):
        self.a=a;self.fixed=fixed;self.root7=root7;self.object_only=getattr(a,'object_only',False);self.anchor=transform(root7[:3],quat=root7[3:]);self.previous_wrist_rpy=np.zeros(3);self.gym=gymapi.acquire_gym();g=self.gym
        check_gym_api(g)
        params=gymapi.SimParams();params.dt=a.dt;params.substeps=1;params.up_axis=gymapi.UP_AXIS_Z
        params.gravity=gymapi.Vec3(0,0,-9.81);params.use_gpu_pipeline=False
        params.physx.use_gpu=False;params.physx.solver_type=1;params.physx.num_threads=4
        params.physx.num_position_iterations=8;params.physx.num_velocity_iterations=4
        params.physx.contact_offset=.002;params.physx.rest_offset=0.
        params.physx.contact_collection=gymapi.ContactCollection.CC_ALL_SUBSTEPS
        self.sim=g.create_sim(0,0 if a.viewer else -1,gymapi.SIM_PHYSX,params)
        if self.sim is None:raise RuntimeError('create_sim failed')
        plane=gymapi.PlaneParams();plane.normal=gymapi.Vec3(0,0,1);plane.static_friction=a.friction;plane.dynamic_friction=a.friction;g.add_ground(self.sim,plane)
        self.env=g.create_env(self.sim,gymapi.Vec3(-1,-1,0),gymapi.Vec3(1,1,2),1)
        origin=g.get_env_origin(self.env);self.origin=np.array([origin.x,origin.y,origin.z])
        self.state_offset=single_env_state_offset(self.origin)
        print('[COORDINATES] one environment; origin='+str(self.origin.tolist())+'; state offset=[0,0,0]',flush=True)
        if not self.object_only:
            options=gymapi.AssetOptions();options.fix_base_link=True;options.disable_gravity=True
            options.collapse_fixed_joints=False;options.use_physx_armature=not a.no_armature
            options.default_dof_drive_mode=int(gymapi.DOF_MODE_EFFORT)
            print('[PHASE] loading '+('fixed' if fixed else 'six-DOF wrist')+' hand asset',flush=True)
            asset=g.load_asset(self.sim,str(hand_path.parent),hand_path.name,options)
            if asset is None:raise RuntimeError('hand asset load failed')
            self.hand=g.create_actor(self.env,asset,sdk_pose(root7),'hand',0,0)
            # Apply source named-pair masks to every convex shape of each body.
            config=json.loads((a.repo/'grasp_seed_banks/two_stage/collision_excludes.json').read_text())
            body_names=list(g.get_actor_rigid_body_names(self.env,self.hand))
            masks=collision_masks(body_names,config['pairs'])
            ranges=g.get_actor_rigid_body_shape_indices(self.env,self.hand)
            shapes=g.get_actor_rigid_shape_properties(self.env,self.hand)
            if len(ranges)!=len(body_names):raise RuntimeError('Rigid-body/shape-range mismatch')
            expected=[None]*len(shapes)
            for name,span in zip(body_names,ranges):
                for i in range(span.start,span.start+span.count):
                    if i>=len(shapes) or expected[i] is not None:raise RuntimeError('Invalid shape range')
                    expected[i]=masks[name];shapes[i].filter=masks[name]
            if any(v is None for v in expected):raise RuntimeError('Unmapped hand collision shape')
            g.set_actor_rigid_shape_properties(self.env,self.hand,shapes)
            check=g.get_actor_rigid_shape_properties(self.env,self.hand)
            if [int(v.filter) for v in check]!=expected:raise RuntimeError('Collision-filter readback mismatch')
            self.collision_config=dict(source=config['source'],commit=config['commit'],pairs=config['pairs'],
                                       bodyMasks=masks,shapeCount=len(shapes),readbackPassed=True)
            print('[COLLISION FILTERS] %d named pairs, %d shapes, readback passed'%(len(config['pairs']),len(shapes)),flush=True)
            self.names=list(g.get_actor_dof_names(self.env,self.hand));expected=set(ACTUAL if fixed else ACTUAL+WRIST)
            if set(self.names)!=expected:raise RuntimeError('Unexpected Isaac Gym DOFs: '+str(self.names))
            self.fi=np.array([self.names.index(n) for n in ACTUAL]);self.wi=np.array([self.names.index(n) for n in WRIST]) if not fixed else np.array([],dtype=int)
            props=g.get_actor_dof_properties(self.env,self.hand);props['driveMode'][:]=int(gymapi.DOF_MODE_EFFORT)
            props['stiffness'][:]=0;props['damping'][:]=0;props['effort'][self.fi]=a.torque_limit
            props['armature'][self.fi]=.001 if not a.no_armature else 0.
            if not fixed:
                props['driveMode'][self.wi]=int(gymapi.DOF_MODE_POS)
                props['stiffness'][self.wi]=[8000,8000,8000,300,300,300]
                props['damping'][self.wi]=[200,200,200,20,20,20];props['effort'][self.wi]=500.
            g.set_actor_dof_properties(self.env,self.hand,props);self.lower=props['lower'][self.fi].copy();self.upper=props['upper'][self.fi].copy()
            self.target=np.zeros(len(self.names),np.float32)
            self.fabase=g.find_actor_rigid_body_handle(self.env,self.hand,'fabase')
            if self.fabase<0:raise RuntimeError('Cannot resolve actual wrist body fabase')
        else:
            self.collision_config=None;self.hand=None;self.names=[];self.fi=np.array([],dtype=int);self.wi=np.array([],dtype=int)
            self.target=np.zeros(0,np.float32);self.lower=np.zeros(0);self.upper=np.zeros(0)
        options=gymapi.AssetOptions();options.fix_base_link=False;options.disable_gravity=False
        self.object_asset_options={'replaceCylinderWithCapsule':bool(getattr(options,'replace_cylinder_with_capsule',False))}
        print('[OBJECT ASSET] '+json.dumps(self.object_asset_options),flush=True)
        obj_asset=g.load_asset(self.sim,str(a.repo/'assets'),'cylinder.urdf',options)
        if obj_asset is None:raise RuntimeError('cylinder asset load failed')
        self.obj=g.create_actor(self.env,obj_asset,sdk_pose(object7),'object',0,0)
        if not g.set_actor_scale(self.env,self.obj,a.scale):raise RuntimeError('set_actor_scale failed')
        mass=g.get_actor_rigid_body_properties(self.env,self.obj);mass[0].mass=a.mass;g.set_actor_rigid_body_properties(self.env,self.obj,mass,True)
        for actor in ([self.hand] if not self.object_only else [])+[self.obj]:
            shapes=g.get_actor_rigid_shape_properties(self.env,actor)
            for shape in shapes:shape.friction=a.friction;shape.restitution=0.
            g.set_actor_rigid_shape_properties(self.env,actor,shapes)
        obj_names=g.get_actor_rigid_body_names(self.env,self.obj)
        if len(obj_names)!=1:raise RuntimeError('Cylinder must have one rigid body')
        self.obody=g.find_actor_rigid_body_handle(self.env,self.obj,obj_names[0])
        self.body_names={-1:'ground'}
        if not self.object_only:
            for name in g.get_actor_rigid_body_names(self.env,self.hand):self.body_names[g.find_actor_rigid_body_handle(self.env,self.hand,name)]=name
        self.body_names[self.obody]='object'
        self.viewer=None
        if a.viewer:
            self.viewer=g.create_viewer(self.sim,gymapi.CameraProperties())
            if self.viewer is None:raise RuntimeError('viewer creation failed')
            g.viewer_camera_look_at(self.viewer,self.env,gymapi.Vec3(.65,-.65,.65),gymapi.Vec3(0,0,.22))
        if not g.prepare_sim(self.sim):raise RuntimeError('prepare_sim failed')
        self.time=0.;self.last_print=time.monotonic();self.command=None
        print('[CONFIG] actual DOFs='+str(self.names)+' gravity=-9.81 mass='+str(mass[0].mass)+' dt='+str(a.dt)+' armature='+str(not a.no_armature),flush=True)

    def initialize(self,q20,wrist7=None):
        if self.object_only:return
        states=self.gym.get_actor_dof_states(self.env,self.hand,gymapi.STATE_ALL).copy()
        states['pos'][:]=0;states['vel'][:]=0;states['pos'][self.fi]=q20
        if not self.fixed:
            coordinates,self.previous_wrist_rpy=wrist_coordinates(self.anchor,transform(wrist7[:3],quat=wrist7[3:]),self.previous_wrist_rpy)
            states['pos'][self.wi]=coordinates
        self.gym.set_actor_dof_states(self.env,self.hand,states,gymapi.STATE_ALL)
        self.target[:]=states['pos'];self.gym.set_actor_dof_position_targets(self.env,self.hand,self.target)

    def step(self,root_xyz,rpy,qtarget,stage):
        g=self.gym
        if not self.object_only:
            dof=g.get_actor_dof_states(self.env,self.hand,gymapi.STATE_ALL)
            tau=np.zeros(len(self.names),np.float32)
            tau[self.fi]=np.clip(self.a.kp*(qtarget-dof['pos'][self.fi])-self.a.kd*dof['vel'][self.fi],-self.a.torque_limit,self.a.torque_limit)
            if not self.fixed:
                coordinates,self.previous_wrist_rpy=wrist_coordinates(self.anchor,transform(root_xyz,rpy=rpy),self.previous_wrist_rpy)
                self.target[self.wi]=coordinates
                g.set_actor_dof_position_targets(self.env,self.hand,self.target)
            g.apply_actor_dof_efforts(self.env,self.hand,tau)
        self.last_target=np.array(qtarget,dtype=float).copy()
        self.last_effort=tau[self.fi].astype(float) if not self.object_only else np.zeros(0)
        self.command=transform(root_xyz,rpy=rpy)
        # Fetch after EVERY simulation step. No stale CPU velocity / queued six-step feedback.
        g.simulate(self.sim);g.fetch_results(self.sim,True);self.time+=self.a.dt
        if self.viewer:
            if g.query_viewer_has_closed(self.viewer):raise KeyboardInterrupt('viewer closed')
            g.step_graphics(self.sim);g.draw_viewer(self.viewer,self.sim,True);g.sync_frame_time(self.sim)
        if time.monotonic()-self.last_print>2:
            print('[SIM] '+stage+' simulated %.2fs'%self.time,flush=True);self.last_print=time.monotonic()

    def snapshot(self,stage):
        g=self.gym;states=g.get_actor_dof_states(self.env,self.hand,gymapi.STATE_ALL) if not self.object_only else np.zeros(0,dtype=[('pos','f4'),('vel','f4')])
        obj=g.get_actor_rigid_body_states(self.env,self.obj,gymapi.STATE_ALL)[0]
        op=np.array(xyz(obj['pose']['p']))
        object7=np.r_[op,quat(obj['pose']['r'])]
        # Single environment at origin: environment and global translations coincide.
        object7[:3]-=self.state_offset
        if not self.object_only:
            hand_states=g.get_actor_rigid_body_states(self.env,self.hand,gymapi.STATE_ALL)
            hand_names=g.get_actor_rigid_body_names(self.env,self.hand)
            wrist=hand_states[hand_names.index('fabase')]
            base=transform(np.array(xyz(wrist['pose']['p']))-self.state_offset,quat=quat(wrist['pose']['r']))
        else:base=self.anchor
        obj_t=transform(object7[:3],quat=object7[3:]);rel=np.linalg.inv(base)@obj_t
        ct=g.get_env_rigid_contacts(self.env)
        summary=contact_summary(ct,self.obody,self.body_names,.04*self.a.scale,.08*self.a.scale)
        hand_floor=False;self_contacts=[]
        for c in ct:
            a,b=int(c['body0']),int(c['body1'])
            if a in self.body_names and b in self.body_names and a not in (-1,self.obody) and b not in (-1,self.obody) and float(c['lambda'])>1e-8:
                self_contacts.append(sorted([self.body_names[a],self.body_names[b]]))
            if -1 in (a,b) and (b if a==-1 else a)!=self.obody and float(c['lambda'])>1e-8:hand_floor=True
        if hand_floor:summary['support'].append('hand_ground')
        axis=obj_t[:3,2];bottom=object7[2]-(.04*self.a.scale*abs(axis[2])+.04*self.a.scale*np.sqrt(max(0,1-axis[2]**2)))
        summary['handSelfContactPairs']=[list(p) for p in sorted(set(tuple(p) for p in self_contacts))]
        q=states['pos'][self.fi].astype(float)
        return dict(stage=stage,simTime=self.time,q20=q.tolist(),target20=self.last_target.tolist(),requestedEffort20=self.last_effort.tolist(),effortSampling='command before current physics step',velocity20=states['vel'][self.fi].astype(float).tolist(),object7=object7.tolist(),objectVelocity6=xyz(obj['vel']['linear'])+xyz(obj['vel']['angular']),hand7=pose7(base),relativeObject7=pose7(rel),bottomZ=float(bottom),contacts=summary,jointViolation=float(np.maximum(np.maximum(self.lower-q,q-self.upper),0).max()) if len(q) else 0.,wristErrorPosition=float(np.linalg.norm(base[:3,3]-self.command[:3,3])),wristErrorAngleDeg=float(np.rad2deg(Rotation.from_matrix(base[:3,:3].T@self.command[:3,:3]).magnitude())))

    def phase(self,name,duration,path,record_every_step=False,grip=None):
        print('[STAGE] '+name+' duration='+str(duration)+'s',flush=True);rows=[];steps=max(1,int(round(duration/self.a.dt)))
        for k in range(steps):
            p,r,q=path(smooth((k+1)/steps))
            if grip is not None:q=expand(grip.command)
            self.step(p,r,q,name)
            if grip is not None or record_every_step or k%max(1,int(round(.05/self.a.dt)))==0 or k==steps-1:
                row=self.snapshot(name);rows.append(row)
                if not np.isfinite(np.r_[row['q20'],row['object7'],row['objectVelocity6']]).all():raise RuntimeError('Nonfinite simulation state')
                if row['jointViolation']>.15:raise RuntimeError('Joint limit runaway')
                if 'hand_ground' in row['contacts']['support']:raise RuntimeError('Hand hit ground in '+name)
                if grip is not None and k<steps-1:grip.update(row,self.a.dt)
        return rows


    def contact_close(self, position, rotation, target16):
        print('[STAGE] contact_close duration=3.5s; flexion budget +.35rad; preload .025rad',flush=True)
        closer=ContactCloser(target16,self.lower,self.upper)
        rows=[];steps=max(1,int(round(3.5/self.a.dt)))
        for k in range(steps):
            self.step(position,rotation,expand(closer.command),'contact_close')
            row=self.snapshot('contact_close')
            if not np.isfinite(np.r_[row['q20'],row['object7'],row['objectVelocity6']]).all():raise RuntimeError('Nonfinite contact closure')
            if row['jointViolation']>.15:raise RuntimeError('Joint limit runaway during contact closure')
            if 'hand_ground' in row['contacts']['support']:raise RuntimeError('Hand hit ground during contact closure')
            rows.append(row)
            # Do not change command after final sample: saved target is exactly
            # the one used for the final physics step and later hold/replay.
            if k<steps-1:closer.update(row['q20'],row['contacts']['fingers'],self.a.dt)
        return rows,closer.report()

def pose_hold_faults(rows,minimum_height,a):
    faults=hold_check(rows,minimum_height)
    if not a.pose_validation:return faults
    sys.path.insert(0,str(a.repo/'hora/tasks'))
    from botyard_pose_contract import evaluate
    converted=[dict(object7=r['object7'],contacts=r['contacts'],sdkVelocity6=r['objectVelocity6'],jointViolation=r['jointViolation']) for r in rows]
    check=evaluate(converted,rows[0]['object7'],a.scale,a.dt,warmup_seconds=0.,max_tilt=90.)
    # Preserve all geometry/contact/relative-grip conditions. Replace ONLY the
    # SDK velocity decision by a per-physics-step pose-motion decision.
    faults=[f for f in faults if f not in ('not_settled_angular','not_settled_linear')]
    return sorted(set(faults+check['reasons']))

def recover_and_freeze(scene,position,rotation,grip,label,result):
    """Allow bounded search recovery, then require an unassisted full 3s hold.

    Transient finger loss is admitted ONLY during this search recovery.
    Support, large relative slip and drop remain disqualifying throughout.
    """
    reference=scene.snapshot(label+'_reference')
    rows=scene.phase(label+'_recovery',5.,lambda u:(position,rotation,expand(grip.command)),grip=grip)
    target=expand(grip.command)
    frozen=scene.phase(label+'_frozen',3.,lambda u:(position,rotation,target),record_every_step=True)
    faults=[]
    rot0=Rotation.from_quat(reference['relativeObject7'][3:])
    for row in rows+frozen:
        if row['contacts']['support'] or row['contacts']['capContacts']:faults.append('recovery_support_or_cap')
        if row['bottomZ']<.03 or reference['object7'][2]-row['object7'][2]>.005:faults.append('recovery_drop')
        if row['jointViolation']>.01:faults.append('recovery_joint_limit')
        if np.linalg.norm(np.array(row['relativeObject7'][:3])-reference['relativeObject7'][:3])>.012:faults.append('recovery_relative_slip')
        if (rot0.inv()*Rotation.from_quat(row['relativeObject7'][3:])).magnitude()>np.deg2rad(10):faults.append('recovery_rotation_slip')
    result.setdefault('holdRecovery',{})[label]=dict(controller=grip.report(),recoverySeconds=5.,frozenSeconds=3.,
        safetyReasons=sorted(set(faults)),frozenTargets=True,coldReplayUsesRecovery=False)
    print('[HOLD RECOVERY] '+label+' '+json.dumps(result['holdRecovery'][label]),flush=True)
    return rows,frozen,sorted(set(faults))

def cold_start(a, geo, source, seed, result):
    result['stage']='cold_start'
    final=result['final'];root7=list(seed['handPosition'])+list(final['hand7'][3:])
    # Preserve the measured aligned wrist orientation during fixed-hand restore.
    # Restoring the old seed quaternion would reintroduce the cylinder tilt.
    local=transform(final['relativeObject7'][:3],quat=final['relativeObject7'][3:]);world=transform(root7[:3],quat=root7[3:])@local
    sc=Scene(a,source,True,root7,pose7(world));rows=[]
    try:
        sc.initialize(final['q20'])
        rpy=Rotation.from_quat(root7[3:]).as_euler('xyz');q=expand(result['candidate']['close16'])
        if a.ff_preload>0:
            old=q.copy();target=forefinger_target(final['q20'],result['candidate']['close16'],sc.lower,sc.upper,a.ff_preload)
            wanted=expand(target)
            result['candidate']['close16']=target.tolist();q=wanted
            print('[FF PRELOAD] extra target gap rad='+str(a.ff_preload)+'; J4/thumb unchanged',flush=True)
            ramp=sc.phase('ff_preload_ramp',1.,lambda u:(root7[:3],rpy,old+u*(q-old)),record_every_step=True)
            result['ffPreloadRamp']=ramp
        result['coldHoldSeconds']=a.cold_hold_seconds
        warm=sc.phase('cold_start_settle',.4,lambda u:(root7[:3],rpy,q),record_every_step=a.cold_trace)
        rows=sc.phase('cold_start_hold',a.cold_hold_seconds,lambda u:(root7[:3],rpy,q),record_every_step=a.cold_trace)
        if a.cold_trace:
            from botyard_velocity_diagnostic import analyze_velocity
            result['velocityDiagnostic']=analyze_velocity(warm+rows,expected_dt=a.dt)
            print('[VELOCITY DIAGNOSTIC] '+json.dumps(result['velocityDiagnostic']['summary']),flush=True)
        faults=hold_check(rows,.03)
        axis_sign=result['axisAlignment']['axisSign']
        vertical=vertical_check(rows,axis_sign);result['coldStartVerticalCheck']=vertical
        faults+=vertical['reasons']
        # Compare against the actual restored snapshot, not only an already slipped warmup frame.
        restore_error=np.linalg.norm(np.array(rows[0]['relativeObject7'][:3])-np.array(final['relativeObject7'][:3]))
        if restore_error>.008:faults.append('cold_start_restore_slip')
        restored_angle=(Rotation.from_quat(final['relativeObject7'][3:]).inv()*Rotation.from_quat(rows[0]['relativeObject7'][3:])).magnitude()
        if restored_angle>np.deg2rad(5):faults.append('cold_start_restore_rotation')
        result['coldStart']=dict(passed=not faults,reasons=sorted(set(faults)),warmup=warm,frames=rows,handRoot7=root7)
        if a.pose_validation:
            sys.path.insert(0,str(a.repo/'hora/tasks'))
            from botyard_pose_contract import evaluate
            converted=[dict(object7=row['object7'],contacts=row['contacts'],sdkVelocity6=row['objectVelocity6'],jointViolation=row['jointViolation']) for row in warm+rows]
            check=evaluate(converted,pose7(world),a.scale,a.dt)
            result['legacyColdStart']=dict(result['coldStart'])
            result['poseColdStartCheck']=check
            faults=sorted(set(check['reasons']+[f for f in faults if f.startswith('cold_start_restore_')]))
            result['coldStart']['passed']=not faults;result['coldStart']['reasons']=faults
        result['accepted']=not faults
        if not faults:
            # 27 values: ACTUAL20 then object xyz + xyzw in ENVIRONMENT coordinates (not wrist local).
            # Must not feed to HORA cache loader until metadata/controller/asset contract is integrated.
            last=rows[-1]
            result['canonical']=dict(q20=last['q20'],holdingTarget16=result['candidate']['close16'],holdingTarget20=expand(result['candidate']['close16']).tolist(),preloadError20=(expand(result['candidate']['close16'])-np.array(last['q20'])).tolist(),velocity20=last['velocity20'],objectVelocity6=last['objectVelocity6'],collisionConfiguration=sc.collision_config,controllerVersion='contact_closure_v3_vertical_named28',verticalAxisSign=axis_sign,verticalToleranceDeg=2.,verticalErrorDeg=vertical['finalErrorDeg'],distalCoupling='equal motor targets; not a passive mechanical tendon',object7Environment=last['object7'],object7HandRelative=last['relativeObject7'],handRoot7=root7,state27=last['q20']+last['object7'],jointNames=ACTUAL,policyJointNames=POLICY,scale=a.scale,mass=a.mass,friction=a.friction,kp=a.kp,kd=a.kd,torqueLimit=a.torque_limit,usePhysxArmature=not a.no_armature,dt=a.dt,assetSha256=hashlib.sha256(source.read_bytes()).hexdigest(),trainingCache=False)
        atomic_json(a.report,result)
    finally:pass  # Dedicated worker exits after saving, bypassing known native cleanup hangs.


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--candidate',type=int,default=0);ap.add_argument('--scale',type=float,default=1.);ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--object-only',action='store_true');ap.add_argument('--viewer',action='store_true');ap.add_argument('--no-armature',action='store_true')
    ap.add_argument('--dt',type=float,default=1/240);ap.add_argument('--kp',type=float,default=3.);ap.add_argument('--kd',type=float,default=.1)
    ap.add_argument('--torque-limit',type=float,default=.5);ap.add_argument('--mass',type=float,default=.05);ap.add_argument('--friction',type=float,default=.5)
    ap.add_argument('--replay',type=Path);ap.add_argument('--cold-trace',action='store_true');ap.add_argument('--pose-validation',action='store_true')
    ap.add_argument('--ff-preload',type=float,default=0.);ap.add_argument('--cold-hold-seconds',type=float,default=3.)
    ap.add_argument('--adaptive-grip',action='store_true',help='Bounded recovery in lift/flip SEARCH only; freeze before hold and cold replay')
    a=ap.parse_args();a.repo=a.repo.resolve()
    if not 0<=a.ff_preload<=.06 or not 3<=a.cold_hold_seconds<=30:ap.error('Invalid preload or hold duration')
    if a.ff_preload and not a.replay:ap.error('--ff-preload requires fixed-hand replay')
    if a.cold_trace and not a.replay:ap.error('--cold-trace requires --replay')
    if a.pose_validation:a.cold_trace=True
    result=dict(completed=False,accepted=False,stage='load',candidateIndex=a.candidate,scale=a.scale)
    source=a.repo/'assets/botyard_dexhand_gym/urdf/right_dexhand.urdf';scene=None
    object_source=a.repo/'assets/cylinder.urdf'
    if object_source.exists():result['objectAssetUrdf']=object_source.read_text();result['objectAssetSha256']=hashlib.sha256(object_source.read_bytes()).hexdigest()
    try:
        geo=Geometry(source);tag={1.:'s10',.9:'s09',.8:'s08'}[a.scale]
        seed=json.loads((a.repo/'grasp_seed_banks/two_stage'/ (tag+'.json')).read_text())['candidates'][0]
        if a.object_only:
            result['stage']='object_only_settle'
            object7=[0,0,.04*a.scale+.001,0,0,0,1]
            scene=Scene(a,source,True,[0,0,0,0,0,0,1],object7)
            scene.initialize([])
            rows=scene.phase('object_only_settle',1.,lambda u:([0,0,0],[0,0,0],np.zeros(0)))
            faults=upright_start_check(rows,object7[:3],.04*a.scale,.08*a.scale)
            result.update(completed=True,objectOnlyPassed=not faults,reasons=faults,frames=rows,objectAssetOptions=scene.object_asset_options)
            atomic_json(a.report,result);print('[OBJECT ONLY] passed='+str(not faults)+' reasons='+str(faults),flush=True)
            return 0
        if a.replay:
            old=json.loads(a.replay.read_text())
            if not old.get('canonical') and not old.get('stage2',{}).get('passed'):raise ValueError('Replay requires a passed two-stage report or canonical data')
            result=dict(old);result['accepted']=False;result['completed']=False
            # Replay precisely the fixed-hand canonical snapshot, not an old transient state.
            if old.get('canonical'):
                can=old['canonical'];result.pop('canonical',None)
                if can.get('controllerVersion')!='contact_closure_v3_vertical_named28':raise ValueError('Replay requires new controller/collision contract; old caches are not compatible')
                if can['collisionConfiguration']['pairs']!=json.loads((a.repo/'grasp_seed_banks/two_stage/collision_excludes.json').read_text())['pairs']:raise ValueError('Replay collision pairs differ')
                if can['scale']!=a.scale or can['assetSha256']!=hashlib.sha256(source.read_bytes()).hexdigest():raise ValueError('Replay asset/scale mismatch')
                for key,value in [('kp',a.kp),('kd',a.kd),('mass',a.mass),('friction',a.friction),('torqueLimit',a.torque_limit),('dt',a.dt),('usePhysxArmature',not a.no_armature)]:
                    if can[key]!=value:raise ValueError('Replay configuration differs: '+key)
                result['final']={'q20':can['q20'],'relativeObject7':can['object7HandRelative'],'hand7':can['handRoot7']}
                result['axisAlignment']={'axisSign':can['verticalAxisSign']}
                result['candidate']['close16']=can['holdingTarget16']
                seed['handPosition']=can['handRoot7'][:3];seed['handRotationRPY']=Rotation.from_quat(can['handRoot7'][3:]).as_euler('xyz').tolist()
            cold_start(a,geo,source,seed,result)
        else:
            rng=np.random.default_rng(a.seed)
            for i in range(a.candidate+1):candidate=sample_candidate(seed,i,rng,geo,a.scale)
            result['candidate']=candidate
            result['controllerVersion']='contact_closure_v3_vertical_named28'
            if candidate['meshFloorClearance']<.003 or candidate['palmGap']<.008 or candidate['pregraspVertexClearance']<.005:
                result['stage']='preflight';result['reasons']=['floor_or_palm_clearance'];result['completed']=True
                atomic_json(a.report,result);print('[REJECT] geometric clearance',flush=True);return 0
            mobile=make_wrist_urdf(source,a.report.parent/('mobile_wrist_'+str(a.candidate)+'.urdf'))
            p0=np.array(candidate['rootStart7'][:3]);r0=np.array(candidate['goalRPY']);r0[0]+=np.pi
            qopen=expand(candidate['open16']);qclose=expand(candidate['close16']);above=p0+np.array([0,0,.10])
            pregrasp7=pose7(transform(above,rpy=r0))
            scene=Scene(a,mobile,False,pregrasp7,candidate['objectStart7']);scene.initialize(qopen,pregrasp7)
            traces=[];result['frames']=traces;result['stage']='pregrasp';traces+=scene.phase('pregrasp',.4,lambda u:(above,r0,qopen))
            result['collisionConfiguration']=scene.collision_config
            result['trackingCheck']=tracking_check(traces)
            if not result['trackingCheck']['passed']:
                result.update(completed=True,stage='tracking_rejected',reasons=result['trackingCheck']['reasons'])
                atomic_json(a.report,result);print('[HALT TRACKING] '+json.dumps(result['trackingCheck']),flush=True);return 0
            startup_faults=upright_start_check(traces,candidate['objectStart7'][:3],.04*a.scale,.08*a.scale)
            result['pregraspCheck']=dict(passed=not startup_faults,reasons=startup_faults)
            if startup_faults:
                result.update(completed=True,stage='pregrasp_rejected',reasons=startup_faults)
                atomic_json(a.report,result);print('[REJECT BEFORE APPROACH] '+str(startup_faults),flush=True);return 0
            result['stage']='approach';traces+=scene.phase('approach',1.,lambda u:(above+u*(p0-above),r0,qopen))
            result['stage']='close';closed=scene.phase('close',1.5,lambda u:(p0,r0,qopen+u*(qclose-qopen)));traces+=closed
            result['stage']='contact_close';atomic_json(a.report,result)
            closure,closure_report=scene.contact_close(p0,r0,candidate['close16']);traces+=closure
            result['closureController']=closure_report
            result['closureStrictCheck']=closed_grip_check(closure,a.scale)
            candidate['nominalClose16']=candidate['close16'].copy()
            candidate['close16']=closure_report['holdingTarget16'];qclose=expand(candidate['close16'])
            contact_faults=[f for f in result['closureStrictCheck']['reasons'] if f!='object_not_settled_after_close']
            if contact_faults:
                result.update(completed=True,stage='closure_rejected',reasons=contact_faults,closedGripCheck=result['closureStrictCheck'])
                atomic_json(a.report,result);print('[REJECT BEFORE SETTLE] '+str(contact_faults),flush=True);return 0
            result['stage']='grip_settle';atomic_json(a.report,result)
            settled=scene.phase('grip_settle',1.,lambda u:(p0,r0,qclose),record_every_step=True);traces+=settled
            result['settleStrictCheck']=closed_grip_check(settled,a.scale)
            result['closedGripCheck']=closed_grip_check(settled,a.scale,settled=True)
            if a.pose_validation:
                result['legacyClosedGripCheck']=dict(result['closedGripCheck'])
                tail=settled[-max(2,int(round(.3/a.dt))):]
                poses=np.array([r['object7'] for r in tail]);rotations=Rotation.from_quat(poses[:,3:])
                linear=np.linalg.norm(np.diff(poses[:,:3],axis=0)/a.dt,axis=1)
                angular=(rotations[1:]*rotations[:-1].inv()).magnitude()/a.dt
                drift=np.linalg.norm(poses[:,:3]-poses[0,:3],axis=1).max()
                angle=(rotations*rotations[0].inv()).magnitude().max()
                reasons=[x for x in result['closedGripCheck']['reasons'] if x!='object_not_settled_after_close']
                if linear.max()>.03 or angular.max()>.3 or drift>.001 or angle>np.deg2rad(1):reasons.append('pose_motion_not_settled_after_close')
                result['closedGripCheck'].update(passed=not reasons,reasons=reasons,validationVersion='per_physics_step_pose_motion',
                    poseMotion=dict(maxLinearSpeed=float(linear.max()),maxAngularSpeed=float(angular.max()),driftM=float(drift),driftDeg=float(np.degrees(angle))))
            result['initialContactBandMeanZ']=result['closedGripCheck']['meanContactBandZ']
            print('[GRIP SETTLE] '+json.dumps(result['closedGripCheck']),flush=True)
            if not result['closedGripCheck']['passed']:
                result.update(completed=True,stage='closure_rejected',reasons=result['closedGripCheck']['reasons'])
                atomic_json(a.report,result);print('[REJECT BEFORE LIFT] '+json.dumps(result['closedGripCheck']),flush=True);return 0
            grip=HoldRecovery(candidate['close16'],scene.lower,scene.upper) if a.adaptive_grip else None
            lifted=p0+np.array([0,0,candidate['liftDistance']]);result['stage']='lift';atomic_json(a.report,result);traces+=scene.phase('lift',2.,lambda u:(p0+u*(lifted-p0),r0,qclose),grip=grip)
            recovery_faults=[]
            if grip is not None:
                recovery,hold1,recovery_faults=recover_and_freeze(scene,lifted,r0,grip,'lift_hold',result);traces+=recovery
                candidate['close16']=grip.command.tolist();qclose=expand(grip.command)
            else:hold1=scene.phase('lift_hold',1.5,lambda u:(lifted,r0,qclose),record_every_step=a.pose_validation)
            traces+=hold1
            faults=pose_hold_faults(hold1,.08,a)+recovery_faults
            band=result['initialContactBandMeanZ']
            if band is None or band<0 or band>.02*a.scale:faults.append('initial_contact_band_not_slightly_above_center')
            result.update(stage='lift',stage1=dict(passed=not faults,reasons=faults),frames=traces)
            atomic_json(a.report,result)
            if faults:print('[REJECT] lift '+str(faults),flush=True)
            else:
                actual=hold1[-1];base=transform(actual['hand7'][:3],quat=actual['hand7'][3:]);relative=np.array(actual['relativeObject7'][:3]);pivot=np.array(actual['object7'][:3])
                # Planned pivot is fixed once from measured state. Never update object pose while moving.
                def flip_path(u):
                    r=r0.copy();r[0]-=np.pi*u;rot=Rotation.from_euler('xyz',r).as_matrix()
                    return pivot-rot@relative,r,qclose
                result['stage']='flip';flip=scene.phase('flip',3.,flip_path,grip=grip);traces+=flip
                pend,rend,_=flip_path(1.)
                recovery_faults=[]
                if grip is not None:
                    recovery,hold2,recovery_faults=recover_and_freeze(scene,pend,rend,grip,'final_hold',result);traces+=recovery
                    candidate['close16']=grip.command.tolist();qclose=expand(grip.command)
                else:hold2=scene.phase('final_hold',3.,lambda u:(pend,rend,qclose),record_every_step=a.pose_validation)
                traces+=hold2
                faults=pose_hold_faults(hold2,.05,a)+recovery_faults
                # Flip path must preserve relative grip and never regain table support.
                reference=np.array(actual['relativeObject7'][:3]);reference_rot=Rotation.from_quat(actual['relativeObject7'][3:])
                for row in flip:
                    if np.linalg.norm(np.array(row['relativeObject7'][:3])-reference)>.012:faults.append('flip_slip')
                    if (reference_rot.inv()*Rotation.from_quat(row['relativeObject7'][3:])).magnitude()>np.deg2rad(10):faults.append('flip_rotation_slip')
                    if row['bottomZ']<.03 or row['contacts']['support']:faults.append('flip_support_or_drop')
                    if 'th' not in row['contacts']['fingers']:faults.append('flip_thumb_lost')
                # Correct the MEASURED cylinder axis rather than assuming a
                # commanded 180-degree wrist flip made the object vertical.
                z=Rotation.from_quat(hold2[-1]['object7'][3:]).as_matrix()[2,2]
                sign=1 if z>=0 else -1
                result['axisAlignment']=dict(axisSign=sign,initialErrorDeg=axis_error(hold2[-1]['object7'],sign),
                    toleranceDeg=2.,cycles=[],maximumTotalCorrectionDeg=20.,objectPoseWritten=False)
                if not faults:
                    measured=hold2[-1];total=0.;pend=np.array(measured['hand7'][:3]);rend=Rotation.from_quat(measured['hand7'][3:]).as_euler('xyz')
                    for cycle in range(3):
                        error=axis_error(measured['object7'],sign)
                        if error<=1.:break
                        if error>20. or total>=20.:
                            faults.append('axis_alignment_out_of_range');break
                        plan=alignment_plan(measured,sign,maximum_step_deg=min(10.,20.-total))
                        result['stage']='axis_align';atomic_json(a.report,result)
                        motion=scene.phase('axis_align_%d'%cycle,1.2,lambda u:(*alignment_pose(plan,u),qclose));traces+=motion
                        pend,rend=alignment_pose(plan,1.)
                        settle=scene.phase('axis_align_settle_%d'%cycle,.8,lambda u:(pend,rend,qclose));traces+=settle
                        reference=np.array(measured['relativeObject7'][:3]);reference_rot=Rotation.from_quat(measured['relativeObject7'][3:])
                        for row in motion+settle:
                            if np.linalg.norm(np.array(row['relativeObject7'][:3])-reference)>.012:faults.append('alignment_slip')
                            if (reference_rot.inv()*Rotation.from_quat(row['relativeObject7'][3:])).magnitude()>np.deg2rad(10):faults.append('alignment_rotation_slip')
                            if row['bottomZ']<.03 or row['contacts']['support']:faults.append('alignment_support_or_drop')
                            if len(row['contacts']['fingers'])!=5:faults.append('alignment_finger_contact_lost')
                        total+=plan['requestedAngleDeg'];measured=settle[-1]
                        result['axisAlignment']['cycles'].append(dict(cycle=cycle,requestedAngleDeg=plan['requestedAngleDeg'],measuredErrorDeg=axis_error(measured['object7'],sign)))
                        print('[AXIS ALIGN] error %.3fdeg, total wrist correction %.3fdeg'%(axis_error(measured['object7'],sign),total),flush=True)
                        if faults:break
                    if not faults:
                        hold2=scene.phase('aligned_final_hold',3.,lambda u:(pend,rend,qclose),record_every_step=a.pose_validation);traces+=hold2
                        faults+=pose_hold_faults(hold2,.05,a)
                vertical=vertical_check(hold2,sign);result['axisAlignment']['finalCheck']=vertical
                faults+=vertical['reasons']
                faults=sorted(set(faults));result.update(stage='flip',stage2=dict(passed=not faults,reasons=faults),frames=traces,final=hold2[-1])
                atomic_json(a.report,result)
                if faults:print('[REJECT] flip '+str(faults),flush=True)
                else:print('[PASS] lift and flip; supervisor will verify FIXED wrist cold start in a separate process',flush=True)
            # Cold start runs in a NEW process with the ORIGINAL fixed-wrist URDF.
            # No native destroy_sim call between trials: the supervisor owns termination.
            if result.get('stage2',{}).get('passed'):result['stage']='awaiting_fixed_wrist_cold_start'
        result['completed']=True;atomic_json(a.report,result)
        print('[RESULT] accepted='+str(result['accepted'])+' report='+str(a.report),flush=True);return 0
    except BaseException as ex:
        result.update(completed=False,accepted=False,error=type(ex).__name__+': '+str(ex));atomic_json(a.report,result);traceback.print_exc();return 130 if isinstance(ex,KeyboardInterrupt) else 1
    # Do not destroy a half-created native scene during error handling; supervisor owns this process.

if __name__=='__main__':
    code=main();sys.stdout.flush();sys.stderr.flush();os._exit(code)
