"""Compact independent per-environment traces; same scalar acceptance contract.

No torch or Gym dependency: fake states can exercise indexing and reset math.
"""
import numpy as np
import math
from pathlib import Path
from .botyard_pose_contract import FINGERS, evaluate, side_contacts, expand_delta


def restore_partial(path,expected):
    from .botyard_grasp_cache import read_cache
    states,targets,meta=read_cache(path,expected['jointNames'],expected['objectScale'],expected['handRootPoseLocal'],
        expected['objectAsset'],expected['handAsset'])
    if meta.get('complete') is not False:raise ValueError('Resume source is not a partial cache')
    exact=['validationVersion','torqueControl','controlFrequencyInv','collisionConfigSha256',
           'solverPositionIterations','solverVelocityIterations']
    numeric=['objectMassKg','friction','pgain','dgain','dt','warmupSeconds','maxTiltDeg','armature','torqueLimit']
    for key in exact:
        if meta.get(key)!=expected[key]:raise ValueError('Partial cache contract changed: '+key)
    for key in numeric:
        if not math.isclose(float(meta.get(key,-1)),float(expected[key]),rel_tol=1e-6,abs_tol=1e-9):
            raise ValueError('Partial cache physics changed: '+key)
    # Older v3 files did not record these two thresholds. They used the fixed
    # v3 values; accept that legacy case only when the current values match.
    for key,legacy in [('maxDriftM',.008),('maxDropM',.005)]:
        if not math.isclose(float(meta.get(key,legacy)),float(expected[key]),rel_tol=1e-6,abs_tol=1e-9):
            raise ValueError('Partial acceptance threshold changed: '+key)
    return states,targets,meta


class LocalSampler:
    """Reduce proposal width on low yield; never change acceptance thresholds."""
    def __init__(self,enabled=False,factor=1.):
        if factor not in (1.,.5,.25):raise ValueError('Invalid proposal factor')
        self.enabled=enabled;self.factor=factor;self.recent=[];self.history=[]
    def update(self,passed,total,accepted):
        self.recent.append(float(passed)/total);self.recent=self.recent[-3:]
        if self.enabled and len(self.recent)==3 and np.mean(self.recent)<.25 and self.factor>.25:
            old=self.factor;self.factor*=.5
            event=dict(acceptedRows=accepted,oldFactor=old,newFactor=self.factor,threeBatchPassRate=float(np.mean(self.recent)))
            self.history.append(event);self.recent=[];return event
        return None


class BatchTrace:
    def __init__(self,steps,envs):
        self.poses=np.empty((steps,envs,7),dtype=np.float64)
        self.velocity=np.empty((steps,envs,6),dtype=np.float32)
        self.contact=np.empty((steps,envs),dtype=np.uint8)
        self.support=np.empty((steps,envs),dtype=bool)
        self.violation=np.empty((steps,envs),dtype=np.float32)
        self.count=0
    def clear(self):self.count=0
    def append(self,poses,velocities,contacts,support,violation):
        if self.count>=len(self.poses):raise RuntimeError('Generation trace exceeded episode capacity')
        i=self.count
        self.poses[i]=poses;self.velocity[i]=velocities;self.contact[i]=contacts
        self.support[i]=support;self.violation[i]=violation;self.count+=1
    def rows(self,index):
        return [dict(object7=self.poses[i,index],sdkVelocity6=self.velocity[i,index],
            jointViolation=float(self.violation[i,index]),
            contacts=dict(fingers=[f for bit,f in enumerate(FINGERS) if int(self.contact[i,index])&(1<<bit)],
                          support=['support'] if self.support[i,index] else [],capContacts=0)) for i in range(self.count)]
    def check(self,index,start,scale,dt,warmup,tilt,drift,drop):
        return evaluate(self.rows(index),start,scale,dt,warmup,tilt,drift,drop)


def observe_batch(task,trace):
    # One refresh and one tensor conversion per batch, never re-read env zero
    # for every environment. Gym env contact handles share the actor layout.
    task._refresh_gym()
    poses=task.object_pose.cpu().numpy().astype(float).copy()
    poses[:,:3]-=task.state_offsets.cpu().numpy()
    velocity=np.concatenate([task.object_linvel.cpu().numpy(),task.object_angvel.cpu().numpy()],axis=1)
    q=task.allegro_hand_dof_pos.cpu().numpy()
    lower=task.allegro_hand_dof_lower_limits.cpu().numpy();upper=task.allegro_hand_dof_upper_limits.cpu().numpy()
    violation=np.maximum(np.maximum(lower-q,q-upper),0).max(axis=1)
    violation[~np.isfinite(q).all(axis=1)]=np.inf
    contacts=np.zeros(task.num_envs,dtype=np.uint8);support=np.zeros(task.num_envs,dtype=bool)
    for index,env in enumerate(task.envs):
        c=side_contacts(task.gym.get_env_rigid_contacts(env),task.object_body_handle,task.finger_body_groups,float(task.env_scales_np[index]))
        contacts[index]=sum(1<<FINGERS.index(f) for f in c['fingers'])
        support[index]=bool(c['support'] or c['capContacts'])
    trace.append(poses,velocity,contacts,support,violation)


def sample_batch(seeds,count,rng,names,lower,upper,policy_lower,policy_upper,noise,xy_noise,z_noise):
    chosen=rng.randint(len(seeds),size=count)
    delta=rng.uniform(-1,1,(count,16))*noise
    delta[:,[0,3,6,9,12,13]]*=.25
    q=np.array([seeds[i]['actualQ20'] for i in chosen])+np.stack([expand_delta(d,names) for d in delta])
    target=np.array([seeds[i]['holdingTarget16'] for i in chosen])+delta
    q=np.clip(q,lower,upper);target=np.clip(target,policy_lower,policy_upper)
    poses=np.array([seeds[i]['object7'] for i in chosen],dtype=float)
    poses[:,:3]+=rng.uniform(-1,1,(count,3))*[xy_noise,xy_noise,z_noise]
    return q,target,poses
