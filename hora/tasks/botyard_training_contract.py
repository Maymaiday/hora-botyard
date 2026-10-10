"""Pure NumPy counterparts for testing reward terms and coordinate selection."""
import numpy as np

def state_offsets(measured_roots, configured_roots, origins):
    measured=np.asarray(measured_roots);configured=np.asarray(configured_roots);origins=np.asarray(origins)
    local=np.max(np.abs(measured-configured));world=np.max(np.abs(measured-configured-origins))
    if world<1e-4:return origins.copy(),'world'
    if local<1e-4:return np.zeros_like(origins),'environment-local'
    raise RuntimeError('Post-physics fixed-root coordinate check failed: local=%g world=%g'%(local,world))

def constrained_targets(previous, action, reference, offset_limits, lower, upper, step_rad):
    lo=np.maximum(lower,np.asarray(reference)-offset_limits);hi=np.minimum(upper,np.asarray(reference)+offset_limits)
    return np.clip(np.asarray(previous)+np.clip(action,-1,1)*step_rad,lo,hi)

def training_terms(q,q0,pose_scale,weights,p,p0,previous_p,quat,dt):
    q=np.asarray(q);weights=np.asarray(weights)
    pose=np.sum(weights*((q-q0)/pose_scale)**2,axis=-1)/weights.sum()
    delta=np.asarray(p)-p0
    drift=np.minimum(np.sum((delta/.01)**2,axis=-1),25.)
    velocity=np.minimum(np.sum(np.abs((np.asarray(p)-previous_p)/dt),axis=-1)/.05,5.)
    quat=np.asarray(quat);axis_z=1-2*(quat[...,0]**2+quat[...,1]**2)
    tilt=np.arccos(np.clip(np.abs(axis_z),0,1))
    tilt_term=np.minimum((tilt/np.deg2rad(5.))**2,25.)
    return dict(pose=pose,drift=drift,linearVelocity=velocity,tilt=tilt_term,tiltRad=tilt)
