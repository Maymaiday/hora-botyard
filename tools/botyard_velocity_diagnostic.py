"""Compare SDK world angular velocities with consecutive quaternion poses.

Diagnostic only. No changes to controller or acceptance thresholds; no claim
that sparse pose samples can rule out oscillation between samples.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def analyze_velocity(rows,expected_dt):
    if len(rows)<3:raise ValueError('Need at least three pose/velocity samples')
    times=np.array([r['simTime'] for r in rows]);dt=np.diff(times)
    if (dt<=0).any():raise ValueError('Non-monotonic sample times')
    rotations=Rotation.from_quat([r['object7'][3:] for r in rows])
    # Spatial/world angular velocity, matching SDK angular field convention.
    pose_omega=(rotations[1:]*rotations[:-1].inv()).as_rotvec()/dt[:,None]
    sdk=np.array([r['objectVelocity6'][3:] for r in rows])
    average_sdk=(sdk[1:]+sdk[:-1])/2
    residual=(Rotation.from_rotvec(average_sdk*dt[:,None])*rotations[:-1])*rotations[1:].inv()
    late=times[1:]>=times[-1]-1.
    fd=pose_omega[late];readout=sdk[1:][late]
    fd_speed=np.linalg.norm(fd,axis=1);sdk_speed=np.linalg.norm(readout,axis=1)
    fine=bool(np.allclose(dt,expected_dt,rtol=.01,atol=1e-8))
    signed_mean_speed=float(np.linalg.norm(fd.mean(axis=0)))
    fd_rms=float(np.sqrt(np.mean(fd_speed**2)))
    mismatch=float(np.sqrt(np.mean(np.sum((fd-average_sdk[late])**2,axis=1))))
    if not fine:classification='sparse_samples_cannot_identify_between_sample_motion'
    elif np.median(sdk_speed)>.3 and fd_speed.max()<.03:
        classification='reported_velocity_pose_mismatch_at_physics_step_resolution'
    elif fd_rms>.3 and signed_mean_speed<fd_rms*.2:
        classification='pose_changes_show_oscillatory_motion'
    elif fd_rms>.3:classification='pose_changes_show_motion_above_angular_limit'
    else:classification='pose_motion_below_limit_inspect_readout_residual'
    position=np.array([r['object7'][:3] for r in rows]);late_rows=times>=times[-1]-1.
    qr=rotations[late_rows];pr=position[late_rows]
    summary=dict(classification=classification,perPhysicsStepSamples=fine,expectedDt=expected_dt,
        medianSampleDt=float(np.median(dt)),sampleCount=len(rows),
        sdkAngularMedianRadS=float(np.median(sdk_speed)),sdkAngularMaxRadS=float(sdk_speed.max()),
        poseAngularRmsRadS=fd_rms,poseAngularMaxRadS=float(fd_speed.max()),
        poseSignedMeanAngularSpeedRadS=signed_mean_speed,
        velocityPoseResidualRmsRadS=mismatch,
        maxTrapezoidIntegrationResidualDeg=float(np.degrees(residual[late].magnitude().max())),
        tailPositionChangeMm=float(np.linalg.norm(pr-pr[0],axis=1).max()*1000),
        tailOrientationChangeDeg=float(np.degrees((qr[0].inv()*qr).magnitude().max())),
        interpretation='Empirical classification only; solver correction versus API readout cause requires additional evidence.',
        acceptanceThresholdsChanged=False)
    return dict(summary=summary,time=times[1:].tolist(),poseAngularWorld=pose_omega.tolist(),
                reportedAngularWorld=sdk[1:].tolist(),integrationResidualDeg=np.degrees(residual.magnitude()).tolist())


def plot_diagnostic(data,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=np.array(data['time']);fd=np.array(data['poseAngularWorld']);sdk=np.array(data['reportedAngularWorld'])
    fig,axes=plt.subplots(4,1,figsize=(10,9),sharex=True)
    for i,name in enumerate(['world X','world Y','world Z']):
        axes[i].plot(t,sdk[:,i],label='SDK readout');axes[i].plot(t,fd[:,i],label='quaternion difference',alpha=.8)
        axes[i].set_ylabel(name+' (rad/s)');axes[i].legend()
    axes[3].plot(t,np.linalg.norm(sdk,axis=1),label='SDK speed')
    axes[3].plot(t,np.linalg.norm(fd,axis=1),label='pose-derived speed');axes[3].axhline(.3,color='red',linestyle='--')
    axes[3].set(xlabel='Simulation time (s)',ylabel='Angular speed (rad/s)');axes[3].legend()
    fig.suptitle(data['summary']['classification']);fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)
