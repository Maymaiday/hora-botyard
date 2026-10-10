"""Search fixed-wrist roots from a real cold-hold snapshot; never rotate only the object."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from run_botyard_full_pipeline import run_job,atomic_json,sha

def correction(axis,max_degrees=8.):
    axis=np.asarray(axis,dtype=float);axis/=np.linalg.norm(axis)
    goal=np.array([0.,0.,1. if axis[2]>=0 else -1.])
    angle=float(np.arccos(np.clip(np.dot(axis,goal),-1.,1.)))
    cross=np.cross(axis,goal)
    if angle<1e-10:return Rotation.identity(),0.
    if np.linalg.norm(cross)<1e-10:raise ValueError('Ambiguous correction axis')
    applied=min(angle,np.deg2rad(max_degrees))
    return Rotation.from_rotvec(cross/np.linalg.norm(cross)*applied),float(np.rad2deg(applied))

def proposal(data):
    cold=data['coldStart'];row=cold['frames'][-1]
    root=cold['handRoot7'];rotate,angle=correction(Rotation.from_quat(row['object7'][3:]).apply([0,0,1]))
    # Rotate the hand and held object together; relative contact geometry and
    # actual finger angles are preserved. Each proposal gets a fresh physics run.
    new_root=list(root[:3])+(rotate*Rotation.from_quat(root[3:])).as_quat().tolist()
    rel=row['relativeObject7'];hand=Rotation.from_quat(new_root[3:])
    obj=(np.array(new_root[:3])+hand.apply(rel[:3])).tolist()+(hand*Rotation.from_quat(rel[3:])).as_quat().tolist()
    out={k:copy.deepcopy(v) for k,v in data.items() if k not in ('frames','coldStart','legacyColdStart','canonical','velocityDiagnostic')}
    out.update(accepted=False,completed=False,stage='fixed_wrist_root_proposal')
    out['final']=dict(q20=row['q20'],relativeObject7=rel,hand7=new_root,object7=obj)
    out['fixedWristRefinement']=dict(correctionDeg=angle,sourceTiltDeg=float(data['poseColdStartCheck']['metrics']['maxTiltDeg']),
        source='last actual cold-hold frame; unchanged holding target; new hand root orientation',proposalIsVerified=False)
    return out,angle

def eligible(data):
    reasons=set(data.get('coldStart',{}).get('reasons',[]))|set(data.get('poseColdStartCheck',{}).get('reasons',[]))
    if not (data.get('completed') and data.get('stage2',{}).get('passed') and data.get('coldStart',{}).get('frames')
        and reasons and reasons.issubset({'cylinder_tilt','cold_start_restore_rotation','tail_pose_drift'})):return False
    # Search admission is separate from cache acceptance. Small residual motion
    # may be used to propose another root; it never marks a candidate accepted.
    metrics=data.get('poseColdStartCheck',{}).get('metrics',{})
    limits=dict(maxTiltDeg=15.,maxDriftM=.008,maxPoseLinearSpeed=.03,maxPoseAngularSpeed=.3,
                tailPositionDriftM=.001,tailOrientationDriftDeg=2.)
    try:return all(np.isfinite(float(metrics[k])) and 0<=float(metrics[k])<=v for k,v in limits.items())
    except (KeyError,ValueError,TypeError):return False

def reject(report,data,reason,history):
    """A finished but physically rejected search is distinct from an SDK error."""
    result=copy.deepcopy(data);result.pop('canonical',None)
    result.update(accepted=False,completed=True,refinementOutcome='search_rejected',refinementRejectionReason=reason,
                  fixedWristRefinementHistory=history)
    atomic_json(report,result);print('[REFINE REJECTED]',reason,flush=True);return 0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--viewer',action='store_true')
    p.add_argument('--timeout',type=float,default=180.);p.add_argument('--settled-window',action='store_true');a=p.parse_args()
    a.repo=a.repo.resolve();a.source=a.source.resolve();a.report=a.report.resolve()
    import os
    os.chdir(str(a.repo));data=json.loads(a.source.read_text());scale=float(data['scale'])
    if data.get('objectAssetSha256')!=sha(a.repo/'assets/cylinder.urdf'):raise ValueError('Source object asset changed')
    if data.get('controllerVersion')!='contact_closure_v3_vertical_named28':raise ValueError('Source controller mismatch')
    out=a.report.parent/(a.report.stem+'_trials');out.mkdir(parents=True,exist_ok=False)
    total=float(data.get('fixedWristRefinement',{}).get('cumulativeCorrectionDeg',
                data.get('fixedWristRefinement',{}).get('correctionDeg',0.)));history=[]
    for cycle in range(4):
        if data.get('accepted') and data.get('canonical'):break
        if a.settled_window:
            from botyard_settled_window import settled_candidate
            try:data=settled_candidate(data)
            except ValueError as exc:return reject(a.report,data,str(exc),history)
            atomic_json(out/('settled_window_'+str(cycle)+'.json'),data)
            print('[SETTLED WINDOW]',json.dumps(data['settledSearchWindow']),flush=True)
        if not eligible(data) and not (a.settled_window and data['settledSearchWindow']['searchCheck']['passed']):return reject(a.report,data,'Candidate outside bounded refinement search',history)
        proposed,angle=proposal(data);total+=angle
        if total>20.:return reject(a.report,data,'Fixed-wrist correction exceeded 20 degrees',history)
        proposed['fixedWristRefinement']['cumulativeCorrectionDeg']=total
        source=out/('proposal_'+str(cycle)+'.json');report=out/('cold_'+str(cycle)+'.json')
        atomic_json(source,proposed)
        print('[FIXED WRIST REFINE] cycle',cycle,'scale',scale,'correctionDeg',round(angle,3),flush=True)
        cmd=[sys.executable,'-u','tools/botyard_two_stage_worker.py','--repo',str(a.repo),'--scale',str(scale),
             '--replay',str(source),'--report',str(report),'--pose-validation','--cold-hold-seconds','8']
        if a.viewer:cmd+=['--viewer']
        if run_job(cmd,out/('cold_'+str(cycle)+'.log'),a.timeout):raise RuntimeError('Refinement physics worker failed')
        data=json.loads(report.read_text());history.append(dict(cycle=cycle,accepted=data.get('accepted'),check=data.get('poseColdStartCheck')))
        print('[REFINE CHECK]',json.dumps(history[-1]),flush=True)
    if not data.get('accepted') or not data.get('canonical'):return reject(a.report,data,'Fixed-wrist refinement exhausted bounded trials',history)
    # Confirm the accepted canonical by another independent restore before use.
    accepted=out/'accepted_candidate.json';atomic_json(accepted,data);frozen=data
    for repeat in range(2):
        confirm=out/('confirmation_'+str(repeat)+'.json')
        cmd=[sys.executable,'-u','tools/botyard_two_stage_worker.py','--repo',str(a.repo),'--scale',str(scale),
            '--replay',str(accepted),'--report',str(confirm),'--pose-validation','--cold-hold-seconds','8']
        if a.viewer:cmd+=['--viewer']
        if run_job(cmd,out/('confirmation_'+str(repeat)+'.log'),a.timeout):raise RuntimeError('Confirmation worker failed')
        confirmed=json.loads(confirm.read_text())
        if not confirmed.get('accepted') or not confirmed.get('canonical'):return reject(a.report,confirmed,'Independent confirmation rejected',history)
    data=frozen
    data['independentColdRestores']=2;data['coldHoldSeconds']=8
    data['fixedWristRefinementHistory']=history;data['fixedWristSourceReport']=str(a.source)
    atomic_json(a.report,data);print('[REFINE PASS]',a.report,flush=True);return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc();sys.exit(1)
