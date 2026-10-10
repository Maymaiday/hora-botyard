"""Bounded forefinger target search with long fixed-hand holds and two independent restores."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
from run_botyard_full_pipeline import atomic_json,run_job,sha
from refine_botyard_cold_grasp import proposal,eligible

def replay_source(data):
    out={k:copy.deepcopy(v) for k,v in data.items() if k not in ('frames','coldStart','legacyColdStart','canonical','ffPreloadRamp','velocityDiagnostic')}
    if data.get('canonical'):
        can=data['canonical'];out['final']=dict(q20=can['q20'],relativeObject7=can['object7HandRelative'],hand7=can['handRoot7'])
        out['candidate']['close16']=can['holdingTarget16']
    out.update(accepted=False,completed=False,stage='ff_preload_proposal')
    return out

def ff_metrics(data):
    rows=data.get('coldStart',{}).get('frames',[])
    if not rows:return dict(passed=False,reasons=['no_hold_frames'])
    tail=[r for r in rows if r['simTime']>=rows[-1]['simTime']-1.]
    contacts=np.array(['ff' in r['contacts']['fingers'] for r in rows])
    # Positive flexion error is a preload proxy, not a contact force measurement.
    gap=np.array([np.array(r['target20'])[[1,2]]-np.array(r['q20'])[[1,2]] for r in tail])
    loaded=np.any(gap>=.005,axis=1)
    return dict(passed=bool(contacts.all() and loaded.mean()>=.95),sideContactFraction=float(contacts.mean()),
        tailLoadedFraction=float(loaded.mean()),medianFlexionGapRad=np.median(gap,axis=0).tolist(),
        criterion='all hold frames have ff side contact; >=95% tail has >=0.005rad positive error on FFJ3 or FFJ2',
        interpretation='controller preload proxy; not measured normal force')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--timeout',type=float,default=180.);p.add_argument('--viewer',action='store_true');a=p.parse_args()
    a.repo=a.repo.resolve();a.source=a.source.resolve();a.report=a.report.resolve()
    import os
    os.chdir(str(a.repo));original=json.loads(a.source.read_text());scale=float(original['scale'])
    if not original.get('stage2',{}).get('passed') or not original.get('final'):raise ValueError('Passed two-stage source required')
    if original.get('objectAssetSha256')!=sha(a.repo/'assets/cylinder.urdf'):raise ValueError('Source object asset changed')
    if original.get('controllerVersion')!='contact_closure_v3_vertical_named28':raise ValueError('Controller mismatch')
    out=a.report.parent/(a.report.stem+'_trials');out.mkdir(parents=True,exist_ok=False);history=[]
    def trial(name,source,preload=0.):
        path=out/(name+'_source.json');report=out/(name+'.json');atomic_json(path,source)
        cmd=[sys.executable,'-u','tools/botyard_two_stage_worker.py','--repo',str(a.repo),'--scale',str(scale),
             '--replay',str(path),'--report',str(report),'--pose-validation','--ff-preload',str(preload),'--cold-hold-seconds','8']
        if a.viewer:cmd+=['--viewer']
        if run_job(cmd,out/(name+'.log'),a.timeout):raise RuntimeError('Physics worker failed: '+name)
        data=json.loads(report.read_text());metrics=ff_metrics(data)
        event=dict(trial=name,accepted=data.get('accepted'),check=data.get('poseColdStartCheck'),ffPreloadCheck=metrics)
        history.append(event);atomic_json(out/'search_summary.json',dict(complete=False,trials=history))
        print('[FF CHECK]',json.dumps(event),flush=True);return data,metrics
    for amount in [.01,.02,.04,.06]:
        name='ff_'+str(int(round(amount*1000)))+'mrad'
        data,metrics=trial(name,replay_source(original),amount)
        # Only after side contact + preload are established, permit two bounded
        # root corrections. All long-hold acceptance criteria remain unchanged.
        total=0.
        for cycle in range(2):
            if data.get('accepted') or not metrics['passed'] or not eligible(data):break
            source,angle=proposal(data);total+=angle
            if total>12.:break
            data,metrics=trial(name+'_align_'+str(cycle),source)
        if not data.get('accepted') or not metrics['passed'] or not data.get('canonical'):continue
        frozen=data;passed=True
        for repeat in range(2):
            # Both confirmations restore the SAME saved canonical, not a chain
            # of successively shifted snapshots.
            confirmed,m=trial(name+'_confirmation_'+str(repeat),frozen)
            if not confirmed.get('accepted') or not m['passed']:passed=False;break
        if passed:
            frozen['ffPreloadValidation']=dict(passed=True,holdSeconds=8,independentRestores=2,extraTargetGapRad=amount,trials=history)
            atomic_json(a.report,frozen);atomic_json(out/'search_summary.json',dict(complete=True,trials=history))
            print('[FF REPAIR PASS]',a.report,flush=True);return 0
    raise RuntimeError('No repeatedly stable loaded forefinger grasp within 10/20/40/60mrad search; inspect '+str(out))

if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc();sys.exit(1)
