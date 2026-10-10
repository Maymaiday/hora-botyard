#!/usr/bin/env python3
"""Finite tabletop lift -> flip -> independent fixed-wrist cold-start search."""
import argparse,json,os,signal,subprocess,sys,time,zipfile
from pathlib import Path

def stop(p):
    if p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:p.wait(timeout=.5);return
    except subprocess.TimeoutExpired:pass
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:return
    try:p.wait(timeout=1.)
    except subprocess.TimeoutExpired:print('[STOP] SIGKILL sent to worker PID '+str(p.pid),flush=True)

def run(args,log,report,timeout):
    start=time.monotonic();heartbeat=start;reason=None
    with log.open('w') as writer,log.open('r') as reader:
        p=subprocess.Popen(args,stdout=writer,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print('[PID] '+str(p.pid)+' log='+str(log),flush=True)
        try:
            while True:
                chunk=reader.read()
                if chunk:print(chunk,end='',flush=True)
                if p.poll() is not None:break
                now=time.monotonic()
                if now-start>timeout:reason='wall_timeout';stop(p);break
                if now-heartbeat>5:
                    print('[RUNNING] PID %d wall %.1fs (Ctrl+C stops this worker)'%(p.pid,now-start),flush=True);heartbeat=now
                time.sleep(.1)
            chunk=reader.read()
            if chunk:print(chunk,end='',flush=True)
        except BaseException:stop(p);raise
    data=json.loads(report.read_text()) if report.exists() else {}
    return dict(exitCode=p.poll(),stopReason=reason,wallSeconds=time.monotonic()-start,report=str(report),completed=bool(data.get('completed')),accepted=bool(data.get('accepted')),stage=data.get('stage'),error=data.get('error')),data

def unique_out(path):
    path=Path(path).resolve()
    if path.exists():path=path.with_name(path.name+'_'+time.strftime('%Y%m%d_%H%M%S'))
    base=path;i=1
    while path.exists():path=base.with_name(base.name+'_'+str(i));i+=1
    path.mkdir(parents=True);return path

def main():
    print('[START] finite two-stage Botyard search',flush=True)
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--scales','--scale',nargs='+',type=float,default=[1.])
    ap.add_argument('--trials',type=int,default=1);ap.add_argument('--target',type=int,default=1)
    ap.add_argument('--start-index',type=int,default=0);ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--timeout',type=float,default=120.,help='maximum wall seconds for EACH worker, including model loading')
    ap.add_argument('--viewer',action='store_true');ap.add_argument('--no-armature',action='store_true')
    ap.add_argument('--preflight',action='store_true',help='geometry checks without loading Isaac Gym')
    ap.add_argument('--plot',action='store_true',help='write offline geometry/diagnostic PNGs (requires matplotlib)')
    ap.add_argument('--out-dir',default='two_stage_results');ap.add_argument('--replay',type=Path)
    ap.add_argument('--dt',type=float,default=1/240);ap.add_argument('--kp',type=float,default=3.);ap.add_argument('--kd',type=float,default=.1)
    ap.add_argument('--torque-limit',type=float,default=.5);ap.add_argument('--mass',type=float,default=.05);ap.add_argument('--friction',type=float,default=.5)
    a=ap.parse_args()
    import math
    if not 1<=a.trials<=256 or not 1<=a.target<=a.trials or a.start_index<0:ap.error('trials 1..256, target 1..trials, start-index >= 0')
    if not all(s in (1.,.9,.8) for s in a.scales) or len(set(a.scales))!=len(a.scales):ap.error('scales must be unique values among 1.0 0.9 0.8')
    if not all(math.isfinite(x) for x in [a.timeout,a.dt,a.kp,a.kd,a.torque_limit,a.mass,a.friction]) or not (a.timeout>0 and 0<a.dt<=.01 and a.kp>0 and a.kd>=0 and 0<a.torque_limit<=.5 and a.mass>0 and a.friction>0):ap.error('invalid finite simulation parameters; torque limit <= .5 Nm')
    a.repo=a.repo.resolve()
    required=[a.repo/'assets/botyard_dexhand_gym/urdf/right_dexhand.urdf',a.repo/'assets/cylinder.urdf',a.repo/'grasp_seed_banks/two_stage/collision_excludes.json']+[a.repo/'grasp_seed_banks/two_stage'/n for n in ['s10.json','s09.json','s08.json']]
    missing=[str(p) for p in required if not p.exists()]
    if missing:ap.error('Copy package contents into hora root first. Missing: '+str(missing))
    out=unique_out(a.out_dir);summary=dict(protocol='tabletop_lift_flip_fixed_wrist',trainingCache=False,runs=[],acceptedFiles=[],settings=vars(a).copy())
    summary['settings']={k:str(v) if isinstance(v,Path) else v for k,v in summary['settings'].items()}
    print('[OUTPUT] '+str(out),flush=True)
    interrupted=False;errored=False
    try:
        if a.preflight:
            from botyard_two_stage_core import Geometry,sample_candidate,atomic_json
            import numpy as np
            geo=Geometry(required[0]);rows=[]
            for scale in a.scales:
                tag={1.:'s10',.9:'s09',.8:'s08'}[scale];seed=json.loads((a.repo/'grasp_seed_banks/two_stage'/(tag+'.json')).read_text())['candidates'][0];rng=np.random.default_rng(a.seed)
                for i in range(a.start_index+a.trials):
                    c=sample_candidate(seed,i,rng,geo,scale)
                    if i<a.start_index:continue
                    rows.append(c)
                    print('[GEOMETRY] %s #%d floor=%.1fmm palmGap=%.1fmm'%(tag,i,c['meshFloorClearance']*1000,c['palmGap']*1000),flush=True)
                    if a.plot and i==a.start_index:
                        from view_botyard_two_stage import plot_geometry
                        plot_geometry(geo,c,out/(tag+'_geometry.png'))
            atomic_json(out/'preflight.json',dict(simulationValidated=False,candidates=rows));summary['geometryOnly']=True
            return 0
        worker=Path(__file__).with_name('botyard_two_stage_worker.py')
        replay_data=json.loads(a.replay.read_text()) if a.replay else None
        if replay_data:
            can=replay_data.get('canonical')
            if not can:ap.error('--replay expects an accepted canonical report')
            a.scales=[can['scale']]
            for arg,key in [('kp','kp'),('kd','kd'),('mass','mass'),('friction','friction'),('torque_limit','torqueLimit'),('dt','dt')]:setattr(a,arg,can[key])
            a.no_armature=not can['usePhysxArmature']
        for scale in a.scales:
            count=0;tag={1.:'s10',.9:'s09',.8:'s08'}[scale]
            for index in range(a.start_index,a.start_index+(1 if a.replay else a.trials)):
                name='%s_%03d'%(tag,index);report=out/(name+'.json')
                args=[sys.executable,'-u',str(worker),'--repo',str(a.repo),'--report',str(report),'--scale',str(scale),'--candidate',str(index),'--seed',str(a.seed)]
                for key in ['dt','kp','kd','torque_limit','mass','friction']:args+=['--'+key.replace('_','-'),str(getattr(a,key))]
                if a.viewer:args+=['--viewer']
                if a.no_armature:args+=['--no-armature']
                if a.replay:args+=['--replay',str(a.replay.resolve())]
                if index==a.start_index and not a.replay:
                    baseline=out/(tag+'_object_only.json');baseline_args=args.copy()
                    baseline_args[baseline_args.index('--report')+1]=str(baseline);baseline_args+=['--object-only']
                    print('[STARTUP CONTROL] cylinder alone, no hand actor, 1 simulated second',flush=True)
                    baseline_status,baseline_data=run(baseline_args,out/(tag+'_object_only.log'),baseline,a.timeout)
                    summary['runs'].append(baseline_status)
                    if baseline_status['exitCode']!=0 or not baseline_status['completed']:errored=True;return 1
                    if not baseline_data.get('objectOnlyPassed'):
                        summary['startupFailure']='cylinder_alone_unstable'
                        print('[HALT BEFORE HAND] cylinder-alone control failed: '+str(baseline_data.get('reasons')),flush=True)
                        return 2
                print('\n[TRIAL] '+name+' accepted %d/%d'%(count,a.target),flush=True)
                status,data=run(args,out/(name+'.log'),report,a.timeout);summary['runs'].append(status)
                if status['exitCode']!=0 or not status['completed']:
                    errored=True;print('[HALT] worker failed/timed out; preserving logs and report, no further trials',flush=True);return 1
                if not a.replay and data.get('stage2',{}).get('passed'):
                    cold_report=out/(name+'_cold.json');cold_args=args.copy();cold_args[cold_args.index('--report')+1]=str(cold_report);cold_args+=['--replay',str(report)]
                    print('[COLD START] separate process, original fixed hand, identical finger controller',flush=True)
                    status,data=run(cold_args,out/(name+'_cold.log'),cold_report,a.timeout);summary['runs'].append(status)
                    if status['exitCode']!=0 or not status['completed']:errored=True;return 1
                if data.get('accepted'):
                    from botyard_two_stage_core import atomic_json
                    import numpy as np
                    can=data['canonical'];accepted=out/('canonical_'+name+'.json');atomic_json(accepted,data)
                    np.savez(out/('canonical_'+name+'.npz'),state27=np.array(can['state27']),holdingTarget16=np.array(can['holdingTarget16']),holdingTarget20=np.array(can['holdingTarget20']),preloadError20=np.array(can['preloadError20']),velocity20=np.array(can['velocity20']),objectVelocity6=np.array(can['objectVelocity6']),object7Environment=np.array(can['object7Environment']),metadataJson=np.array(json.dumps(can)),object7HandRelative=np.array(can['object7HandRelative']),handRoot7=np.array(can['handRoot7']),jointNames=np.array(can['jointNames']),trainingCache=np.array(False))
                    summary['acceptedFiles'].append(str(accepted));count+=1
                    print('[ACCEPT] '+str(accepted),flush=True)
                else:
                    if data.get('stage')=='tracking_rejected':
                        summary['trackingFailure']=data['trackingCheck']
                        print('[HALT] empty-hand tracking failed; no additional pose trials',flush=True)
                        return 2
                    reasons=data.get('coldStart',{}).get('reasons',data.get('stage2',data.get('stage1',{})).get('reasons',data.get('reasons',[])))
                    print('[REJECT] '+str(reasons),flush=True)
                (out/'summary.json').write_text(json.dumps(summary,indent=2))
                if count>=a.target:break
            if count==0:print('[NO SUCCESS] '+tag+'; this is a failed search, not a usable canonical pose',flush=True)
    except KeyboardInterrupt:
        interrupted=True;print('\n[INTERRUPTED] worker stopped; saving available diagnostics',flush=True)
    except Exception as ex:
        errored=True;summary['error']=type(ex).__name__+': '+str(ex);print('[ERROR] '+summary['error'],flush=True)
    finally:
        summary['interrupted']=interrupted;summary['workerError']=errored
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        if a.plot and not a.preflight:
            try:
                from view_botyard_two_stage import plot_report
                for report in out.glob('s*.json'):plot_report(report)
            except Exception as ex:print('[PLOT] '+str(ex),flush=True)
        archive=out.with_suffix('.zip')
        with zipfile.ZipFile(str(archive),'w',zipfile.ZIP_DEFLATED) as z:
            for p in out.iterdir():
                if p.suffix in ['.json','.log','.npz','.png']:z.write(str(p),out.name+'/'+p.name)
        print('[DONE] results '+str(archive),flush=True)
    return 130 if interrupted else 1 if errored else 0 if summary['acceptedFiles'] else 2
if __name__=='__main__':sys.exit(main())
