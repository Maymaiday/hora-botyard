"""Calibrate each size -> generate -> replay exact mixed layout -> PPO.

The active Python interpreter must be the user's hora conda interpreter.
Native jobs run in their own process groups with disk logs and finite limits.
"""
import argparse,hashlib,importlib.util,json,math,os,signal,subprocess,sys,time
from pathlib import Path

SCALE_TAG={1.:'s10',.9:'s09',.8:'s08'}
def count_tag(n):return str(n//1000)+'k' if n%1000==0 else str(n)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
def load_module(repo,name):
    spec=importlib.util.spec_from_file_location(name,repo/'hora/tasks'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def stop(process):
    if process.poll() is not None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()

def run_job(command,log,timeout):
    print('[COMMAND]',command,flush=True);print('[LOG]',log,flush=True)
    started=time.monotonic();last=started;offset=0;reason=None
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('wb') as out:
        process=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        try:
            while True:
                with log.open('rb') as f:f.seek(offset);data=f.read();offset=f.tell()
                if data:print(data.decode(errors='replace'),end='',flush=True)
                now=time.monotonic()
                if process.poll() is not None:break
                if now-last>=5:print('[RUNNING]',log.stem,'seconds',round(now-started,1),flush=True);last=now
                if now-started>timeout:reason='timeout';stop(process);break
                time.sleep(.2)
        except BaseException:stop(process);raise
    with log.open('rb') as f:
        f.seek(offset);data=f.read()
        if data:print(data.decode(errors='replace'),end='',flush=True)
    code=124 if reason else process.returncode
    print('[JOB DONE]',log.stem,'exit',code,'reason',reason,flush=True)
    return code

class Pipeline:
    def __init__(self,a,repo=None):
        self.a=a;self.repo=Path(repo).resolve() if repo is not None else Path(__file__).resolve().parents[1];os.chdir(str(self.repo))
        self.out=self.repo/'botyard_pipeline_runs'/time.strftime('%Y%m%d_%H%M%S')
        self.out.mkdir(parents=True,exist_ok=False)
        self.summary=dict(complete=False,phase='start',scales=a.scales,countPerScale=a.cache_count,
                          trainSteps=a.train_steps,numEnvs=a.envs,events=[],output=None)
        self.cache_api=load_module(self.repo,'botyard_grasp_cache')
        self.anchor_attempts={}
        print('[PIPELINE OUTPUT]',self.out,flush=True)
    def save(self):atomic_json(self.out/'summary.json',self.summary)
    def job(self,name,args,timeout):
        self.summary['phase']=name;self.save()
        code=run_job([sys.executable,'-u']+args,self.out/(name+'.log'),timeout)
        self.summary['events'].append(dict(phase=name,exit=code));self.save();return code
    def common(self,envs,scales):
        return ['pipeline=cpu','sim_device=cpu','rl_device='+self.a.rl_device,'num_subscenes=1',
                'task.env.numEnvs='+str(envs),'task.env.randomization.randomizeScaleList='+str(scales).replace(' ',''),
                'headless='+str(not self.a.viewer)]
    def seed_path(self,scale):return self.repo/'grasp_seed_banks/calibrated_v4'/(SCALE_TAG[scale]+'.json')
    def cache_path(self,scale,count):return self.repo/'cache'/('botyard_multiscale_v4_grasp_'+count_tag(count)+'_'+SCALE_TAG[scale]+'.npy')
    def validated_cache(self,scale,count):
        path=self.cache_path(scale,count)
        if not path.exists() or not self.seed_path(scale).exists():return None
        try:
            seed=json.loads(self.seed_path(scale).read_text())['candidates'][0]
            states,targets,meta=self.cache_api.read_cache(path,seed['actualJointNames'],scale,seed['handRoot7'],
                'assets/cylinder.urdf','assets/botyard_dexhand_gym/urdf/right_dexhand.urdf')
            if not meta.get('complete') or len(states)!=count or meta.get('validationVersion')!='five_side_pose_motion_v3':raise ValueError('Incomplete cache, row count or validation version mismatch')
            # SimParams.dt is returned as float32 by Preview 4. Allow only
            # representation rounding; a genuinely changed timestep is rejected.
            if not math.isclose(float(meta.get('dt',-1)),1/240,rel_tol=1e-6,abs_tol=1e-10):raise ValueError('Time step mismatch: '+str(meta.get('dt')))
            if not math.isclose(float(meta.get('pgain',-1)),3.,rel_tol=1e-6,abs_tol=1e-8) or not math.isclose(float(meta.get('dgain',-1)),.1,rel_tol=1e-6,abs_tol=1e-8) or not meta.get('torqueControl'):raise ValueError('Controller configuration mismatch')
            if meta.get('collisionConfigSha256')!=sha('grasp_seed_banks/current_verified/collision_excludes.json'):raise ValueError('Collision configuration mismatch')
            return states,targets,meta
        except (ValueError,TypeError,OSError,KeyError) as exc:
            print('[CACHE REJECTED]',path,str(exc),flush=True);return None
    def seed(self,scale,q,target,root,obj,provenance):
        c=load_module(self.repo,'botyard_pose_contract')
        return dict(schemaVersion=3,objectScale=scale,actualJointNames=c.ACTUAL,actualQ20=list(map(float,q)),
                    holdingTarget16=list(map(float,target)),handRoot7=list(map(float,root)),object7=list(map(float,obj)),
                    handAssetSha256=sha('assets/botyard_dexhand_gym/urdf/right_dexhand.urdf'),objectAssetSha256=sha('assets/cylinder.urdf'),
                    source=provenance,validationVersion='five_side_pose_motion_v3',sourceLegacyAccepted=False)
    def write_seed(self,scale,seed):atomic_json(self.seed_path(scale),dict(candidates=[seed]))
    def generate(self,scale,count,anchor=False):
        if not anchor and not self.a.fresh and self.validated_cache(scale,count):
            print('[RESUME] validated complete cache',self.cache_path(scale,count),flush=True);return True
        name=('anchor_' if anchor else 'generate_')+SCALE_TAG[scale]+'_'+count_tag(count)
        if anchor:
            attempt=self.anchor_attempts.get(scale,0);self.anchor_attempts[scale]=attempt+1
            name+='_attempt_'+str(attempt)
        report=self.out/(name+'_report.json')
        args=['gen_grasp_botyard.py','task=BotyardHandGrasp','task.env.baseObjScale='+str(scale),
              'task.env.targetGraspCount='+str(count),'task.env.maxGraspEpisodes='+str(max(1000,count*50)),
              'task.env.graspReport='+str(report)]+self.common(1,[scale])
        if anchor:args+=['task.env.graspPoseNoise=0.0','task.env.graspObjectXYNoise=0.0','task.env.graspObjectZNoise=0.0','task.env.graspStallEpisodeLimit=1','task.env.episodeLength=336']
        code=self.job(name,args,self.a.worker_timeout if anchor else self.a.cache_timeout)
        # A window-close SystemExit(0) is not a completed generation phase.
        data=json.loads(report.read_text()) if report.exists() else {}
        return code==0 and data.get('complete',False) and data.get('accepted')==count and self.validated_cache(scale,count) is not None
    def calibrate(self,scale):
        import numpy as np
        candidates=[]
        for path in getattr(self.a,'recover_cold',[]) or []:
            data=json.loads(Path(path).resolve().read_text())
            if float(data['scale'])!=scale:continue
            report=self.out/(SCALE_TAG[scale]+'_fixed_refinement.json')
            repair=bool(getattr(self.a,'repair_ff',False))
            script='tools/repair_botyard_ff_preload.py' if repair else 'tools/refine_botyard_cold_grasp.py'
            args=[script,'--repo',str(self.repo),'--source',str(Path(path).resolve()),
                  '--report',str(report),'--timeout',str(self.a.worker_timeout)]
            if getattr(self.a,'settled_window',False):args+=['--settled-window']
            if self.a.viewer:args+=['--viewer']
            if self.job(SCALE_TAG[scale]+'_fixed_refinement',args,self.a.worker_timeout*(21 if repair else 6)+30):raise RuntimeError('Fixed-wrist calibration failed; inspect its per-trial reports')
            result=json.loads(report.read_text());can=result.get('canonical')
            if not result.get('accepted') or not can:raise RuntimeError('Refinement has no confirmed canonical')
            candidates.append(self.seed(scale,can['q20'],can['holdingTarget16'],can['handRoot7'],can['object7Environment'],str(report)))
        if self.seed_path(scale).exists():candidates+=json.loads(self.seed_path(scale).read_text())['candidates'][:1]
        if scale==1.:
            # Prefer the user's settled, accepted cache to the original anchor
            # which was rejected at 2.584 degrees in the integration scene.
            old=Path('cache/botyard_measured_v3_grasp_100_s10.npy')
            if old.exists() and old.with_suffix('.meta.npz').exists():
                try:
                    with np.load(old.with_suffix('.meta.npz'),allow_pickle=False) as f:meta=json.loads(str(f['metadata'].item()))
                    states,targets,meta=self.cache_api.read_cache(old,meta['jointNames'],scale,meta['handRootPoseLocal'],'assets/cylinder.urdf','assets/botyard_dexhand_gym/urdf/right_dexhand.urdf')
                    if meta.get('complete'):
                        candidates += [self.seed(scale,q[:20],t,meta['handRootPoseLocal'],q[20:],str(old)+' row '+str(i)) for i,(q,t) in enumerate(zip(states[:8],targets[:8]))]
                except (ValueError,OSError,KeyError) as exc:print('[OLD CACHE REJECTED]',exc,flush=True)
            candidates+=json.loads(Path('grasp_seed_banks/current_verified/s10.json').read_text())['candidates'][:1]
        for seed in candidates:
            self.write_seed(scale,seed)
            if self.generate(scale,1,anchor=True):
                self.promote_anchor(scale);return
        # Other scales must physically close, lift, flip, align and cold-replay.
        from botyard_hold_recovery import candidate_order
        for index in candidate_order(self.a.canonical_trials,getattr(self.a,'candidate_order',[])):
            report=self.out/(SCALE_TAG[scale]+'_two_stage_'+str(index)+'.json')
            args=['tools/botyard_two_stage_worker.py','--repo',str(self.repo),'--report',str(report),'--scale',str(scale),
                  '--candidate',str(index),'--pose-validation']
            if getattr(self.a,'adaptive_grip',False):args+=['--adaptive-grip']
            if self.a.viewer:args+=['--viewer']
            code=self.job(SCALE_TAG[scale]+'_search_'+str(index),args,self.a.worker_timeout)
            if code:raise RuntimeError('Canonical worker failed; inspect logs rather than skipping SDK errors')
            data=json.loads(report.read_text()) if report.exists() else {}
            diagnostic=dict(candidate=index,stage=data.get('stage','missing_report'),
                reasons=data.get('stage2',data.get('stage1',data)).get('reasons',[]),
                closureContactFractions=data.get('closedGripCheck',{}).get('sideContactFractions',{}),report=str(report))
            self.summary.setdefault('canonicalSearch',{}).setdefault(SCALE_TAG[scale],[]).append(diagnostic);self.save()
            if not data.get('stage2',{}).get('passed'):
                print('[SEARCH REJECTED]',json.dumps(diagnostic),flush=True);continue
            cold=self.out/(SCALE_TAG[scale]+'_cold_'+str(index)+'.json')
            args=['tools/botyard_two_stage_worker.py','--repo',str(self.repo),'--report',str(cold),'--scale',str(scale),
                  '--replay',str(report),'--pose-validation']
            if getattr(self.a,'adaptive_grip',False):args+=['--cold-hold-seconds','8']
            if self.a.viewer:args+=['--viewer']
            code=self.job(SCALE_TAG[scale]+'_cold_'+str(index),args,self.a.worker_timeout)
            if code:raise RuntimeError('Cold worker failed')
            data=json.loads(cold.read_text()) if cold.exists() else {}
            diagnostic['coldReasons']=data.get('poseColdStartCheck',{}).get('reasons',data.get('coldStart',{}).get('reasons',[]));self.save()
            if not data.get('accepted'):
                from refine_botyard_cold_grasp import eligible
                if eligible(data):
                    refined=self.out/(SCALE_TAG[scale]+'_refined_'+str(index)+'.json')
                    args=['tools/refine_botyard_cold_grasp.py','--repo',str(self.repo),'--source',str(cold),
                          '--report',str(refined),'--timeout',str(self.a.worker_timeout)]
                    if self.a.viewer:args+=['--viewer']
                    if self.job(SCALE_TAG[scale]+'_refine_'+str(index),args,self.a.worker_timeout*6+30):
                        raise RuntimeError('Tilt-only fixed-wrist refinement failed; inspect reports before more search')
                    data=json.loads(refined.read_text());cold=refined
                    diagnostic['refinementOutcome']=data.get('refinementOutcome','accepted' if data.get('accepted') else 'rejected')
                    diagnostic['refinementRejectionReason']=data.get('refinementRejectionReason');self.save()
            can=data.get('canonical')
            if not can or not data.get('accepted'):continue
            self.write_seed(scale,self.seed(scale,can['q20'],can['holdingTarget16'],can['handRoot7'],can['object7Environment'],str(cold)))
            if self.generate(scale,1,anchor=True):self.promote_anchor(scale);return
        records=self.summary.get('canonicalSearch',{}).get(SCALE_TAG[scale],[])
        stages={stage:sum(r['stage']==stage for r in records) for stage in sorted(set(r['stage'] for r in records))}
        raise RuntimeError('No training-scene verified grasp for '+str(scale)+'. Search stage counts '+str(stages)+'. Per-candidate reasons saved in summary.json; see '+str(self.out))
    def promote_anchor(self,scale):
        states,targets,meta=self.validated_cache(scale,1)
        self.write_seed(scale,self.seed(scale,states[0,:20],targets[0],meta['handRootPoseLocal'],states[0,20:],'Accepted training-scene zero-action anchor '+str(self.cache_path(scale,1))))
        print('[CALIBRATED]',scale,'actual q20, holding target16 and measured tilted root preserved',flush=True)
    def fingerprints(self):
        return {str(s):dict(states=sha(self.cache_path(s,self.a.cache_count)),targets=sha(self.cache_path(s,self.a.cache_count).with_suffix('.meta.npz')),seed=sha(self.seed_path(s))) for s in self.a.scales}
    def proof_ok(self):
        path=Path('cache/training_replay_v4.json')
        if not path.exists():return False
        proof=json.loads(path.read_text())
        try:return bool(proof.get('passed') and proof.get('numEnvs')==self.a.envs and proof.get('countTag')==count_tag(self.a.cache_count) and proof.get('fingerprints')==self.fingerprints() and proof.get('taskSourceSha256')==sha('hora/tasks/botyard_hand_hora.py') and proof.get('configSha256')==sha('configs/task/BotyardHandHora.yaml'))
        except OSError:return False
    def check(self):
        # Always replay this run: an old successful proof is not a new run.
        proof=Path('cache/training_replay_v4.json');previous=proof.stat().st_mtime_ns if proof.exists() else 0
        args=['check_botyard_cache.py','task=BotyardHandHora','task.env.graspCacheCountTag='+count_tag(self.a.cache_count)]+self.common(self.a.envs,self.a.scales)
        if self.job('mixed_training_replay',args,self.a.check_timeout) or not self.proof_ok() or proof.stat().st_mtime_ns==previous:
            raise RuntimeError('Mixed-scale training-scene replay did not complete/pass; training blocked')
    def train(self):
        if not self.proof_ok():raise RuntimeError('Missing matching mixed-scale replay proof; run --phase check first')
        name='BotyardHandHora/multiscale_v4_'+self.out.name;self.summary['output']='outputs/'+name;self.save()
        args=['train_botyard.py','task=BotyardHandHora','task.env.graspCacheCountTag='+count_tag(self.a.cache_count),
              'train.ppo.output_name='+name,'train.ppo.max_agent_steps='+str(self.a.train_steps),
              'train.ppo.minibatch_size='+str(min(1024,self.a.envs*64)),
              'task.env.metricsPath='+str(self.out/'training_metrics.jsonl')]+self.common(self.a.envs,self.a.scales)
        # Enforce a divisible minibatch for arbitrary multiples of three.
        batch=self.a.envs*64;mini=min(1024,batch)
        while batch%mini:mini//=2
        args[args.index('train.ppo.minibatch_size='+str(min(1024,batch)))]= 'train.ppo.minibatch_size='+str(mini)
        if self.a.resume_checkpoint:args+=['checkpoint='+str(Path(self.a.resume_checkpoint).resolve())]
        print('[TRAIN OUTPUT]',self.summary['output'],flush=True)
        if self.job('train',args,self.a.train_timeout):raise RuntimeError('Training worker stopped with error/timeout')
        # Short/stopped runs need an explicit completion line, not just exit 0.
        if 'max steps achieved' not in (self.out/'train.log').read_text():raise RuntimeError('Training exited before reaching the configured budget')
    def execute(self):
        if self.a.phase in ('all','calibrate'):
            for scale in self.a.scales:self.calibrate(scale)
        if self.a.phase in ('all','generate'):
            for scale in self.a.scales:
                if not self.seed_path(scale).exists():raise RuntimeError('Calibrate first: '+str(scale))
                if not self.generate(scale,self.a.cache_count):raise RuntimeError('Cache generation incomplete for '+str(scale))
        if self.a.phase in ('all','check'):self.check()
        if self.a.phase in ('all','train'):self.train()
        self.summary.update(complete=True,phase='done');self.save()
        print('[PIPELINE DONE]',self.out,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['all','calibrate','generate','check','train'],default='all')
    p.add_argument('--scales',nargs='+',type=float,default=[1.,.9,.8]);p.add_argument('--cache-count',type=int,default=5000)
    p.add_argument('--envs',type=int,default=);p.add_argument('--train-steps',type=int,default=1000000)
    p.add_argument('--canonical-trials',type=int,default=16);p.add_argument('--viewer',action='store_true')
    p.add_argument('--adaptive-grip',action='store_true',help='Bounded lift/flip contact recovery for new canonical searches; cold targets remain frozen')
    p.add_argument('--candidate-order',nargs='+',type=int,default=[],help='Prioritize existing candidate indices without duplicating trials')
    p.add_argument('--rl-device',choices=['cpu','cuda:0'],default='cpu');p.add_argument('--resume-checkpoint');p.add_argument('--fresh',action='store_true')
    p.add_argument('--recover-cold',action='append',type=Path,default=[],help='Refine a prior tilt-only cold failure before searching that size')
    p.add_argument('--repair-ff',action='store_true',help='Use forefinger preload search on --recover-cold reports')
    p.add_argument('--settled-window',action='store_true',help='Use stable final three seconds for search; cold acceptance remains strict')
    p.add_argument('--worker-timeout',type=float,default=180);p.add_argument('--cache-timeout',type=float,default=86400)
    p.add_argument('--check-timeout',type=float,default=1800);p.add_argument('--train-timeout',type=float,default=86400)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    from botyard_hold_recovery import candidate_order
    try:candidate_order(a.canonical_trials,a.candidate_order)
    except ValueError as exc:p.error(str(exc))
    if a.repair_ff and not a.recover_cold:p.error('--repair-ff requires --recover-cold')
    if a.settled_window and (not a.recover_cold or a.repair_ff):p.error('--settled-window requires --recover-cold and cannot combine with --repair-ff')
    if len(set(a.scales))!=len(a.scales) or any(s not in SCALE_TAG for s in a.scales):p.error('scales must be unique values among 1.0 0.9 0.8')
    if a.cache_count<1 or a.envs<1 or a.envs%len(a.scales) or a.train_steps<64*a.envs or not 1<=a.canonical_trials<=256:p.error('Invalid count, envs, budget or trial cap')
    if any(not 0<x<float('inf') for x in [a.worker_timeout,a.cache_timeout,a.check_timeout,a.train_timeout]):p.error('Timeouts must be finite positive seconds')
    if a.dry_run:
        print(json.dumps(dict(phase=a.phase,scales=a.scales,cacheRowsPerScale=a.cache_count,numEnvs=a.envs,trainAgentSteps=a.train_steps,
            order=['per-scale physics calibration','per-scale full cache generation','mixed-layout zero-action replay','Stage1 PPO'],
            physics='cpu',rlDevice=a.rl_device,initialGraspNotYetVerified=True),indent=2));return 0
    pipeline=Pipeline(a)
    try:pipeline.execute();return 0
    except KeyboardInterrupt:pipeline.summary['error']='interrupted';pipeline.save();return 130
    except Exception as exc:
        import traceback
        traceback.print_exc();pipeline.summary['error']=str(exc);pipeline.save();print('[HALT]',exc,flush=True);return 1

if __name__=='__main__':sys.exit(main())
