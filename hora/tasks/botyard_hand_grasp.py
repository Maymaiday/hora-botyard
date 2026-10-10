"""Parallel generation from measured poses in the exact training scene."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch
from .botyard_hand_hora import BotyardHandHoraVecTask
from .botyard_grasp_cache import write_cache, count_tag
from .botyard_pose_contract import ACTUAL, POLICY, VALIDATION_VERSION, atomic_json
from .botyard_generation_batch import BatchTrace, observe_batch, sample_batch, restore_partial, LocalSampler
from collections import Counter

class BotyardHandGrasp(BotyardHandHoraVecTask):
    def __init__(self,config,sim_device,graphics_device_id,headless):
        super().__init__(config,sim_device,graphics_device_id,headless)
        e=config['env']
        if not self.torque_control:raise ValueError('Measured seed requires torque control')
        if any((self.randomize_scale,self.randomize_mass,self.randomize_pd_gains,self.randomize_friction,self.randomize_com)):
            raise ValueError('Generation requires exact physical parameters, no domain randomization')
        bank=Path(e['botyard']['scaleSeedFiles'][str(float(self.base_obj_scale))]);self.seeds=json.loads(bank.read_text())['candidates']
        if not self.seeds:raise ValueError('Empty measured seed bank')
        for seed in self.seeds:
            if seed.get('schemaVersion')!=3 or seed['actualJointNames']!=self.sim_dof_names:raise ValueError('Measured seed joint contract mismatch')
            if not np.isclose(seed['objectScale'],self.base_obj_scale):raise ValueError('Each size needs its own measured seed; do not reuse the 1.0 pose')
            if not np.allclose(seed['handRoot7'],self.fixed_hand_root_local.cpu().numpy(),atol=2e-6):raise ValueError('Measured tilted root was changed')
            for key,path in [('handAsset',e['asset']['handAsset']),('objectAsset',e['object']['assetFile'])]:
                if seed[key+'Sha256']!=hashlib.sha256(Path(path).read_bytes()).hexdigest():raise ValueError('Measured seed asset mismatch: '+path)
        self.goal=int(e['targetGraspCount']);self.warmup=float(e.get('graspWarmupSeconds',.4))
        if self.goal<1 or self.max_episode_length*self.control_freq_inv*self.dt<self.warmup+3-1e-6:
            raise ValueError('Need positive target count and at least .4s warmup + 3s hold')
        self.trace=BatchTrace(self.max_episode_length*self.control_freq_inv,self.num_envs)
        self.states=[];self.targets=[];self.fingerprints=set();self.completed=0;self.last_success=0;self.started=time.monotonic()
        self.next_checkpoint=int(e.get("graspCheckpointEvery",100));self.batch_reasons={};self.batches=0
        self.terminal_pass=np.zeros(self.num_envs,dtype=bool);self.last_checks=[];self.last_check=None;self.report_path=Path(e['graspReport']);self.initial_pose=None
        self.physics_observer=self._observe
        self.metadata=dict(jointNames=self.sim_dof_names,objectScale=float(self.base_obj_scale),
            handRootPoseLocal=self.fixed_hand_root_local.cpu().numpy().tolist(),objectMassKg=float(e['object']['massKg']),
            friction=float(e['object']['fixedFriction']),torqueControl=bool(self.torque_control),pgain=float(e['controller']['pgain']),
            dgain=float(e['controller']['dgain']),controlFrequencyInv=self.control_freq_inv,dt=float(self.dt),
            validationVersion=VALIDATION_VERSION,sdkVelocityIsDiagnosticOnly=True,
            validation='3s gravity hold; five finger side groups, no palm/cap/ground support, pose motion per physics step; not force-closure certification',
            maxDriftM=float(e['maxGraspDrift']),maxDropM=float(e['maxGraspDrop']),warmupSeconds=self.warmup,maxTiltDeg=float(e['maxCylinderTiltDeg']),seedBank=str(bank),generationNumEnvs=self.num_envs,
            collisionConfigSha256=hashlib.sha256(Path(e['botyard']['collisionExcludes']).read_bytes()).hexdigest(),
            solverPositionIterations=8,solverVelocityIterations=4,armature=.001,torqueLimit=.5,
            simulationBackend=self.simulation_backend)
        for key,path in [('objectAsset',e['object']['assetFile']),('handAsset',e['asset']['handAsset'])]:
            self.metadata[key]=path;self.metadata[key+'Sha256']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.sampler=LocalSampler(bool(e.get('graspAdaptiveSampling',False)))
        self.observer_seconds=0.;self.check_seconds=0.;self.batch_started=time.monotonic();self.batch_timings={}
        partial=Path('cache')/('%s_grasp_%s_s%s.partial.npy'%(self.grasp_cache_name,count_tag(self.goal),str(self.base_obj_scale).replace('.','')))
        if e.get('graspResumePartial',False) and partial.exists():
            states,targets,old=restore_partial(partial,self.metadata)
            old_backend=old.get('simulationBackend',dict(simDevice='cpu',physics='cpu',tensorPipeline='cpu',controllerFeedback='every_physics_step'))
            if old_backend!=self.simulation_backend:
                raise ValueError('Partial cache was collected on a different physics backend. Use --phase available to replay/train existing rows; resume its generation with --sim-device cpu.')
            if len(states)>self.goal:raise ValueError('Partial cache larger than requested count')
            self.states=[r.copy() for r in states];self.targets=[r.copy() for r in targets]
            self.fingerprints={tuple(np.rint(r[:20]/.0005).astype(int)) for r in states}
            if len(self.fingerprints)!=len(states):raise ValueError('Partial cache has duplicate joint fingerprints')
            self.completed=int(old['completedEpisodes']);self.last_success=self.completed
            self.next_checkpoint=len(states)+int(e.get('graspCheckpointEvery',100))
            self.metadata['resumedRows']=len(states);self.metadata['resumedFrom']=str(partial)
            factor=float(old.get('generationSamplingFactor',1.))
            if self.sampler.enabled and len(states)/max(1,self.completed)<.25:factor=min(factor,.5)
            self.sampler=LocalSampler(self.sampler.enabled,factor if self.sampler.enabled else 1.)
            self.metadata['priorSamplingHistory']=old.get('samplingHistory',[])
            print('[RESUME PARTIAL]',len(states),'accepted rows; tested',self.completed,'proposal factor',self.sampler.factor,flush=True)
        if len(self.states)>=self.goal:self._save(False);self._report(True);raise SystemExit(0)
        print('[GEN] parallel environments',self.num_envs,'independent actual q20 + target16 + object7; acceptance='+VALIDATION_VERSION,flush=True)

    def _observe(self,task):
        started=time.monotonic();observe_batch(task,self.trace);self.observer_seconds+=time.monotonic()-started

    def _report(self,complete=False):
        atomic_json(self.report_path,dict(complete=complete,accepted=len(self.states),target=self.goal,tested=self.completed,
                    elapsedSeconds=round(time.monotonic()-self.started,2),lastCheck=self.last_check,metadata=self.metadata,
                    generationNumEnvs=self.num_envs,batches=self.batches,lastBatchReasonCounts=self.batch_reasons,
                    lastBatchTimings=getattr(self,"batch_timings",{}),proposalFactor=getattr(getattr(self,"sampler",None),"factor",1.)))

    def _save(self,partial):
        if not self.states:return
        name='%s_grasp_%s_s%s%s.npy'%(self.grasp_cache_name,count_tag(self.goal),str(self.base_obj_scale).replace('.',''),'.partial' if partial else '')
        if hasattr(self,'sampler'):
            self.metadata.update(generationSamplingFactor=self.sampler.factor,samplingHistory=self.sampler.history,
                baseSamplingNoise=dict(jointRad=float(self.config['env']['graspPoseNoise']),xyM=float(self.config['env']['graspObjectXYNoise']),zM=float(self.config['env']['graspObjectZNoise'])))
        write_cache(Path('cache')/name,np.array(self.states[:self.goal]),np.array(self.targets[:self.goal]),
                    dict(self.metadata,complete=not partial,completedEpisodes=self.completed))
        print('[SAVE]',name,'rows',min(len(self.states),self.goal),'complete',not partial,flush=True)

    def reset_idx(self,ids):
        if len(ids)!=self.num_envs:raise RuntimeError('Generation requires synchronized full-batch resets')
        if self.trace.count:
            self.completed+=len(ids);self.batches+=1
            poses=self.root_state_tensor[self.object_indices,:7].cpu().numpy().copy()
            poses[:,:3]-=self.state_offsets.cpu().numpy()
            joints=self.allegro_hand_dof_pos.cpu().numpy();targets=self.cur_policy_targets.cpu().numpy()
            for index in ids.cpu().numpy().tolist():
                if not self.terminal_pass[index]:continue
                state=np.r_[joints[index],poses[index]]
                key=tuple(np.rint(state[:20]/.0005).astype(int))
                if key in self.fingerprints:continue
                self.fingerprints.add(key);self.states.append(state.copy());self.targets.append(targets[index].copy());self.last_success=self.completed
                if len(self.states)>=self.goal:break
            if hasattr(self,'sampler'):
                self.batch_timings=dict(wallSeconds=round(time.monotonic()-self.batch_started,3),observerSeconds=round(self.observer_seconds,3),acceptanceSeconds=round(self.check_seconds,3))
                event=self.sampler.update(int(self.terminal_pass.sum()),self.num_envs,len(self.states))
                if event:print('[SAMPLING] reduced proposal width; acceptance unchanged',event,flush=True)
            self._report(complete=len(self.states)>=self.goal)
            print('[BATCH]',self.batches,'tested',self.completed,'accepted',len(self.states),'/',self.goal,
                  'reasonCounts',self.batch_reasons,'timings',getattr(self,'batch_timings',{}),'proposalFactor',getattr(getattr(self,'sampler',None),'factor',1.),'elapsedSeconds',round(time.monotonic()-self.started,1),flush=True)
            if len(self.states)>=self.goal:self._save(False);raise SystemExit(0)
            if len(self.states)>=self.next_checkpoint:
                self._save(True)
                every=int(self.config['env'].get('graspCheckpointEvery',100))
                self.next_checkpoint=(len(self.states)//every+1)*every
            stall=max(int(self.config['env']['graspStallEpisodeLimit']),self.num_envs*4)
            if self.completed>=int(self.config['env']['maxGraspEpisodes']) or self.completed-self.last_success>=stall:
                self._save(True);raise RuntimeError('Generation stalled; inspect '+str(self.report_path)+'. Partial cache is not accepted for training.')
        e=self.config['env'];factor=getattr(getattr(self,'sampler',None),'factor',1.)
        q,target,pose=sample_batch(self.seeds,len(ids),np.random,self.sim_dof_names,
            self.allegro_hand_dof_lower_limits.cpu().numpy(),self.allegro_hand_dof_upper_limits.cpu().numpy(),
            self.policy_dof_lower_limits.cpu().numpy(),self.policy_dof_upper_limits.cpu().numpy(),
            float(e['graspPoseNoise'])*factor,float(e['graspObjectXYNoise'])*factor,float(e['graspObjectZNoise'])*factor)
        self.initial_pose=pose.copy();self.trace.clear();self.terminal_pass[:]=False
        self.observer_seconds=0.;self.check_seconds=0.;self.batch_started=time.monotonic()
        self.allegro_hand_dof_pos[ids]=to_torch(q,device=self.device);self.allegro_hand_dof_vel[ids]=0
        target_t=to_torch(target,device=self.device);self.holding_reference[ids]=target_t;full=self.expand_policy_targets(target_t)
        self.prev_policy_targets[ids]=target_t;self.cur_policy_targets[ids]=target_t;self.prev_targets[ids]=full;self.cur_targets[ids]=full
        self.init_pose_buf[ids]=self.allegro_hand_dof_pos[ids]
        oi=self.object_indices[ids].to(torch.int32);hi=self.hand_indices[ids].to(torch.int32)
        self.root_state_tensor[oi.long(),:7]=to_torch(pose,device=self.device);self.root_state_tensor[oi.long(),:3]+=self.state_offsets[ids];self.root_state_tensor[oi.long(),7:]=0
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_state_tensor),gymtorch.unwrap_tensor(oi),len(oi))
        self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.dof_state),gymtorch.unwrap_tensor(hi),len(hi))
        self.progress_buf[ids]=0;self.reset_buf[ids]=0;self.obs_buf[ids]=0;self.rb_forces[ids]=0;self.proprio_hist_buf[ids]=0;self.at_reset_buf[ids]=1
        self._report()

    def compute_reward(self,actions):
        self.rew_buf[:]=0;self.reset_buf[:]=0
        if torch.any(self.progress_buf>=self.max_episode_length):
            if not torch.all(self.progress_buf>=self.max_episode_length):raise RuntimeError('Generation batch lost synchronization')
            e=self.config['env'];started=time.monotonic()
            self.last_checks=[self.trace.check(i,self.initial_pose[i],self.base_obj_scale,self.dt,self.warmup,
                float(e['maxCylinderTiltDeg']),float(e['maxGraspDrift']),float(e['maxGraspDrop'])) for i in range(self.num_envs)]
            self.check_seconds=time.monotonic()-started
            self.last_check=self.last_checks[-1]
            self.terminal_pass[:]=[c['passed'] for c in self.last_checks]
            self.batch_reasons=dict(Counter(r for c in self.last_checks for r in c['reasons']))
            self.reset_buf[:]=1
