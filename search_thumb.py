"""BotYard fixed-root grasp-release screening (NO PPO, NO grasp cache overwrite).

Run inside existing HORA checkout, see README. Uses the user's installed task and
its exact PhysX/PD/mimic implementation. Tests release candidates from canonical.
This is a first-stage *tilted-root* screen, NOT an upright validation.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

import hydra
import numpy as np
import isaacgym  # must be imported before torch in Isaac Gym Preview 4
import torch
from omegaconf import DictConfig, OmegaConf

# Match the HORA custom OmegaConf resolvers registered in train_botyard.py.
# The grasp search is its own Hydra entrypoint and must register them too.
for _resolver_name, _resolver_fn in (
    ('eq', lambda x, y: str(x).lower() == str(y).lower()),
    ('contains', lambda x, y: str(x).lower() in str(y).lower()),
    ('if', lambda pred, a, b: a if pred else b),
    ('resolve_default', lambda default, arg: default if arg == '' else arg),
):
    if not OmegaConf.has_resolver(_resolver_name):
        OmegaConf.register_new_resolver(_resolver_name, _resolver_fn)
from isaacgym import gymtorch

from hora.tasks.botyard_hand_hora import BotyardHandHoraVecTask

FINGER_INDICES = {
    '1': (12, 13, 14, 15),  # thumb THJ4,3,2,1
    '2': (0, 1, 2),        # index FFJ4,3,2
    '3': (6, 7, 8),        # middle MFJ4,3,2
    '4': (9, 10, 11),      # ring RFJ4,3,2
    '5': (3, 4, 5),        # little LFJ4,3,2
}
# Only distal flexion is backed off toward joint's less-curled target.
RELEASE_FLEX = {'1':(14,15), '2':(1,2), '3':(7,8), '4':(10,11), '5':(4,5)}
# Required: 125, 134, all one-finger release cases, including no-thumb 2345.
GROUPS = ['125', '134', '1234', '1235', '1245', '1345', '2345']


def quat_tilt(q):
    x,y,z,w=[float(v) for v in q]
    cos=1-2*(x*x+y*y)
    return math.degrees(math.acos(max(-1.,min(1.,abs(cos)))))


class Search:
    def __init__(self, cfg, args):
        self.args=args
        # Construct one environment and omit loading old 50k cache.
        t=OmegaConf.to_container(cfg.task,resolve=True)
        t['env']['numEnvs']=1
        t['env']['genGrasps']=True
        t['env']['trainingMode']='constrained'
        t['env']['forceScale']=0.0
        t['env']['randomForceProbScalar']=0.0
        t['env']['randomization']['randomizeScaleList']=[1.0]
        t['env']['randomization']['randomizeScale']=True
        t['env']['randomization']['scaleListInit']=True
        t['env']['cGait']={'enabled':False}
        t['env']['debugMetricsEvery']=0
        t['env']['metricsPath']=''
        t['rl_device']='cuda:0'
        self.task=BotyardHandHoraVecTask(t,'cuda:0',-1,True)
        self.gym=self.task.gym
        self.sim=self.task.sim
        self.d=float(self.task.dt)
        self.freq=int(self.task.control_freq_inv)
        self.device=self.task.device
        self.canonical=self.task.canonical_policy_pose.detach().clone()
        self.low=self.task.policy_dof_lower_limits.clone()
        self.high=self.task.policy_dof_upper_limits.clone()
        self.seed=self.task.scale_seeds['1.0']
        self.seed_q=torch.as_tensor(self.seed['actualQ20'],device=self.device,dtype=torch.float32)
        self.seed_obj=torch.as_tensor(self.seed['object7'],device=self.device,dtype=torch.float32)
        self.initial_global_obj=self.seed_obj.clone()
        self.initial_global_obj[:3]+=self.task.state_offsets[0]
        self.bodies=self.task.finger_body_groups
        self.object_body=self.task.object_body_handle
        self.sim_steps=0
        self.parent_pool=[]

    def set_targets(self, p):
        p=torch.minimum(torch.maximum(p,self.low),self.high)
        self.task.prev_policy_targets[0]=p
        self.task.cur_policy_targets[0]=p
        full=self.task.expand_policy_targets(p.unsqueeze(0))[0]
        self.task.prev_targets[0,:20]=full
        self.task.cur_targets[0,:20]=full

    def reset(self, candidate):
        t=self.task
        t.allegro_hand_dof_pos[0]=self.seed_q
        t.allegro_hand_dof_vel[0]=0
        t.root_state_tensor[t.object_indices[0],:7]=self.initial_global_obj
        t.root_state_tensor[t.object_indices[0],7:13]=0
        self.set_targets(candidate)
        hi=t.hand_indices.to(torch.int32)
        oi=t.object_indices.to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(t.dof_state),gymtorch.unwrap_tensor(hi),1)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(t.root_state_tensor),gymtorch.unwrap_tensor(oi),1)
        if not t.torque_control:
            self.gym.set_dof_position_target_tensor_indexed(self.sim,gymtorch.unwrap_tensor(t.prev_targets),gymtorch.unwrap_tensor(hi),1)
        t._refresh_gym()

    def step(self):
        # Use the same low-level per-substep PD controller as BotYard VecTask.
        self.task.update_low_level_control()
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim,True)
        self.task._refresh_gym()
        self.sim_steps+=1

    def contacts(self):
        # get_env_rigid_contacts is available in Isaac Gym CPU pipeline.
        record=[]
        try:
            raw=self.gym.get_env_rigid_contacts(self.task.envs[0])
            for contact in raw:
                a=int(contact['body0']);b=int(contact['body1'])
                if a==self.object_body and b in self.bodies:
                    record.append(self.bodies[b])
                elif b==self.object_body and a in self.bodies:
                    record.append(self.bodies[a])
        except (AttributeError,TypeError,ValueError,KeyError) as exc:
            raise RuntimeError('Unable to read body-pair contacts; do not claim release success') from exc
        abbrev={'th':'1','ff':'2','mf':'3','rf':'4','lf':'5'}
        return set(abbrev[f] for f in record)

    def pose(self):
        t=self.task
        obj=t.object_pose[0].detach().cpu().numpy()
        diff=obj[:3]-self.initial_global_obj[:3].detach().cpu().numpy()
        return {'drift_mm':round(float(np.linalg.norm(diff))*1000,3),
                'horizontal_mm':round(float(np.linalg.norm(diff[:2]))*1000,3),
                'drop_mm':round(float(max(0.,-diff[2]))*1000,3),
                'tilt_deg':round(quat_tilt(obj[3:]),3),
                'object7':[float(v) for v in obj]}

    def accepted_geometry(self, values):
        return (values['horizontal_mm'] <= self.args.max_horizontal_mm and
                values['drop_mm'] <= self.args.max_drop_mm and
                values['tilt_deg'] <= self.args.max_tilt_deg)

    def dwell(self, n, group=None, released=None):
        max_h,max_drop,max_tilt=0.,0.,0.
        contact_fracs={x:0 for x in '12345'}
        violations=0
        final=None
        for _ in range(n):
            self.step()
            p=self.pose()
            cc=self.contacts()
            final=p
            max_h=max(max_h,p['horizontal_mm'])
            max_drop=max(max_drop,p['drop_mm'])
            max_tilt=max(max_tilt,p['tilt_deg'])
            for x in cc:contact_fracs[x]+=1
            if not self.accepted_geometry(p):violations+=1
        return {'max_horizontal_mm':max_h,'max_drop_mm':max_drop,'max_tilt_deg':max_tilt,
                'contact_fraction':{k:round(v/max(1,n),3) for k,v in contact_fracs.items()},
                'violation_steps':violations,'last_pose':final}

    def move(self, src,dst,n):
        for k in range(n):
            u=(k+1)/n
            # Per-sim-step target ramp, bounded in URDF limits.
            self.set_targets(src+(dst-src)*u)
            self.step()
        return self.pose()

    def thumb_targets(self, candidate):
        """Multiple escape directions: a smaller thumb joint angle is not always an opening."""
        choices=[]
        for direction in ('lower','upper','mixed'):
            p=candidate.clone()
            for i in FINGER_INDICES['1']:
                if direction=='lower': endpoint=self.low[i]
                elif direction=='upper': endpoint=self.high[i]
                else: endpoint=self.high[i] if i in (12,13) else self.low[i]
                p[i]=candidate[i]+self.args.release_fraction*(endpoint-candidate[i])
            choices.append((direction,p))
        return choices

    def release_target(self,candidate,group):
        target=candidate.clone()
        for finger in '12345':
            if finger in group: continue
            for idx in RELEASE_FLEX[finger]:
                # WARNING: must verify contact disappearance; lower angular value
                # does not guarantee opening for every thumb joint.
                target[idx] += self.args.release_fraction*(self.low[idx]-target[idx])
        return target

    def thumb_test(self,candidate, direction, released_target):
        """Evaluate the entire ramp + independent hold. Never call a contactful thumb released."""
        self.reset(candidate)
        settle=self.dwell(max(1,round(self.args.settle_s/self.d)))
        result={'group':'2345','direction':direction,'passed':False,'settle':settle,
                'reason':'', 'safe_detached_steps':0,'max_consecutive_safe_detached_steps':0}
        if settle['violation_steps']:
            result['reason']='unstable_initial_hold';return result
        # Require thumb contact before release, or this is not a genuine thumb-release test.
        if settle['contact_fraction']['1']<.2:
            result['reason']='thumb_not_initially_contacting';return result
        n_release=max(1,round(self.args.release_s/self.d))
        n_hold=max(1,round(self.args.hold_s/self.d))
        max_h=max_drop=max_tilt=0.
        thumb_contact_steps=0
        supporting_steps=0
        safe_detached=0
        consecutive=0
        best_consecutive=0
        release_errors=[]
        last_pose=None
        for k in range(n_release+n_hold):
            if k<n_release:
                alpha=(k+1)/n_release
                self.set_targets(candidate+(released_target-candidate)*alpha)
            self.step()
            pos=self.pose(); cc=self.contacts(); last_pose=pos
            max_h=max(max_h,pos['horizontal_mm'])
            max_drop=max(max_drop,pos['drop_mm'])
            max_tilt=max(max_tilt,pos['tilt_deg'])
            if '1' in cc: thumb_contact_steps+=1
            support=sum(f in cc for f in '2345')
            if support>=3: supporting_steps+=1
            safe=(self.accepted_geometry(pos) and '1' not in cc and support>=3)
            if safe:
                safe_detached+=1;consecutive+=1
                best_consecutive=max(best_consecutive,consecutive)
            else:consecutive=0
            if k>=n_release and not self.accepted_geometry(pos):
                release_errors.append('geometry')
        hold_thumb_fraction=thumb_contact_steps/(n_release+n_hold)
        result.update({'safe_detached_steps':safe_detached,
                       'max_consecutive_safe_detached_steps':best_consecutive,
                       'safe_detached_seconds':round(safe_detached*self.d,4),
                       'max_consecutive_safe_detached_seconds':round(best_consecutive*self.d,4),
                       'thumb_contact_fraction':round(hold_thumb_fraction,3),
                       'support_ge3_fraction':round(supporting_steps/(n_release+n_hold),3),
                       'max_horizontal_mm':max_h,'max_drop_mm':max_drop,
                       'max_tilt_deg':max_tilt,'last_pose':last_pose,
                       'target16':[float(v) for v in released_target.detach().cpu()]})
        # Strong terminal condition: measure the last hold window, not only an instant.
        tail_steps=max(1,round(.2/self.d))
        # A full-window detached, stable hold is certified by the consecutive metric.
        result['passed']=(best_consecutive>=min(n_hold,tail_steps) and
                          safe_detached>=int(.8*n_hold) and
                          max_h<=self.args.max_horizontal_mm and
                          max_drop<=self.args.max_drop_mm and
                          max_tilt<=self.args.max_tilt_deg)
        if result['passed']: result['reason']='passed'
        elif safe_detached==0 and hold_thumb_fraction>.8:result['reason']='thumb_contact_not_released'
        elif safe_detached==0:result['reason']='insufficient_four_finger_support_or_instability'
        else:result['reason']='partial_detachment_but_not_sustained'
        # Smooth score even when every candidate fails hard success threshold.
        # Safe 4-finger detached duration dominates; penalize tilt/drop and residual thumb contact.
        result['fitness']=(8.0*safe_detached*self.d + 5.0*best_consecutive*self.d
                            + 1.0*supporting_steps/(n_release+n_hold)
                            - 1.2*hold_thumb_fraction
                            - .02*max_h-.025*max_drop-.02*max_tilt)
        return result

    def one_test(self,candidate,group):
        self.reset(candidate)
        start=self.dwell(round(self.args.settle_s/self.d))
        if start['violation_steps']:
            return {'group':group,'passed':False,'reason':'unstable_initial_hold','settle':start}
        initial_support=sum(start['contact_fraction'][f]>.1 for f in group)
        if initial_support<2:
            return {'group':group,'passed':False,'reason':'insufficient_initial_support','settle':start}
        release=self.release_target(candidate,group)
        self.move(candidate,release,max(1,round(self.args.release_s/self.d)))
        held=self.dwell(max(1,round(self.args.hold_s/self.d)))
        detached=[f for f in '12345' if f not in group]
        release_ok=all(held['contact_fraction'][f] < .05 for f in detached)
        supported=sum(held['contact_fraction'][f]>.5 for f in group)
        stable=(held['violation_steps']==0 and supported>=3 and release_ok)
        return {'group':group,'passed':stable,'settle':start,'hold':held,
                'removed':detached,'released_fingers_contact_free':release_ok,
                'actual_support_fingers':supported,'required_support':3,
                'target16':[float(v) for v in release.detach().cpu()]}

    def angular_spread_deg(self):
        """Angular separation of currently contacting fingertip positions about world z."""
        cc=self.contacts()
        xyz=self.task.rigid_body_states[0,:,0:3].detach().cpu().numpy()
        center=self.task.object_pos[0].detach().cpu().numpy()
        angles=[]
        for finger, handle in zip(('2','5','3','4','1'),self.task.fingertip_handles):
            if finger in cc:
                v=xyz[handle,:2]-center[:2]
                angles.append(math.atan2(float(v[1]),float(v[0])))
        if len(angles)<3: return 0.0
        angles.sort()
        gaps=[(angles[(i+1)%len(angles)]-angles[i])%(2*math.pi) for i in range(len(angles))]
        # Smaller max empty sector generally means better angular coverage.
        return round(math.degrees(2*math.pi-max(gaps)),2)

    def make_candidate(self, rng, k):
        if k==0:return self.canonical.clone(),-1
        # Evolution: mutate one of the best parent poses rather than sampling around
        # the original every time. Interleave some exploratory canonical restarts.
        use_parent=(len(self.parent_pool)>0 and k%7!=0)
        if use_parent:
            upper=min(self.args.elite,len(self.parent_pool))
            parent_i=int(rng.randint(upper))
            base=self.parent_pool[parent_i]['q']
        else:
            parent_i=-1;base=self.canonical
        noise=np.zeros(16,dtype=np.float32)
        # Bigger J4 exploration plus finger flexion changes for contact geometry.
        scale=max(.3,1.0-k/max(1,self.args.candidates)*.65)
        for j in (0,3,6,9):
            noise[j]=rng.normal(0,self.args.lateral_rad*scale)
        for j in (1,2,4,5,7,8,10,11):
            noise[j]=rng.normal(0,self.args.flex_rad*scale)
        # Thumb holding pose may change minimally to allow more support redistribution.
        if k%4==0:
            for j in (12,13,14,15):
                noise[j]=rng.normal(0,self.args.thumb_rad*scale)
        q=base+torch.as_tensor(noise,device=self.device)
        q=torch.minimum(torch.maximum(q,self.low),self.high)
        return q,parent_i

    def run(self):
        rng=np.random.RandomState(self.args.seed)
        output=Path(self.args.output)
        output.parent.mkdir(exist_ok=True,parents=True)
        if output.exists() and not self.args.overwrite:
            raise FileExistsError(f'{output} already exists. Use a different --output or --overwrite.')
        if self.args.overwrite:output.write_text('')
        results=[]
        print('[THUMB V3] Tilted-root search; 2345 first, continuous metrics and elite mutations.',flush=True)
        for k in range(self.args.candidates):
            q,parent_i=self.make_candidate(rng,k)
            # Test multiple thumb directions; record all rather than assume low-angle opens.
            thumb_attempts=[self.thumb_test(q,direction,goal) for direction,goal in self.thumb_targets(q)]
            thumb=max(thumb_attempts,key=lambda d:(d['passed'],d.get('fitness',-999.)))
            # Secondary groups evaluated only for thumb promising candidates or periodic baselines.
            secondary=[]
            if thumb['passed'] or k==0 or k%self.args.secondary_every==0:
                secondary=[self.one_test(q,g) for g in GROUPS if g!='2345']
            passed=[v['group'] for v in secondary if v['passed']]
            if thumb['passed']:passed.append('2345')
            # Strong primary objective, small tie-break for secondary versatility.
            score=float(thumb.get('fitness',-100.))+ (20. if thumb['passed'] else 0.) + .2*len(passed)
            record={'candidate':k,'parent_rank':parent_i,'fitness':round(score,5),
                    'thumb':thumb,'thumb_all_directions':thumb_attempts,
                    'passed_groups':passed,'policyTargets16':[float(v) for v in q.detach().cpu()],
                    'secondary_tests':secondary,'status':'TILTED_ROOT_SCREEN_ONLY'}
            results.append(record)
            self.parent_pool.append({'fitness':score,'q':q.clone()})
            self.parent_pool.sort(key=lambda d:d['fitness'],reverse=True)
            self.parent_pool=self.parent_pool[:max(1,self.args.elite)]
            with output.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            print(f'[THUMB SEARCH] {k+1}/{self.args.candidates} fitness={score:.3f} '
                  f'2345={thumb["passed"]} detached={thumb.get("max_consecutive_safe_detached_seconds",0):.3f}s '
                  f'direction={thumb["direction"]} reason={thumb["reason"]} '
                  f'other={passed}',flush=True)
        result=output.with_suffix('.top10.json')
        best=sorted(results,key=lambda r:r['fitness'],reverse=True)[:min(10,len(results))]
        result.write_text(json.dumps(best,ensure_ascii=False,indent=2))
        print('[RESULT]',result,flush=True)


@hydra.main(config_name='config',config_path='configs')
def main(cfg:DictConfig):
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidates',type=int,default=30)
    parser.add_argument('--elite',type=int,default=8)
    parser.add_argument('--secondary-every',type=int,default=10)
    parser.add_argument('--thumb-rad',type=float,default=.02)
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--lateral-rad',type=float,default=.10)
    parser.add_argument('--flex-rad',type=float,default=.06)
    parser.add_argument('--release-fraction',type=float,default=.65)
    parser.add_argument('--settle-s',type=float,default=.5)
    parser.add_argument('--release-s',type=float,default=.3)
    parser.add_argument('--hold-s',type=float,default=.6)
    parser.add_argument('--max-horizontal-mm',type=float,default=8.)
    parser.add_argument('--max-drop-mm',type=float,default=8.)
    parser.add_argument('--max-tilt-deg',type=float,default=10.)
    parser.add_argument('--output',default='grasp_search/thumb_v3.jsonl')
    # Separate script args from Hydra overrides with a literal -- separator.
    args=parser.parse_args(getattr(sys,'_grasp_search_extra',[]))
    Search(cfg,args).run()

if __name__=='__main__':
    # Hydra sees only config overrides; argparse sees the arguments after --.
    if '--' in sys.argv:
        boundary=sys.argv.index('--')
        extra=sys.argv[boundary+1:]
        sys.argv=sys.argv[:boundary]
        sys.argv.extend([])
        # parsed args must be saved because main() sees cleaned sys.argv.
        setattr(sys,'_grasp_search_extra',extra)
    main()
