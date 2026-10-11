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

    def release_target(self,candidate,group):
        target=candidate.clone()
        for finger in '12345':
            if finger in group: continue
            for idx in RELEASE_FLEX[finger]:
                # WARNING: must verify contact disappearance; lower angular value
                # does not guarantee opening for every thumb joint.
                target[idx] += self.args.release_fraction*(self.low[idx]-target[idx])
        return target

    def one_test(self,candidate,group):
        self.reset(candidate)
        start=self.dwell(round(self.args.settle_s/self.d))
        if start['violation_steps']:
            return {'group':group,'passed':False,'reason':'unstable_initial_hold','settle':start}
        # Requires the commanded group to bear measured object contact.
        initial_support=sum(start['contact_fraction'][f]>.1 for f in group)
        if initial_support<2:
            return {'group':group,'passed':False,'reason':'insufficient_initial_support','settle':start}
        release=self.release_target(candidate,group)
        self.move(candidate,release,max(1,round(self.args.release_s/self.d)))
        held=self.dwell(max(1,round(self.args.hold_s/self.d)))
        detached=[f for f in '12345' if f not in group]
        release_ok=all(held['contact_fraction'][f] < .05 for f in detached)
        supported=sum(held['contact_fraction'][f]>.5 for f in group)
        # At least 2 actual contact-bearing fingers; most groups require >= 3.
        required_support=3
        stable=(held['violation_steps']==0 and supported>=required_support and release_ok)
        out={'group':group,'passed':stable,'settle':start,'hold':held,
             'removed':detached,'released_fingers_contact_free':release_ok,
             'actual_support_fingers':supported,'required_support':required_support,
             'target16':[float(v) for v in release.detach().cpu()]}
        return out

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

    def run(self):
        rng=np.random.RandomState(self.args.seed)
        output=Path(self.args.output)
        output.parent.mkdir(exist_ok=True,parents=True)
        lateral=[0,3,6,9] # FF,LF,MF,RF J4, bounds verified by URDF only
        seen=[]
        print('Tilted-root canonical search; do not interpret these as upright tests.',flush=True)
        for k in range(self.args.candidates):
            q=self.canonical.clone()
            if k:
                # Local perturbation around canonical; never change object or root pose.
                noise=np.zeros(16,dtype=np.float32)
                for j in lateral: noise[j]=rng.uniform(-self.args.lateral_rad,self.args.lateral_rad)
                for j in (1,4,7,10,13): noise[j]=rng.uniform(-self.args.flex_rad,self.args.flex_rad)
                q=torch.minimum(torch.maximum(q+torch.as_tensor(noise,device=self.device),self.low),self.high)
            self.reset(q)
            self.dwell(max(1,round(self.args.settle_s/self.d)))
            spread=self.angular_spread_deg()
            tests=[self.one_test(q,g) for g in GROUPS]
            passed=[v['group'] for v in tests if v['passed']]
            score=(2*sum(g in passed for g in ('125','134')) + len(passed))
            record={'candidate':k,'canonical_start':k==0,'score':score,'contact_angular_coverage_deg':spread,'passed_groups':passed,
                    'policyTargets16':[float(v) for v in q.detach().cpu()],
                    'tests':tests,'status':'TILTED_ROOT_SCREEN_ONLY'}
            seen.append(record)
            with output.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            print(f'[GRASP SEARCH] {k+1}/{self.args.candidates} score={score} passed={passed}',flush=True)
        best=sorted(seen,key=lambda r:(r['score'],r['contact_angular_coverage_deg']),reverse=True)[:min(10,len(seen))]
        result=output.with_suffix('.top10.json')
        result.write_text(json.dumps(best,ensure_ascii=False,indent=2))
        print('[RESULT]',result,flush=True)


@hydra.main(config_name='config',config_path='configs')
def main(cfg:DictConfig):
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidates',type=int,default=30)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--lateral-rad',type=float,default=.05)
    parser.add_argument('--flex-rad',type=float,default=.03)
    parser.add_argument('--release-fraction',type=float,default=.65)
    parser.add_argument('--settle-s',type=float,default=.5)
    parser.add_argument('--release-s',type=float,default=.3)
    parser.add_argument('--hold-s',type=float,default=.6)
    parser.add_argument('--max-horizontal-mm',type=float,default=8.)
    parser.add_argument('--max-drop-mm',type=float,default=8.)
    parser.add_argument('--max-tilt-deg',type=float,default=10.)
    parser.add_argument('--output',default='grasp_search/release_search.jsonl')
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
