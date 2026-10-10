"""Parallel indexing tests without native Gym; actual worker reset methods tested."""
import ast
import importlib.util
import sys
import types
import time
import tempfile
import hashlib
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
package=types.ModuleType('generation_test_package');package.__path__=[str(ROOT/'hora/tasks')]
sys.modules[package.__name__]=package
spec=importlib.util.spec_from_file_location(package.__name__+'.botyard_generation_batch',ROOT/'hora/tasks/botyard_generation_batch.py')
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
C=sys.modules[package.__name__+'.botyard_pose_contract']

class Tensor(np.ndarray):
    def cpu(self):return self
    def numpy(self):return np.asarray(self)
    def to(self,dtype=None):return self.astype(dtype).view(Tensor)
    def long(self):return self.astype(np.int64).view(Tensor)
def tensor(value,device=None):return np.array(value).view(Tensor)

class Parallel(unittest.TestCase):
    def test_continue_skips_calibration_and_keeps_generate_check_train_order(self):
        from run_botyard_full_pipeline import Pipeline
        events=[]
        with tempfile.TemporaryDirectory() as temp:
            seed=Path(temp)/'existing_seed.json';seed.write_text('{}')
            task=types.SimpleNamespace(a=types.SimpleNamespace(phase='continue',scales=[1.,.9,.8],cache_count=50000),
                seed_path=lambda scale:seed,generate=lambda scale,count:events.append(('generate',scale)) or True,
                check=lambda:events.append(('check',None)),train=lambda:events.append(('train',None)),
                summary={},save=lambda:None,out=Path(temp))
            Pipeline.execute(task)
        self.assertEqual(events,[('generate',1.),('generate',.9),('generate',.8),('check',None),('train',None)])
    def partial(self,temp):
        from generation_test_package.botyard_grasp_cache import write_cache
        assets=dict(objectAsset=str(ROOT/'assets/cylinder.urdf'),handAsset=str(ROOT/'assets/botyard_dexhand_gym/urdf/right_dexhand.urdf'))
        expected=dict(assets,jointNames=C.ACTUAL,objectScale=.9,handRootPoseLocal=[0,0,.5,0,0,0,1],
            validationVersion='five_side_pose_motion_v3',torqueControl=True,controlFrequencyInv=6,collisionConfigSha256='test',
            solverPositionIterations=8,solverVelocityIterations=4,objectMassKg=.05,friction=.5,pgain=3.,dgain=.1,
            dt=1/240,warmupSeconds=.4,maxTiltDeg=2.,armature=.001,torqueLimit=.5,maxDriftM=.008,maxDropM=.005)
        for key,path in assets.items():expected[key+'Sha256']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        state=np.r_[np.zeros(20),[0,0,.7,0,0,0,1]][None,:];target=np.full((1,16),.02)
        old=dict(expected,complete=False,completedEpisodes=123);old.pop('maxDriftM');old.pop('maxDropM');old['dt']=float(np.float32(1/240))
        path=Path(temp)/'partial.npy';write_cache(path,state,target,old)
        return path,expected,state,target
    def test_legacy_partial_rows_and_controller_targets_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            path,expected,state,target=self.partial(temp)
            a,b,meta=B.restore_partial(path,expected)
            np.testing.assert_allclose(a,state);np.testing.assert_allclose(b,target);self.assertEqual(meta['completedEpisodes'],123)
    def test_partial_with_different_physics_or_acceptance_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path,expected,state,target=self.partial(temp)
            for key,value in [('dt',1/120),('collisionConfigSha256','other'),('maxTiltDeg',3.),('maxDriftM',.01)]:
                changed=dict(expected);changed[key]=value
                with self.assertRaises(ValueError):B.restore_partial(path,changed)
    def test_sampling_reduction_is_bounded_and_disabled_when_not_requested(self):
        sampler=B.LocalSampler(True)
        for _ in range(3):sampler.update(32,512,100)
        self.assertEqual(sampler.factor,.5)
        for _ in range(100):sampler.update(32,512,100)
        self.assertEqual(sampler.factor,.25);self.assertEqual(len(sampler.history),2)
        fixed=B.LocalSampler(False);good=B.LocalSampler(True)
        for _ in range(100):fixed.update(1,512,100);good.update(300,512,100)
        self.assertEqual(fixed.factor,1.);self.assertEqual(good.factor,1.)
    def test_real_runner_requests_parallel_generation_but_single_anchor(self):
        from run_botyard_full_pipeline import Pipeline
        calls=[]
        with tempfile.TemporaryDirectory() as temp:
            task=types.SimpleNamespace(a=types.SimpleNamespace(fresh=True,generation_envs=512,worker_timeout=180,cache_timeout=60),
                out=Path(temp),anchor_attempts={},common=lambda count,scales:['task.env.numEnvs='+str(count)],
                job=lambda name,args,timeout:calls.append(args) or 1)
            self.assertFalse(Pipeline.generate(task,1.,50000))
            self.assertFalse(Pipeline.generate(task,1.,1,anchor=True))
        self.assertIn('task.env.numEnvs=512',calls[0]);self.assertIn('task.env.numEnvs=1',calls[1])
    def test_every_environment_has_its_own_full_trace(self):
        trace=B.BatchTrace(816,3);poses=np.tile([0,0,.7,0,0,0,1],(3,1));velocity=np.zeros((3,6))
        for i in range(816):
            contacts=np.full(3,31,np.uint8);support=np.zeros(3,bool)
            if i==200:contacts[1]=15
            if i==250:support[2]=True
            trace.append(poses,velocity,contacts,support,np.zeros(3))
        checks=[trace.check(i,poses[i],1.,1/240,.4,2.,.008,.005) for i in range(3)]
        self.assertTrue(checks[0]['passed'])
        self.assertIn('missing_finger_side_contact',checks[1]['reasons'])
        self.assertIn('palm_ground_or_cap_support',checks[2]['reasons'])
        self.assertEqual(checks[0]['physicsSteps'],816)
        trace.clear();self.assertEqual(trace.count,0)
    def test_observer_reads_each_env_and_removes_its_origin(self):
        point=np.dtype([('x',float),('y',float),('z',float)])
        records=np.zeros(5,dtype=[('body0',int),('body1',int),('lambda',float),('localPos0',point),('localPos1',point)])
        for i in range(5):records[i]['body0']=99;records[i]['body1']=i;records[i]['lambda']=1.;records[i]['localPos0']=(.04,0,.005)
        class Gym:
            def __init__(self):self.called=[]
            def get_env_rigid_contacts(self,env):self.called.append(env);return records if env==0 else records[:4]
        task=types.SimpleNamespace(num_envs=2,envs=[0,1],gym=Gym(),object_body_handle=99,
            finger_body_groups=dict(enumerate(C.FINGERS)),env_scales_np=np.ones(2),
            object_pose=tensor([[0,0,.7,0,0,0,1],[10,0,.7,0,0,0,1]]),state_offsets=tensor([[0,0,0],[10,0,0]]),
            object_linvel=tensor(np.zeros((2,3))),object_angvel=tensor(np.zeros((2,3))),
            allegro_hand_dof_pos=tensor(np.zeros((2,20))),allegro_hand_dof_lower_limits=tensor(-np.ones(20)),
            allegro_hand_dof_upper_limits=tensor(np.ones(20)))
        task.refreshes=0
        def refresh():task.refreshes+=1
        task._refresh_gym=refresh
        trace=B.BatchTrace(1,2);B.observe_batch(task,trace)
        self.assertEqual(task.gym.called,[0,1]);self.assertEqual(task.refreshes,1)
        np.testing.assert_array_equal(trace.poses[0,0],trace.poses[0,1]);self.assertEqual(trace.contact[0].tolist(),[31,15])
    def test_independent_noise_preserves_preload_and_actual_distal_offsets(self):
        seed=dict(actualQ20=np.full(20,.2).tolist(),holdingTarget16=np.full(16,.25).tolist(),object7=[0,0,.7,0,0,0,1])
        seed['actualQ20'][3]=.18
        q,target,poses=B.sample_batch([seed],512,np.random.RandomState(2026),C.ACTUAL,-np.ones(20),np.ones(20),
            -np.ones(16),np.ones(16),.02,.0005,.0005)
        self.assertEqual(q.shape,(512,20));self.assertEqual(target.shape,(512,16));self.assertEqual(poses.shape,(512,7))
        self.assertGreater(np.unique(q[:,1]).size,500)
        np.testing.assert_allclose(target[:,1]-q[:,1],.05)
        np.testing.assert_allclose(q[:,2]-q[:,3],.02)
        self.assertFalse(np.array_equal(poses[0],poses[1]))
    def methods(self):
        tree=ast.parse((ROOT/'hora/tasks/botyard_hand_grasp.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['reset_idx','compute_reward']]
        from collections import Counter
        ns=dict(np=np,time=time,sample_batch=B.sample_batch,to_torch=tensor,Counter=Counter,
            torch=types.SimpleNamespace(int32=np.int32,any=np.any,all=np.all),gymtorch=types.SimpleNamespace(unwrap_tensor=lambda x:x))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_grasp_methods','exec'),ns)
        return ns
    def test_real_reset_saves_only_passing_env_with_matching_targets_and_local_object(self):
        ns=self.methods()
        obj=tensor([[0,0,.7,0,0,0,1]+[0]*6,[10,.1,.8,0,0,0,1]+[0]*6])
        task=types.SimpleNamespace(num_envs=2,trace=types.SimpleNamespace(count=816),completed=0,batches=0,states=[],targets=[],fingerprints=set(),
            root_state_tensor=obj,object_indices=tensor([0,1]),state_offsets=tensor([[0,0,0],[10,0,0]]),
            allegro_hand_dof_pos=tensor(np.stack([np.full(20,.2),np.full(20,.3)])),cur_policy_targets=tensor(np.stack([np.full(16,.25),np.full(16,.35)])),
            terminal_pass=np.array([False,True]),goal=1,_report=lambda **kw:None,_save=lambda partial:None,
            batch_reasons={'missing_finger_side_contact':1})
        import time
        task.started=time.monotonic()
        with self.assertRaises(SystemExit):ns['reset_idx'](task,tensor([0,1]))
        self.assertEqual(len(task.states),1);np.testing.assert_allclose(task.states[0][:20],.3)
        np.testing.assert_allclose(task.states[0][20:23],[0,.1,.8]);np.testing.assert_allclose(task.targets[0],.35)
        self.assertEqual(task.completed,2)
    def test_actual_reset_initializes_distinct_poses_and_targets_for_all_envs(self):
        ns=self.methods();seed=dict(actualQ20=np.full(20,.2).tolist(),holdingTarget16=np.full(16,.25).tolist(),object7=[0,0,.7,0,0,0,1])
        class Gym:
            def __init__(self):self.writes=[]
            def set_actor_root_state_tensor_indexed(self,*args):self.writes.append(('root',args[-1]))
            def set_dof_state_tensor_indexed(self,*args):self.writes.append(('dof',args[-1]))
        task=types.SimpleNamespace(num_envs=2,trace=B.BatchTrace(816,2),seeds=[seed],device='cpu',sim=None,gym=Gym(),
            terminal_pass=np.zeros(2,bool),sim_dof_names=C.ACTUAL,config={'env':dict(graspPoseNoise=.02,graspObjectXYNoise=.0005,graspObjectZNoise=.0005)},
            allegro_hand_dof_lower_limits=tensor(-np.ones(20)),allegro_hand_dof_upper_limits=tensor(np.ones(20)),
            policy_dof_lower_limits=tensor(-np.ones(16)),policy_dof_upper_limits=tensor(np.ones(16)),
            root_state_tensor=tensor(np.zeros((4,13))),object_indices=tensor([1,3]),hand_indices=tensor([0,2]),
            state_offsets=tensor([[0,0,0],[10,0,0]]),dof_state=tensor(np.zeros((40,2))),_report=lambda **kw:None)
        for name,shape in [('allegro_hand_dof_pos',(2,20)),('allegro_hand_dof_vel',(2,20)),('holding_reference',(2,16)),
                           ('prev_policy_targets',(2,16)),('cur_policy_targets',(2,16)),('prev_targets',(2,20)),('cur_targets',(2,20)),
                           ('init_pose_buf',(2,20)),('progress_buf',(2,)),('reset_buf',(2,)),('obs_buf',(2,3)),('rb_forces',(2,3)),
                           ('proprio_hist_buf',(2,3)),('at_reset_buf',(2,))]:setattr(task,name,tensor(np.zeros(shape)))
        task.expand_policy_targets=lambda target:tensor(np.stack([C.expand_delta(d,C.ACTUAL) for d in target]))
        ns['reset_idx'](task,tensor([0,1]))
        self.assertFalse(np.array_equal(task.cur_policy_targets[0],task.cur_policy_targets[1]))
        np.testing.assert_allclose(task.root_state_tensor[[1,3],:3],task.initial_pose[:,:3]+task.state_offsets)
        np.testing.assert_allclose(task.cur_policy_targets[:,1]-task.allegro_hand_dof_pos[:,1],.05)
        self.assertEqual(task.gym.writes,[('root',2),('dof',2)])

if __name__=='__main__':unittest.main()
