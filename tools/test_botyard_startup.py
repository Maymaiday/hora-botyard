#!/usr/bin/env python3
"""Execute control/snapshot paths against a strict SDK stub. NOT physics simulation."""
import ast,importlib.util,sys,time,types,unittest
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
api=types.SimpleNamespace(STATE_ALL=3)
fake_package=types.ModuleType('isaacgym');fake_package.gymapi=api
old=sys.modules.get('isaacgym');sys.modules['isaacgym']=fake_package
spec=importlib.util.spec_from_file_location('worker_under_test',HERE/'botyard_two_stage_worker.py');worker=importlib.util.module_from_spec(spec)
try:spec.loader.exec_module(worker)
finally:
 if old is None:sys.modules.pop('isaacgym',None)
 else:sys.modules['isaacgym']=old
vec=np.dtype([('x','f4'),('y','f4'),('z','f4')]);rot=np.dtype([('x','f4'),('y','f4'),('z','f4'),('w','f4')]);pose=np.dtype([('p',vec),('r',rot)])
rb=np.dtype([('pose',pose),('vel',[('linear',vec),('angular',vec)])])
class StrictGym:
    # Deliberately has no set_actor_dof_actuation_forces method.
    def __init__(self,n):self.dof=np.zeros(n,dtype=[('pos','f4'),('vel','f4')]);self.events=[];self.torques=None
    def get_actor_dof_states(self,env,actor,flags):return self.dof.copy()
    def set_actor_dof_position_targets(self,env,actor,targets):
        assert targets.dtype==np.float32 and targets.shape==self.dof.shape;self.events.append('targets');return True
    def apply_actor_dof_efforts(self,env,actor,efforts):
        assert efforts.dtype==np.float32 and efforts.flags.c_contiguous and efforts.shape==self.dof.shape
        self.torques=efforts.copy();self.events.append('efforts');return True
    def simulate(self,sim):self.events.append('simulate')
    def fetch_results(self,sim,wait):assert wait is True;self.events.append('fetch')
    def get_actor_rigid_body_names(self,env,actor):return ['fabase']
    def get_actor_rigid_body_states(self,env,actor,flags):
        assert self.events[-1]=='fetch'
        states=np.zeros(1,dtype=rb);states['pose']['r']['w']=1
        states['pose']['p']['z']=.5 if actor==0 else .7
        return states
    def get_env_rigid_contacts(self,env):return []
class Regression(unittest.TestCase):
    def scene(self,fixed=False):
        s=worker.Scene.__new__(worker.Scene);s.fixed=fixed;s.object_only=False;s.anchor=np.eye(4);s.previous_wrist_rpy=np.zeros(3);s.names=worker.ACTUAL.copy() if fixed else worker.WRIST+worker.ACTUAL
        s.fi=np.arange(20) if fixed else np.arange(6,26);s.wi=np.array([],dtype=int) if fixed else np.arange(6)
        s.gym=StrictGym(len(s.names));s.env=object();s.sim=object();s.hand=0;s.obj=1;s.obody=9;s.body_names={9:'object',-1:'ground'}
        s.a=types.SimpleNamespace(kp=3.,kd=.1,torque_limit=.5,dt=1/240,scale=1.)
        s.target=np.zeros(len(s.names),np.float32);s.viewer=None;s.time=0.;s.last_print=time.monotonic();s.state_offset=np.zeros(3)
        s.lower=np.full(20,-2.);s.upper=np.full(20,2.);return s
    def test_mobile_controller_calls_real_actor_effort_api(self):
        s=self.scene();s.step([0,0,.5],[0,0,0],np.ones(20),'pregrasp')
        self.assertEqual(s.gym.events,['targets','efforts','simulate','fetch'])
        np.testing.assert_array_equal(s.gym.torques[:6],np.zeros(6));np.testing.assert_allclose(s.gym.torques[6:],.5)
    def test_fixed_controller_twenty_efforts(self):
        s=self.scene(True);s.step([0,0,.5],[0,0,0],np.ones(20),'cold_start')
        self.assertEqual(s.gym.events,['efforts','simulate','fetch']);self.assertEqual(s.gym.torques.shape,(20,))
    def test_snapshot_structured_fields_and_relative_pose(self):
        s=self.scene();s.step([0,0,.5],[0,0,0],np.zeros(20),'pregrasp');r=s.snapshot('pregrasp')
        self.assertEqual(len(r['q20']),20);self.assertEqual(len(r['object7']),7);self.assertEqual(len(r['objectVelocity6']),6)
        self.assertAlmostEqual(r['relativeObject7'][2],.2,places=6);self.assertEqual(r['contacts']['fingers'],[])
    def test_missing_methods_reported_together(self):
        with self.assertRaises(RuntimeError) as raised:worker.check_gym_api(types.SimpleNamespace())
        self.assertIn('apply_actor_dof_efforts',str(raised.exception));self.assertIn('get_env_rigid_contacts',str(raised.exception))
    def test_required_list_covers_every_sdk_call(self):
        tree=ast.parse((HERE/'botyard_two_stage_worker.py').read_text(),feature_version=(3,8));calls=set()
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
                base=n.func.value
                if (isinstance(base,ast.Name) and base.id=='g') or (isinstance(base,ast.Attribute) and base.attr=='gym'):calls.add(n.func.attr)
        self.assertEqual(calls,set(worker.REQUIRED_GYM_APIS))
        self.assertNotIn('set_actor_dof_actuation_forces',calls)
    def test_single_origin_guard_preserved(self):
        np.testing.assert_array_equal(worker.single_env_state_offset([0,0,0]),[0,0,0])
        with self.assertRaises(RuntimeError):worker.single_env_state_offset([1,0,0])
    def test_wrist_anchor_round_trip_and_continuous_flip(self):
        anchor=worker.transform([-.008,-.25,.41],rpy=[4.18,0,.09]);prev=np.zeros(3)
        first,_=worker.wrist_coordinates(anchor,anchor,prev)
        np.testing.assert_allclose(first,0,atol=1e-12)
        for u in np.linspace(0,1,51):
            desired=worker.transform([-.008,.02,.5],rpy=[4.18-np.pi*u,0,.09])
            coords,prev=worker.wrist_coordinates(anchor,desired,prev)
            recovered=anchor@worker.transform(coords[:3],rpy=[coords[5],coords[4],coords[3]])
            np.testing.assert_allclose(recovered,desired,atol=1e-12)
        self.assertAlmostEqual(prev[0],-np.pi,places=8)
    def test_uploaded_startup_impulse_rejected(self):
        from botyard_two_stage_core import upright_start_check
        row=dict(object7=[-.0022638,.0032241,.0439017,-.0386384,-.0269373,-.0001372,.9988902],objectVelocity6=[-.5806175,.8391950,1.0209997,-20.5636139,-13.9335289,.7421769])
        reasons=upright_start_check([row],[0,0,.041],.04,.08)
        self.assertIn('startup_impulse',reasons);self.assertIn('cylinder_tipped_before_approach',reasons)
        good=dict(object7=[0,0,.0401,0,0,0,1],objectVelocity6=[0]*6)
        self.assertEqual(upright_start_check([good],[0,0,.041],.04,.08),[])
    def test_object_only_step_applies_no_hand_efforts(self):
        s=self.scene();s.object_only=True;s.fi=np.array([],dtype=int);s.lower=np.zeros(0);s.upper=np.zeros(0);s.anchor=np.eye(4)
        s.step([0,0,0],[0,0,0],np.zeros(0),'object_only_settle')
        self.assertEqual(s.gym.events,['simulate','fetch']);self.assertIsNone(s.gym.torques)
        row=s.snapshot('object_only_settle');self.assertEqual(row['q20'],[])
    def test_mobile_actor_created_at_supplied_anchor(self):
        source=(HERE/'botyard_two_stage_worker.py').read_text()
        self.assertIn("sdk_pose(root7),'hand'",source)
        self.assertIn('Scene(a,mobile,False,pregrasp7',source)
        self.assertLess(source.index('startup_faults=upright_start_check(traces'),source.index("result['stage']='approach'"))
if __name__=='__main__':unittest.main(verbosity=2)
