"""Offline contract tests; not a substitute for native Gym replay."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT=Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'hora/tasks'/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
C=load('botyard_pose_contract');CACHE=load('botyard_grasp_cache')

class Contracts(unittest.TestCase):
    def setUp(self):
        self.seed=json.loads((ROOT/'grasp_seed_banks/current_verified/s10.json').read_text())['candidates'][0]
        self.rows=[dict(object7=list(self.seed['object7']),jointViolation=0.,sdkVelocity6=[0,0,0,.52,0,0],
                       contacts=dict(fingers=list(C.FINGERS),support=[],capContacts=0)) for _ in range(816)]
    def check(self,rows):return C.evaluate(rows,self.seed['object7'],1.,1/240)
    def test_pose_criterion_keeps_sdk_diagnostic(self):
        r=self.check(self.rows);self.assertTrue(r['passed']);self.assertGreater(r['metrics']['maxSDKAngularSpeed'],.3)
    def test_actual_quaternion_rotation_is_rejected(self):
        rows=copy.deepcopy(self.rows)
        rot=Rotation.from_quat(self.seed['object7'][3:])
        for i,row in enumerate(rows):row['object7'][3:]=(Rotation.from_rotvec([0,0,i/240])*rot).as_quat().tolist()
        self.assertIn('pose_motion_not_settled',self.check(rows)['reasons'])
    def test_quaternion_sign_and_axis_sign_invariant(self):
        rows=copy.deepcopy(self.rows)
        for i,row in enumerate(rows):
            if i%2:row['object7'][3:]=(-np.array(row['object7'][3:])).tolist()
        self.assertTrue(self.check(rows)['passed'])
    def test_contact_drop_is_rejected(self):
        rows=copy.deepcopy(self.rows);rows[200]['contacts']['fingers'].remove('th')
        self.assertIn('missing_finger_side_contact',self.check(rows)['reasons'])
    def test_palm_and_cap_support_rejected(self):
        rows=copy.deepcopy(self.rows);rows[-1]['contacts']['support']=['palm'];rows[-1]['contacts']['capContacts']=1
        self.assertIn('palm_ground_or_cap_support',self.check(rows)['reasons'])
    def test_real_drop_rejected(self):
        rows=copy.deepcopy(self.rows);rows[-1]['object7'][2]-=.02
        self.assertIn('object_drop_or_ground',self.check(rows)['reasons'])
    def test_side_contact_groups_middle_and_distal(self):
        point=np.dtype([('x',float),('y',float),('z',float)])
        dtype=[('body0',int),('body1',int),('lambda',float),('localPos0',point),('localPos1',point)]
        records=np.zeros(5,dtype=dtype)
        for i in range(5):
            records[i]['body0']=100;records[i]['body1']=i;records[i]['lambda']=1
            records[i]['localPos0']=(.04,0,.005)
        result=C.side_contacts(records,100,dict(enumerate(C.FINGERS)),1.)
        self.assertEqual(result['fingers'],sorted(C.FINGERS));self.assertEqual(result['support'],[])
    def test_root_relative_object_and_assets(self):
        np.testing.assert_allclose(C.world_object(self.seed['handRoot7'],self.seed['relativeObject7']),self.seed['object7'],atol=1e-8)
        for key,path in [('handAsset','assets/botyard_dexhand_gym/urdf/right_dexhand.urdf'),('objectAsset','assets/cylinder.urdf')]:
            self.assertEqual(self.seed[key+'Sha256'],hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
        cfg=yaml.safe_load((ROOT/'configs/task/BotyardHandHora.yaml').read_text())
        np.testing.assert_allclose(cfg['env']['botyard']['handQuaternionXYZW'],self.seed['handRoot7'][3:])
        self.assertEqual(cfg['sim']['physx']['num_velocity_iterations'],4)
        self.assertAlmostEqual(cfg['sim']['dt'],1/240)
    def test_distal_actual_angles_preserved_and_perturbed(self):
        q=np.array(self.seed['actualQ20']);target=C.expand_delta(self.seed['holdingTarget16'],C.ACTUAL)
        self.assertGreater(np.max(np.abs(q-target)),.02)
        self.assertNotEqual(q[2],q[3])
        delta=C.expand_delta(np.full(16,.002),C.ACTUAL)
        np.testing.assert_allclose((q+delta)-(target+delta),q-target,atol=1e-12)
    def test_named_collision_exclusions(self):
        pairs=json.loads((ROOT/'grasp_seed_banks/current_verified/collision_excludes.json').read_text())['pairs']
        names=sorted(set(n for p in pairs for n in p));m=C.collision_masks(names,pairs)
        self.assertTrue(m['palm']&m['thproximal'])
        self.assertFalse(m['ffdistal']&m['thdistal'])
        with self.assertRaises(ValueError):C.collision_masks(names,pairs+[pairs[0]])
    def test_cache_state_target_roundtrip_and_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'cache.npy';q=np.array(self.seed['actualQ20']);obj=np.array(self.seed['object7'])
            state=np.r_[q,obj][None,:];target=np.array(self.seed['holdingTarget16'])[None,:]
            hand='assets/botyard_dexhand_gym/urdf/right_dexhand.urdf';object_path='assets/cylinder.urdf'
            metadata=dict(jointNames=C.ACTUAL,objectScale=1.,handRootPoseLocal=self.seed['handRoot7'],
                          objectAsset=object_path,handAsset=hand,objectAssetSha256=self.seed['objectAssetSha256'],handAssetSha256=self.seed['handAssetSha256'])
            CACHE.write_cache(p,state,target,metadata)
            states,targets,_=CACHE.read_cache(p,C.ACTUAL,1.,self.seed['handRoot7'],object_path,hand)
            np.testing.assert_allclose(states,state,atol=1e-7);np.testing.assert_allclose(targets,target,atol=1e-7)
            bad=states.copy();bad[0,0]+=.001;np.save(p,bad)
            with self.assertRaises(ValueError):CACHE.read_cache(p,C.ACTUAL,1.,self.seed['handRoot7'],object_path,hand)
    def test_python38_and_feedback_order(self):
        for path in ROOT.rglob('*.py'):ast.parse(path.read_text(),filename=str(path),feature_version=(3,8))
        base=(ROOT/'hora/tasks/base/vec_task.py').read_text()
        step=base[base.index('    def step('):base.index('    def zero_actions(')]
        self.assertLess(step.index('self.gym.simulate'),step.index("hook=getattr"))
        self.assertLess(step.index('self.gym.fetch_results'),step.index("hook=getattr"))
    def test_actual_initializer_preserves_measured_quaternions(self):
        from types import SimpleNamespace
        class Vec3:
            def __init__(self,x,y,z):self.x,self.y,self.z=x,y,z
        class Quat:
            def __init__(self,x,y,z,w):self.values=[x,y,z,w]
            @staticmethod
            def from_axis_angle(axis,angle):
                q=Rotation.from_rotvec(np.array([axis.x,axis.y,axis.z])*angle).as_quat()
                return Quat(*q)
        class Transform:pass
        source=ast.parse((ROOT/'hora/tasks/botyard_hand_hora.py').read_text())
        cls=next(n for n in source.body if isinstance(n,ast.ClassDef))
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_seed_actor_poses')
        namespace={'gymapi':SimpleNamespace(Vec3=Vec3,Quat=Quat,Transform=Transform)}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<actual initializer>','exec'),namespace)
        cfg=yaml.safe_load((ROOT/'configs/task/BotyardHandHora.yaml').read_text())
        # A stale horizontal RPY must not override the measured quaternion.
        cfg['env']['botyard']['handRotationRPY']=[np.pi/2,0,0]
        hand,obj=namespace['_seed_actor_poses'](SimpleNamespace(config=cfg),self.seed)
        np.testing.assert_allclose(hand.r.values,self.seed['handRoot7'][3:])
        np.testing.assert_allclose(obj.r.values,self.seed['object7'][3:])

if __name__=='__main__':
    import os
    os.chdir(str(ROOT));unittest.main(verbosity=2)
