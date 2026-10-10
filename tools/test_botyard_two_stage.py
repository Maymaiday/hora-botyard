#!/usr/bin/env python3
"""Offline regression checks, NOT a substitute for Isaac Gym evaluation."""
import argparse,ast,copy,json,sys,tempfile,unittest
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from botyard_two_stage_core import *
ROOT=None
class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=ROOT/'assets/botyard_dexhand_gym/urdf/right_dexhand.urdf';cls.geo=Geometry(cls.source)
        cls.seed=json.loads((Path(__file__).resolve().parents[1]/'grasp_seed_banks/two_stage/s10.json').read_text())['candidates'][0]
    def test_original_twenty_and_injected_twenty_six(self):
        with tempfile.TemporaryDirectory() as d:
            out=make_wrist_urdf(self.source,Path(d)/'mobile.urdf');r=ET.parse(out).getroot()
            joints=r.findall('joint');names={j.get('name') for j in joints if j.get('type')!='fixed'}
            self.assertEqual(names,set(ACTUAL+WRIST));self.assertEqual(r.find("joint[@name='FOREARM']/parent").get('link'),'search_link_5')
            for mesh in r.findall('.//mesh'):self.assertTrue(Path(mesh.get('filename')).is_file())
            parents={j.find('child').get('link'):j.find('parent').get('link') for j in joints}
            for child in parents:
                seen=set();cursor=child
                while cursor in parents:
                    self.assertNotIn(cursor,seen);seen.add(cursor);cursor=parents[cursor]
                self.assertEqual(cursor,'world')
    def test_actual_fk_does_not_force_mimic(self):
        q=expand(self.seed['canonicalPolicyPose']);base=np.eye(4)
        before=self.geo.fk(q,base);q[ACTUAL.index('FFJ1')]+=.2;after=self.geo.fk(q,base)
        self.assertGreater(np.linalg.norm(before['ffdistal']-after['ffdistal']),.01)
        np.testing.assert_allclose(before['ffmiddle'],after['ffmiddle'])
    def test_all_scales_pregrasp_and_full_flip_clearance(self):
        bank=Path(__file__).resolve().parents[1]/'grasp_seed_banks/two_stage'
        for scale,tag in [(1.,'s10'),(.9,'s09'),(.8,'s08')]:
            seed=json.loads((bank/(tag+'.json')).read_text())['candidates'][0]
            c=sample_candidate(seed,0,np.random.default_rng(2026),self.geo,scale)
            self.assertGreater(c['meshFloorClearance'],.003);self.assertGreater(c['palmGap'],.008);self.assertGreater(c['pregraspVertexClearance'],.005)
            self.assertGreater(c['liftDistance'],.16)
            start=transform(c['rootStart7'][:3],quat=c['rootStart7'][3:]);local=np.linalg.inv(start)@np.r_[c['objectStart7'][:3],1.]
            pivot=np.array(c['objectStart7'][:3])+[0,0,c['liftDistance']]
            for u in np.linspace(0,1,51):
                r=np.array(c['goalRPY']);r[0]+=np.pi*(1-u);rot=Rotation.from_euler('xyz',r).as_matrix();base=transform(pivot-rot@local[:3],rpy=r)
                np.testing.assert_allclose((base@local)[:3],pivot,atol=1e-12)
                minimum=min(v[:,2].min() for v in self.geo.points(expand(c['close16']),base).values())
                self.assertGreater(minimum,.048)
    def test_transform_round_trip_and_twenty_seven(self):
        h=transform([.1,-.2,.5],rpy=[1.05,.2,-.1]);o=transform([.03,-.3,.8],rpy=[.2,.4,.3]);local=np.linalg.inv(h)@o
        np.testing.assert_allclose(h@local,o,atol=1e-12)
        state=expand(self.seed['canonicalPolicyPose']).tolist()+pose7(o)
        self.assertEqual(len(state),27);self.assertEqual(len(pose7(local)),7)
    def test_side_contacts_allow_middle_not_palm_caps_floor(self):
        names={1:'ffmiddle',2:'lfdistal',3:'mfdistal',4:'rfdistal',5:'thdistal',6:'palm',-1:'ground'}
        def c(body,z=0):return dict(body0=99,body1=body,localPos0={'x':.04,'y':0,'z':z},**{'lambda':.1})
        good=contact_summary([c(i) for i in range(1,6)],99,names,.04,.08)
        self.assertEqual(len(good['fingers']),5);self.assertFalse(good['support'])
        self.assertEqual(contact_summary([c(6),c(-1)],99,names,.04,.08)['support'],['ground','palm'])
        cap=contact_summary([c(5,.039)],99,names,.04,.08);self.assertEqual(cap['capContacts'],1);self.assertNotIn('th',cap['fingers'])
    def rows(self):
        return [dict(q20=[0]*20,object7=[0,0,.2,0,0,0,1],relativeObject7=[0,0,.2,0,0,0,1],objectVelocity6=[0]*6,bottomZ=.16,contacts={'fingers':FINGERS.copy(),'support':[],'capContacts':0},jointViolation=0,wristErrorPosition=0,wristErrorAngleDeg=0) for _ in range(10)]
    def test_rejection_for_slip_support_thumb_and_motion(self):
        rows=self.rows();self.assertEqual(hold_check(rows,.08),[])
        for field,value,reason in [('bottomZ',0,'insufficient_lift'),('relativeObject7',[.02,0,.2,0,0,0,1],'relative_slip'),('objectVelocity6',[1,0,0,0,0,0],'not_settled_linear'),('wristErrorAngleDeg',10,'wrist_tracking')]:
            rows=self.rows();rows[-1][field]=value;self.assertIn(reason,hold_check(rows,.08))
        rows=self.rows();rows[-1]['contacts']['support']=['ground'];self.assertIn('external_or_palm_support',hold_check(rows,.08))
        rows=self.rows();rows[-1]['contacts']['fingers'].remove('th');self.assertIn('no_thumb_side_contact',hold_check(rows,.08))
    def test_nonfinite_never_accepted(self):
        rows=self.rows();rows[0]['q20'][0]=float('nan');self.assertIn('nonfinite',hold_check(rows,.08))
    def test_motion_never_sets_object_pose(self):
        source=Path(__file__).with_name('botyard_two_stage_worker.py').read_text();tree=ast.parse(source,feature_version=(3,8))
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('set_actor_rigid_body_states',calls);self.assertNotIn('set_actor_root_state_tensor',calls)
        self.assertEqual(calls.count('set_actor_dof_states'),1) # initialization only
    def test_supervisor_timeout_and_live_output(self):
        from search_botyard_two_stage import run
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);status,data=run([sys.executable,'-u','-c','import time;print("TEST heartbeat");time.sleep(30)'],path/'child.log',path/'none.json',.2)
            self.assertEqual(status['stopReason'],'wall_timeout');self.assertIsNotNone(status['exitCode']);self.assertFalse(data)
    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.json';atomic_json(p,{'accepted':False});self.assertFalse(json.loads(p.read_text())['accepted']);self.assertFalse(p.with_suffix('.json.tmp').exists())
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);a=ap.parse_args();ROOT=a.repo.resolve()
    unittest.main(argv=[sys.argv[0]],verbosity=2)
