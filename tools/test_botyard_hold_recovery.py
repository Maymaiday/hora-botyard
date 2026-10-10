"""Controller and real worker phase tests with synthetic states, not physics."""
import ast
import copy
import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from botyard_hold_recovery import HoldRecovery, candidate_order
from botyard_two_stage_core import expand, smooth

ROOT=Path(__file__).resolve().parents[1]

def row(target,contacts=()):
    return dict(q20=expand(target).tolist(),target20=expand(target).tolist(),
                contacts=dict(fingers=list(contacts),support=[],capContacts=0),bottomZ=.5,
                jointViolation=0.,object7=[0,0,.55,0,0,0,1],relativeObject7=[0,0,.1,0,0,0,1])

class Recovery(unittest.TestCase):
    def setUp(self):
        self.c=HoldRecovery(np.zeros(16),-np.ones(20),np.ones(20))
    def test_delay_and_only_missing_finger_flexion(self):
        r=row(self.c.command,['lf','mf','rf','th'])
        for _ in range(10):self.c.update(r,1/240)
        np.testing.assert_equal(self.c.command,0.)
        for _ in range(30):self.c.update(r,1/240)
        self.assertGreater(self.c.command[1],0)
        np.testing.assert_equal(self.c.command[[0,3,4,5,6,7,8,9,10,11,12,13,14,15]],0.)
    def test_global_budget_and_positive_error_cap(self):
        for _ in range(3000):self.c.update(row(self.c.command),1/240)
        self.assertLessEqual(float(self.c.command.max()),.06+1e-12)
        c=HoldRecovery(np.zeros(16),-np.ones(20),np.ones(20));r=row(np.zeros(16))
        for _ in range(3000):c.update(r,1/240)
        self.assertLessEqual(float(c.command.max()),.025+1e-12)
    def test_distal_limit_is_respected(self):
        hi=np.ones(20);hi[3]=.003
        c=HoldRecovery(np.zeros(16),-hi,hi)
        for _ in range(3000):c.update(row(c.command),1/240)
        self.assertLessEqual(expand(c.command)[3],.003)
    def test_contact_restoration_stops_increment_never_opens(self):
        for _ in range(100):self.c.update(row(self.c.command),1/240)
        saved=self.c.command.copy()
        for _ in range(100):self.c.update(row(self.c.command-.015,['ff','lf','mf','rf','th']),1/240)
        np.testing.assert_array_equal(self.c.command,saved)
    def test_contact_with_weak_gap_restores_small_bounded_error(self):
        r=row(np.zeros(16),['ff','lf','mf','rf','th'])
        for _ in range(47):self.c.update(r,1/240)
        np.testing.assert_array_equal(self.c.command,0.)
        for _ in range(1000):self.c.update(r,1/240)
        self.assertGreater(self.c.command[1],.0049)
        self.assertLessEqual(self.c.command.max(),.01)
        np.testing.assert_array_equal(self.c.command[[0,3,6,9,12,13]],0.)
    def test_support_cap_drop_and_limit_disable_recovery(self):
        for key,value in [('support',['palm']),('capContacts',1),('bottomZ',.02),('jointViolation',.02)]:
            c=HoldRecovery(np.zeros(16),-np.ones(20),np.ones(20));r=row(c.command)
            if key in r['contacts']:r['contacts'][key]=value
            else:r[key]=value
            for _ in range(100):c.update(r,1/240)
            np.testing.assert_array_equal(c.command,0.)
    def test_candidate_priority_without_repeats(self):
        order=candidate_order(16,[13,2,4,14]);self.assertEqual(order[:4],[13,2,4,14]);self.assertEqual(sorted(order),list(range(16)))
        with self.assertRaises(ValueError):candidate_order(16,[16])
        with self.assertRaises(ValueError):candidate_order(16,[2,2])
    def test_refinement_rejection_cannot_promote_canonical(self):
        from refine_botyard_cold_grasp import reject
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'rejected.json'
            self.assertEqual(reject(path,dict(accepted=True,canonical={'q20':[0]*20}),'Confirmation lost contact',[]),0)
            data=json.loads(path.read_text());self.assertFalse(data['accepted']);self.assertNotIn('canonical',data)
            self.assertEqual(data['refinementOutcome'],'search_rejected')
    def test_real_worker_phase_freezes_last_applied_command(self):
        tree=ast.parse((ROOT/'tools/botyard_two_stage_worker.py').read_text())
        phase=next(n for n in next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Scene').body if isinstance(n,ast.FunctionDef) and n.name=='phase')
        recovery=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='recover_and_freeze')
        ns=dict(np=np,expand=expand,smooth=smooth,Rotation=Rotation,json=__import__('json'))
        exec(compile(ast.Module(body=[phase,recovery],type_ignores=[]),'actual_worker_functions','exec'),ns)
        class Fake:
            phase=ns['phase']
            a=SimpleNamespace(dt=1/240)
            def __init__(self):self.commands=[];self.time=0.;self.target=np.zeros(20)
            def step(self,p,r,q,stage):self.target=q.copy();self.commands.append((stage,q.copy()));self.time+=self.a.dt
            def snapshot(self,stage):
                r=row(np.zeros(16),['lf','mf','rf','th']);r['target20']=self.target.tolist();r['simTime']=self.time
                r['objectVelocity6']=[0]*6;return r
        fake=Fake()
        recovering,frozen,faults=ns['recover_and_freeze'](fake,[0,0,.5],[0,0,0],self.c,'test',{})
        self.assertEqual(len(recovering),1200);self.assertEqual(len(frozen),720);self.assertEqual(faults,[])
        np.testing.assert_array_equal(recovering[-1]['target20'],frozen[0]['target20'])
        for r in frozen:np.testing.assert_array_equal(r['target20'],expand(self.c.command))
        self.assertGreater(self.c.command[1],0.)

if __name__=='__main__':unittest.main()
