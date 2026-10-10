"""Offline logic regressions, NOT Isaac Gym physics validation."""
import itertools,json,unittest
from pathlib import Path
import numpy as np
from botyard_grip_control import collision_masks,tracking_check,ContactCloser,closed_grip_check
from botyard_two_stage_core import ACTUAL,expand

class ContactControlTests(unittest.TestCase):
    def test_source_28_masks_exclude_exactly_requested_pairs(self):
        p=Path(__file__).resolve().parents[1]/'grasp_seed_banks/two_stage/collision_excludes.json'
        pairs=json.loads(p.read_text())['pairs'];names=sorted(set(sum(pairs,[])))+['thtip','world']
        m=collision_masks(names,pairs);self.assertEqual(len(pairs),28)
        excluded={frozenset((a,b)) for a,b in itertools.combinations(names,2) if m[a]&m[b]}
        self.assertEqual(excluded,{frozenset(p) for p in pairs})
        self.assertTrue(all(0<=v<2**31 for v in m.values()))
        self.assertEqual(m['thtip'],0)
        self.assertTrue(all(v&0==0 for v in m.values())) # object mask remains 0

    def test_unknown_pair_stops_before_setting_shapes(self):
        with self.assertRaises(ValueError):collision_masks(['palm'],[['palm','thbase']])

    def closer(self):
        return ContactCloser(np.full(16,.5),np.zeros(20),np.ones(20))

    def test_no_contact_closes_flexion_only_with_finite_budget(self):
        c=self.closer()
        for _ in range(200):c.update(expand(np.full(16,.5)),[],.05)
        np.testing.assert_allclose(c.command[[0,3,6,9,12,13]],.5)
        np.testing.assert_allclose(c.command[[1,2,4,5,7,8,10,11,14,15]],.85)

    def test_sustained_contact_uses_actual_angle_plus_preload(self):
        c=self.closer();q=expand(np.full(16,.4))
        c.update(q,['th'],.03);self.assertFalse(c.latched['th'])
        c.update(q,['th'],.03);self.assertTrue(c.latched['th'])
        np.testing.assert_allclose(c.command[14:16],.425)
        for _ in range(20):c.update(q,['th'],.05)
        np.testing.assert_allclose(c.command[14:16],.425)

    def test_contact_loss_resumes_bounded_closure(self):
        c=self.closer();q=expand(np.full(16,.4));c.update(q,['th'],.06)
        c.update(q,[],.1);self.assertFalse(c.latched['th'])
        np.testing.assert_allclose(c.command[14:16],.437)

    def test_uploaded_thumb_tracking_error_rejected(self):
        q=np.zeros(20);target=q.copy();q[16]=-1.047208;target[16]=-.923196
        r=dict(simTime=.4,q20=q.tolist(),target20=target.tolist(),velocity20=[0]*20)
        check=tracking_check([r]);self.assertFalse(check['passed'])
        self.assertEqual(check['joints'][0]['joint'],'THJ4')

    def row(self,fingers):
        return dict(simTime=1.,contacts=dict(fingers=fingers,support=['ground'],capContacts=0,
                    sideContactZByFinger={f:.005 for f in fingers}),objectVelocity6=[0]*6)

    def test_ground_allowed_only_during_closure_and_all_five_required(self):
        row=self.row(['ff','lf','mf','rf','th']);self.assertTrue(closed_grip_check([row],1.)['passed'])
        self.assertFalse(closed_grip_check([self.row(['rf'])],1.)['passed'])
        row['contacts']['support'].append('palm');self.assertFalse(closed_grip_check([row],1.)['passed'])

    def test_closure_acceptance_requires_persistent_thumb_contact(self):
        rows=[]
        for i in range(7):
            r=self.row(['ff','lf','mf','rf']+(['th'] if i==6 else []));r['simTime']=i*.05;rows.append(r)
        self.assertIn('thumb_side_contact_not_established',closed_grip_check(rows,1.)['reasons'])

    def settled_rows(self):
        rows=[]
        for i in range(73):
            r=self.row(['ff','lf','mf','rf','th']);r['simTime']=i/240
            r['object7']=[i/72*.0001,0,.04,0,0,0,1]
            r['objectVelocity6']=[.014,0,0,0,0,.22 if i!=40 else .3169];rows.append(r)
        return rows

    def test_one_small_speed_spike_passes_new_gate_but_preserves_strict_failure(self):
        rows=self.settled_rows()
        self.assertFalse(closed_grip_check(rows,1.)['passed'])
        self.assertTrue(closed_grip_check(rows,1.,settled=True)['passed'])

    def test_settle_rejects_real_slide_rotation_or_persistent_speed(self):
        for mode in ('slide','rotation','speed','peak','thumb'):
            rows=self.settled_rows()
            for i,r in enumerate(rows):
                if mode=='slide':r['object7'][0]=i/72*.002
                if mode=='rotation':r['object7'][3:]=[0,0,np.sin(i/72*.02),np.cos(i/72*.02)]
                if mode=='speed':r['objectVelocity6'][5]=.4
                if mode=='thumb':r['contacts']['fingers'].remove('th')
            if mode=='peak':rows[40]['objectVelocity6'][5]=.7
            self.assertFalse(closed_grip_check(rows,1.,settled=True)['passed'],mode)

if __name__=='__main__':unittest.main(verbosity=2)
