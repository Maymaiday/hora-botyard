"""Pure preload math and evidence checks; no physics simulation."""
import copy,unittest
import numpy as np
from botyard_grip_control import forefinger_target
from botyard_two_stage_core import expand
from repair_botyard_ff_preload import ff_metrics,replay_source

class Preload(unittest.TestCase):
    def test_target_changes_only_ff_flexion_and_preserves_actual_state(self):
        q=np.zeros(20);q[1]=1.01116502285;q[2]=.19894637167
        target=np.zeros(16);target[1]=1.01211693287;target[2]=.19946787655
        before=q.copy();result=forefinger_target(q,target,-np.ones(20)*2,np.ones(20)*2,.02)
        self.assertAlmostEqual(result[1],q[1]+.02);self.assertAlmostEqual(result[2],q[2]+.02)
        unchanged=[i for i in range(16) if i not in (1,2)]
        np.testing.assert_array_equal(result[unchanged],target[unchanged]);np.testing.assert_array_equal(q,before)
        self.assertEqual(expand(result)[2],expand(result)[3])
    def test_no_opening_or_silent_limit_clipping(self):
        q=np.zeros(20);target=np.zeros(16);target[1]=.3
        self.assertEqual(forefinger_target(q,target,-np.ones(20),np.ones(20),.01)[1],.3)
        q[2]=.99
        with self.assertRaises(ValueError):forefinger_target(q,target,-np.ones(20),np.ones(20),.02)
    def test_touching_without_preload_does_not_pass(self):
        rows=[dict(simTime=i/240,contacts=dict(fingers=['ff']),target20=[0.]*20,q20=[0.]*20) for i in range(240)]
        data=dict(coldStart=dict(frames=rows));self.assertFalse(ff_metrics(data)['passed'])
        for r in rows:r['target20'][2]=.02
        self.assertTrue(ff_metrics(data)['passed'])
        rows[-1]['contacts']['fingers']=[];self.assertFalse(ff_metrics(data)['passed'])
    def test_proposal_is_not_accepted_and_can_target_is_preserved(self):
        can=dict(q20=[0.]*20,object7HandRelative=[0,0,.2,0,0,0,1],handRoot7=[0,0,.5,0,0,0,1],holdingTarget16=[.01]*16)
        data=dict(accepted=True,completed=True,canonical=can,candidate=dict(close16=[0.]*16))
        p=replay_source(data);self.assertFalse(p['accepted']);self.assertNotIn('canonical',p)
        self.assertEqual(p['candidate']['close16'],can['holdingTarget16']);self.assertEqual(p['final']['q20'],can['q20'])

if __name__=='__main__':unittest.main(verbosity=2)
