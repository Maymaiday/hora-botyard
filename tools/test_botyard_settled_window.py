"""Search-window admission tests; no native physics."""
import copy,unittest
from scipy.spatial.transform import Rotation
from botyard_settled_window import settled_candidate

class Window(unittest.TestCase):
    def source(self):
        quat=Rotation.from_euler('x',3,degrees=True).as_quat().tolist();rows=[]
        for i in range(1920):
            target=[0.]*20;target[1]=.02
            rows.append(dict(simTime=(i+1)/240,object7=[0,0,.7]+quat,objectVelocity6=[0.]*6,jointViolation=0.,
                q20=[0.]*20,target20=target,contacts=dict(fingers=['ff','lf','mf','rf','th'] if i>800 else ['ff','lf','rf','th'],support=[],capContacts=0)))
        return dict(scale=.9,completed=True,accepted=False,poseColdStartCheck=dict(passed=False,reasons=['missing_finger_side_contact','cylinder_tilt']),
            coldStart=dict(frames=rows,warmup=[],handRoot7=[0,0,.5,0,0,0,1]))
    def test_early_missing_contact_does_not_admit_cache(self):
        source=self.source();out=settled_candidate(source)
        self.assertTrue(out['settledSearchWindow']['searchCheck']['passed']);self.assertFalse(out['accepted'])
        self.assertFalse(out['settledSearchWindow']['acceptedForCache']);self.assertEqual(len(out['coldStart']['frames']),720)
        self.assertFalse(source['accepted']);self.assertEqual(len(source['coldStart']['frames']),1920)
    def test_late_contact_loss_or_wrong_sampling_is_rejected(self):
        source=self.source();source['coldStart']['frames'][-1]['contacts']['fingers'].remove('mf')
        with self.assertRaises(ValueError):settled_candidate(source)
        source=self.source();source['coldStart']['frames'][-1]['simTime']+=.01
        with self.assertRaises(ValueError):settled_candidate(source)

if __name__=='__main__':unittest.main(verbosity=2)
