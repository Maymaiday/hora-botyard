"""Tests the real cache writer/reader and acceptance logic without Isaac Gym."""
import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('cache',Path(__file__).resolve().parents[1]/'hora/tasks/botyard_grasp_cache.py');cache=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache)

class CacheContract(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.asset=self.root/'cylinder.urdf';self.asset.write_text('test object');self.hand=self.root/'hand.urdf';self.hand.write_text('test hand')
        self.names=cache.POLICY_NAMES+['FFJ1','LFJ1','MFJ1','RFJ1'];self.pose=[0,0,.5,0,0,0,1]
        self.meta={'jointNames':self.names,'objectScale':.9,'handRootPoseLocal':self.pose,'objectAsset':str(self.asset),'handAsset':str(self.hand)}
        for key in ['objectAsset','handAsset']:self.meta[key+'Sha256']=hashlib.sha256(Path(self.meta[key]).read_bytes()).hexdigest()
        self.states=np.zeros((3,27),dtype=np.float32);self.states[:,:20]=np.arange(60).reshape(3,20)/100;self.states[:,20:23]=[[.01,-.25,.76],[.02,-.24,.77],[.03,-.23,.78]];self.states[:,26]=1
        self.targets=np.arange(48,dtype=np.float32).reshape(3,16)/100;self.path=self.root/'cache.npy';cache.write_cache(self.path,self.states,self.targets,self.meta)
    def read(self,**kw):
        args=dict(path=self.path,joint_names=self.names,scale=.9,hand_pose=self.pose,object_asset=str(self.asset),hand_asset=str(self.hand));args.update(kw);return cache.read_cache(**args)
    def test_round_trip_keeps_20_joints_and_object_7d(self):
        states,targets,meta=self.read();np.testing.assert_array_equal(states,self.states);np.testing.assert_array_equal(targets,self.targets);self.assertEqual(meta['coordinateFrame'],'environment-local')
    def test_row_sampling_preserves_own_target(self):
        states,targets,_=self.read();idx=[2,0,2];np.testing.assert_array_equal(targets[idx],self.targets[idx]);np.testing.assert_array_equal(states[idx,20:27],self.states[idx,20:27]);self.assertFalse(np.array_equal(targets[idx,0:16],states[idx,0:16]))
    def test_different_environment_origin_round_trip(self):
        states,_,_=self.read();origin=np.array([[3.,-2.,0.],[-1.,4.,0.],[0,0,0]]);world=states[:,20:23]+origin;np.testing.assert_allclose(world-origin,self.states[:,20:23],atol=1e-7)
    def test_wrong_scale_rejected(self):
        with self.assertRaises(ValueError):self.read(scale=.8)
    def test_wrong_joint_order_rejected(self):
        with self.assertRaises(ValueError):self.read(joint_names=self.names[::-1])
    def test_wrong_fixed_root_rejected(self):
        with self.assertRaises(ValueError):self.read(hand_pose=[0,0,.6,0,0,0,1])
    def test_changed_asset_rejected(self):
        self.asset.write_text('changed')
        with self.assertRaises(ValueError):self.read()
    def test_changed_npy_rejected(self):
        states=self.states.copy();states[0,0]+=.1;np.save(self.path,states)
        with self.assertRaises(ValueError):self.read()
    def test_missing_sidecar_rejected(self):
        self.path.with_suffix('.meta.npz').unlink()
        with self.assertRaises(FileNotFoundError):self.read()
    def test_target_row_count_rejected(self):
        with self.assertRaises(ValueError):cache.write_cache(self.path,self.states,self.targets[:2],self.meta)
    def test_invalid_quaternion_rejected(self):
        bad=self.states.copy();bad[:,23:27]=0
        with self.assertRaises(ValueError):cache.write_cache(self.path,bad,self.targets,self.meta)
    def test_terminal_failure_not_saved_as_timeout(self):
        actual=cache.terminal_success([False,True,True,True],[True,False,True,True],[180,180,179,180],180,12)
        np.testing.assert_array_equal(actual,[False,False,False,True])
    def test_side_contacts_exclude_end_caps_and_support(self):
        vec=np.dtype([('x','f4'),('y','f4'),('z','f4')]);dtype=np.dtype([('body0','i4'),('body1','i4'),('lambda','f4'),('localPos0',vec),('localPos1',vec)])
        rows=np.zeros(5,dtype=dtype)
        for i in range(5):rows[i]['body0']=30;rows[i]['body1']=i;rows[i]['lambda']=1;rows[i]['localPos0']=(.036,0,0)
        tips=set(range(5));touch,support=cache.side_contact_summary(rows,30,tips,.036,.072);self.assertEqual(touch,tips);self.assertFalse(support)
        rows[4]['localPos0']=(.036,0,.036);touch,_=cache.side_contact_summary(rows,30,tips,.036,.072);self.assertNotEqual(touch,tips)
        rows[4]['body1']=15;_,support=cache.side_contact_summary(rows,30,tips,.036,.072);self.assertTrue(support)

if __name__=='__main__':unittest.main(verbosity=2)
