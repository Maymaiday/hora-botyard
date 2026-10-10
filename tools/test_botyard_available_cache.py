"""Export/provenance tests only; no native simulation or training."""
import hashlib,importlib.util,json,os,tempfile,types,unittest
from pathlib import Path
import numpy as np
from botyard_available_cache import export_available
from run_botyard_full_pipeline import Pipeline

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('available_cache_api',ROOT/'hora/tasks/botyard_grasp_cache.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
NAMES=['FFJ4','FFJ3','FFJ2','FFJ1','LFJ4','LFJ3','LFJ2','LFJ1','MFJ4','MFJ3','MFJ2','MFJ1','RFJ4','RFJ3','RFJ2','RFJ1','THJ4','THJ3','THJ2','THJ1']

class Available(unittest.TestCase):
    def setUp(self):
        self.cwd=os.getcwd();self.temp=tempfile.TemporaryDirectory();self.repo=Path(self.temp.name);os.chdir(self.repo)
        for name in ['assets/cylinder.urdf','assets/botyard_dexhand_gym/urdf/right_dexhand.urdf','grasp_seed_banks/current_verified/collision_excludes.json']:
            p=self.repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
        self.sources={}
        for scale,n,partial in [(1.,3,False),(.9,2,True)]:
            suffix='s'+str(scale).replace('.','');root=[0,0,.5,0,0,0,1]
            seed=self.repo/'grasp_seed_banks/calibrated_v4'/(suffix+'.json');seed.parent.mkdir(parents=True,exist_ok=True)
            seed.write_text(json.dumps(dict(candidates=[dict(actualJointNames=NAMES,handRoot7=root)])))
            states=np.zeros((n,27));states[:,20:27]=[0,0,.7,0,0,0,1];states[:,0]=np.arange(n)*.01
            targets=np.zeros((n,16));targets[:,0]=states[:,0]+.02
            meta=dict(jointNames=NAMES,objectScale=scale,handRootPoseLocal=root,complete=not partial,completedEpisodes=123,
                validationVersion='five_side_pose_motion_v3',torqueControl=True,controlFrequencyInv=6,
                collisionConfigSha256=hashlib.sha256(Path('grasp_seed_banks/current_verified/collision_excludes.json').read_bytes()).hexdigest(),
                solverPositionIterations=8,solverVelocityIterations=4,objectMassKg=.05,friction=.5,pgain=3.,dgain=.1,
                dt=float(np.float32(1/240)),warmupSeconds=.4,maxTiltDeg=2.,armature=.001,torqueLimit=.5,
                objectAsset='assets/cylinder.urdf',handAsset='assets/botyard_dexhand_gym/urdf/right_dexhand.urdf')
            for key in ['objectAsset','handAsset']:meta[key+'Sha256']=hashlib.sha256(Path(meta[key]).read_bytes()).hexdigest()
            source=Path('cache')/('botyard_multiscale_v4_grasp_3_'+suffix+('.partial' if partial else '')+'.npy')
            C.write_cache(source,states,targets,meta);self.sources[scale]=(source,states,targets,meta)
    def tearDown(self):os.chdir(self.cwd);self.temp.cleanup()
    def test_unequal_row_counts_and_targets_preserved_without_mutating_sources(self):
        before={s:(hashlib.sha256(v[0].read_bytes()).hexdigest(),hashlib.sha256(v[0].with_suffix('.meta.npz').read_bytes()).hexdigest()) for s,v in self.sources.items()}
        result=export_available(self.repo,[1.,.9],3,'available_test',C)
        self.assertEqual([result[str(s)]['rows'] for s in [1.,.9]],[3,2])
        for scale,(source,old,target,meta) in self.sources.items():
            states,targets,m=C.read_cache(result[str(scale)]['trainingCache'],NAMES,scale,meta['handRootPoseLocal'],meta['objectAsset'],meta['handAsset'])
            np.testing.assert_array_equal(states,old.astype(np.float32));np.testing.assert_array_equal(targets,target.astype(np.float32))
            self.assertTrue(m['complete']);self.assertEqual(m['originalGenerationComplete'],scale==1.)
            self.assertEqual(before[scale],(hashlib.sha256(source.read_bytes()).hexdigest(),hashlib.sha256(source.with_suffix('.meta.npz').read_bytes()).hexdigest()))
    def test_changed_controller_rejected(self):
        source,states,targets,meta=self.sources[.9];meta['pgain']=4.;C.write_cache(source,states,targets,meta)
        with self.assertRaises(ValueError):export_available(self.repo,[.9],3,'available_test',C)
    def test_available_phase_does_not_call_search_or_generation(self):
        events=[]
        task=types.SimpleNamespace(a=types.SimpleNamespace(phase='available'),summary={},save=lambda:None,out=self.repo,
            prepare_available=lambda:events.append('export'),check=lambda:events.append('check'),train=lambda:events.append('train'))
        Pipeline.execute(task);self.assertEqual(events,['export','check','train'])
    def test_named_dataset_paths_are_used_for_proof_and_training(self):
        task=types.SimpleNamespace(a=types.SimpleNamespace(cache_count=50000,cache_tag='available_test'),repo=self.repo)
        task.training_tag=lambda:Pipeline.training_tag(task)
        self.assertEqual(Pipeline.cache_path(task,.9,50000).name,'botyard_multiscale_v4_grasp_available_test_s09.npy')
        self.assertEqual(Pipeline.cache_path(task,.9,1).name,'botyard_multiscale_v4_grasp_1_s09.npy')

if __name__=='__main__':unittest.main()
