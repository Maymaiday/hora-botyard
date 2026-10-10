"""Constraint mathematics and orchestration with explicit synthetic workers.

These tests do not simulate physics or establish native grasp stability.
"""
import copy,importlib.util,json,os,shutil,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import yaml
from run_botyard_full_pipeline import Pipeline,sha

ROOT=Path(__file__).resolve().parents[1]
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'hora/tasks'/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
C=module('botyard_training_contract')

class Constraints(unittest.TestCase):
    def test_target_never_runs_to_global_limits(self):
        ref=np.zeros(16);q=ref.copy();offset=np.ones(16)*.08
        for _ in range(1000):q=C.constrained_targets(q,np.ones(16),ref,offset,-np.ones(16),np.ones(16),.006)
        np.testing.assert_allclose(q,offset)
    def test_action_step_not_legacy_step(self):
        q=C.constrained_targets(np.zeros(16),np.ones(16),np.zeros(16),np.ones(16),-np.ones(16),np.ones(16),.006)
        np.testing.assert_allclose(q,.006)
    def test_normalized_pose_includes_distal_and_thumb(self):
        config=yaml.safe_load((ROOT/'configs/task/BotyardHandHora.yaml').read_text());weights=np.array(config['env']['constraints']['poseWeights20'])
        q=np.zeros(20);q[3]=.2
        terms=C.training_terms(q,np.zeros(20),np.full(20,.3),weights,[0,0,.7],[0,0,.7],[0,0,.7],[0,0,0,1],.025)
        self.assertGreater(terms['pose'],0)
        thumb=np.zeros(20);thumb[16]=.2
        other=C.training_terms(thumb,np.zeros(20),np.full(20,.3),weights,[0,0,.7],[0,0,.7],[0,0,.7],[0,0,0,1],.025)
        self.assertGreater(other['pose'],terms['pose'])
    def test_stopped_drift_still_penalized(self):
        t=C.training_terms(np.zeros(20),np.zeros(20),np.ones(20),np.ones(20),[.01,0,.7],[0,0,.7],[.01,0,.7],[0,0,0,1],.025)
        self.assertAlmostEqual(float(t['linearVelocity']),0);self.assertAlmostEqual(float(t['drift']),1)
    def test_vertical_axis_both_directions(self):
        for quat in ([0,0,0,1],[1,0,0,0]):
            t=C.training_terms(np.zeros(20),np.zeros(20),np.ones(20),np.ones(20),[0,0,.7],[0,0,.7],[0,0,.7],quat,.025)
            self.assertEqual(float(t['tilt']),0)
    def test_coordinate_detection_after_physics(self):
        roots=np.tile([0,0,.5],(3,1));origins=np.array([[0,0,0],[1,0,0],[0,1,0]])
        np.testing.assert_allclose(C.state_offsets(roots+origins,roots,origins)[0],origins)
        np.testing.assert_allclose(C.state_offsets(roots,roots,origins)[0],0)
        with self.assertRaises(RuntimeError):C.state_offsets(roots+.2,roots,origins)

class SyntheticPipeline(Pipeline):
    """Fake native job output for testing phase order and failure handling only."""
    failure_scale=None
    def calibrate(self,scale):
        seed=json.loads((ROOT/'grasp_seed_banks/current_verified/s10.json').read_text())['candidates'][0]
        seed=copy.deepcopy(seed);seed['objectScale']=scale;seed['source']='SYNTHETIC UNIT TEST ONLY'
        self.write_seed(scale,seed);self.summary['events'].append(dict(phase='calibrate_'+str(scale)))
    def job(self,name,args,timeout):
        self.summary['events'].append(dict(phase=name))
        if name.startswith('generate_'):
            scale=float(next(x.split('=',1)[1] for x in args if x.startswith('task.env.baseObjScale=')))
            if scale==self.failure_scale:return 1
            seed=json.loads(self.seed_path(scale).read_text())['candidates'][0]
            count=int(next(x.split('=',1)[1] for x in args if x.startswith('task.env.targetGraspCount=')))
            states=np.tile(seed['actualQ20']+seed['object7'],(count,1));targets=np.tile(seed['holdingTarget16'],(count,1))
            meta=dict(jointNames=seed['actualJointNames'],handRootPoseLocal=seed['handRoot7'],objectScale=scale,complete=True,
                objectAsset='assets/cylinder.urdf',handAsset='assets/botyard_dexhand_gym/urdf/right_dexhand.urdf',
                objectAssetSha256=seed['objectAssetSha256'],handAssetSha256=seed['handAssetSha256'],
                dt=1/240,pgain=3.,dgain=.1,torqueControl=True,validationVersion='five_side_pose_motion_v3',
                collisionConfigSha256=sha('grasp_seed_banks/current_verified/collision_excludes.json'))
            self.cache_api.write_cache(self.cache_path(scale,count),states,targets,meta)
            path=Path(next(x.split('=',1)[1] for x in args if x.startswith('task.env.graspReport=')))
            path.write_text(json.dumps(dict(complete=True,accepted=count)))
        elif name=='mixed_training_replay':
            Path('cache/training_replay_v4.json').write_text(json.dumps(dict(passed=True,numEnvs=self.a.envs,countTag=str(self.a.cache_count),
                simulationBackend=dict(simDevice=getattr(self.a,'sim_device','cpu'),physics='gpu' if getattr(self.a,'sim_device','cpu').startswith('cuda:') else 'cpu',tensorPipeline='cpu',controllerFeedback='every_physics_step'),
                fingerprints=self.fingerprints(),taskSourceSha256=sha('hora/tasks/botyard_hand_hora.py'),configSha256=sha('configs/task/BotyardHandHora.yaml'))))
        elif name=='train':(self.out/'train.log').write_text('SYNTHETIC: max steps achieved\n')
        return 0

class Workflow(unittest.TestCase):
    def setUp(self):
        self.cwd=os.getcwd();self.temp=tempfile.TemporaryDirectory();self.repo=Path(self.temp.name)
        for folder in ['hora','configs','assets','grasp_seed_banks']:shutil.copytree(ROOT/folder,self.repo/folder)
        self.args=SimpleNamespace(scales=[1.,.9,.8],cache_count=3,train_steps=4096,envs=3,fresh=False,
            viewer=False,rl_device='cpu',worker_timeout=10,cache_timeout=10,check_timeout=10,train_timeout=10,
            canonical_trials=1,resume_checkpoint=None,phase='all')
    def tearDown(self):os.chdir(self.cwd);self.temp.cleanup()
    def test_all_three_caches_before_check_and_training(self):
        pipeline=SyntheticPipeline(self.args,self.repo);pipeline.execute()
        phases=[e['phase'] for e in pipeline.summary['events']]
        self.assertEqual(phases[-2:],['mixed_training_replay','train'])
        self.assertEqual(sum(n.startswith('generate_') for n in phases),3)
        self.assertTrue(pipeline.summary['complete']);self.assertTrue(pipeline.proof_ok())
    def test_baseline_reads_existing_cache_and_trains_without_replay(self):
        self.args.scales=[1.];pipeline=SyntheticPipeline(self.args,self.repo)
        pipeline.calibrate(1.);pipeline.generate(1.,3)
        path=pipeline.cache_path(1.,3);before=(sha(path),sha(path.with_suffix('.meta.npz')))
        self.args.phase='baseline';pipeline.summary['events']=[];commands=[]
        def job(name,args,timeout):
            commands.append((name,args));return SyntheticPipeline.job(pipeline,name,args,timeout)
        pipeline.job=job;pipeline.execute()
        self.assertEqual([name for name,args in commands],['train'])
        args=commands[0][1]
        self.assertIn('+task.env.trainingMode=hora_baseline',args)
        self.assertIn('task.env.graspCacheCountTag=3',args)
        self.assertIn('train.ppo.horizon_length=8',args)
        self.assertEqual(before,(sha(path),sha(path.with_suffix('.meta.npz'))))
        self.assertFalse(pipeline.summary['pretrainingReplayRequired'])
    def test_baseline_missing_full_cache_is_reported_before_launch(self):
        self.args.scales=[1.];self.args.phase='baseline';pipeline=SyntheticPipeline(self.args,self.repo)
        with self.assertRaises(RuntimeError):pipeline.execute()
        self.assertEqual(pipeline.summary['events'],[])
    def test_failed_size_blocks_training(self):
        pipeline=SyntheticPipeline(self.args,self.repo);pipeline.failure_scale=.9
        with self.assertRaises(RuntimeError):pipeline.execute()
        self.assertNotIn('train',[e['phase'] for e in pipeline.summary['events']])
    def test_changed_seed_invalidates_replay_proof(self):
        pipeline=SyntheticPipeline(self.args,self.repo);pipeline.execute()
        path=pipeline.seed_path(.8);path.write_text(path.read_text()+'\n')
        self.assertFalse(pipeline.proof_ok())
    def test_cpu_proof_cannot_authorize_gpu_training(self):
        pipeline=SyntheticPipeline(self.args,self.repo);pipeline.execute()
        self.args.sim_device='cuda:0'
        self.assertFalse(pipeline.proof_ok())
        with self.assertRaises(RuntimeError):pipeline.train()
        pipeline.check()
        self.assertTrue(pipeline.proof_ok())
        self.assertIn('sim_device=cuda:0',pipeline.common(512,[1.,.9]))
        self.assertIn('pipeline=cpu',pipeline.common(512,[1.,.9]))
        self.args.sim_device='cpu'
        self.assertFalse(pipeline.proof_ok())
    def test_old_proof_without_backend_requires_new_replay(self):
        pipeline=SyntheticPipeline(self.args,self.repo);pipeline.execute()
        path=Path('cache/training_replay_v4.json');proof=json.loads(path.read_text())
        proof.pop('simulationBackend');path.write_text(json.dumps(proof))
        self.assertFalse(pipeline.proof_ok())
    def test_native_float32_metadata_is_accepted_but_different_dt_is_rejected(self):
        pipeline=SyntheticPipeline(self.args,self.repo);pipeline.calibrate(1.)
        self.assertTrue(pipeline.generate(1.,3))
        states,targets,meta=pipeline.validated_cache(1.,3)
        # Actual values from the user's 20261010_014656 native report.
        meta.update(dt=0.004166666883975267,pgain=float(np.float32(3.)),dgain=float(np.float32(.1)))
        pipeline.cache_api.write_cache(pipeline.cache_path(1.,3),states,targets,meta)
        self.assertIsNotNone(pipeline.validated_cache(1.,3))
        meta['dt']=1/120
        pipeline.cache_api.write_cache(pipeline.cache_path(1.,3),states,targets,meta)
        self.assertIsNone(pipeline.validated_cache(1.,3))

if __name__=='__main__':unittest.main(verbosity=2)
