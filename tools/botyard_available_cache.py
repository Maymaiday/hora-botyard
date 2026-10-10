"""Export the accepted rows as an explicitly selected training dataset.

Original generation caches remain untouched. Completion refers to export of
available rows, not completion of the original 50k generation budget.
"""
import json,math,hashlib
from pathlib import Path

def export_available(repo,scales,source_count,tag,cache_api):
    repo=Path(repo);results={}
    count_tag=cache_api.count_tag(source_count)
    hand='assets/botyard_dexhand_gym/urdf/right_dexhand.urdf';obj='assets/cylinder.urdf'
    collision='grasp_seed_banks/current_verified/collision_excludes.json'
    for scale in scales:
        suffix='s'+str(float(scale)).replace('.','')
        seed=json.loads((repo/'grasp_seed_banks/calibrated_v4'/(suffix+'.json')).read_text())['candidates'][0]
        stem='botyard_multiscale_v4_grasp_'+count_tag+'_'+suffix
        source=repo/'cache'/(stem+'.npy')
        partial=False
        if not source.exists():source=repo/'cache'/(stem+'.partial.npy');partial=True
        states,targets,meta=cache_api.read_cache(source,seed['actualJointNames'],scale,seed['handRoot7'],obj,hand)
        if meta.get('complete') is not (not partial):raise ValueError('Unexpected source completion status: '+str(source))
        if not partial and len(states)!=source_count:raise ValueError('Complete source has wrong row count')
        exact=dict(validationVersion='five_side_pose_motion_v3',torqueControl=True,controlFrequencyInv=6,
            collisionConfigSha256=hashlib.sha256((repo/collision).read_bytes()).hexdigest(),solverPositionIterations=8,solverVelocityIterations=4)
        for key,value in exact.items():
            if meta.get(key)!=value:raise ValueError('Available source contract mismatch: '+key)
        numeric=dict(objectMassKg=.05,friction=.5,pgain=3.,dgain=.1,dt=1/240,warmupSeconds=.4,maxTiltDeg=2.,armature=.001,torqueLimit=.5)
        for key,value in numeric.items():
            if not math.isclose(float(meta.get(key,-1)),value,rel_tol=1e-6,abs_tol=1e-9):raise ValueError('Available source physics mismatch: '+key)
        for key,value in [('maxDriftM',.008),('maxDropM',.005)]:
            if not math.isclose(float(meta.get(key,value)),value,rel_tol=1e-6,abs_tol=1e-9):raise ValueError('Available source acceptance mismatch: '+key)
        target=repo/'cache'/('botyard_multiscale_v4_grasp_'+tag+'_'+suffix+'.npy')
        export_meta=dict(meta,complete=True,purpose='explicit_available_training_dataset',
            originalGenerationTarget=source_count,originalGenerationComplete=not partial,availableRowsSelected=len(states),
            sourceCache=str(source),sourceCacheSha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            sourceSidecarSha256=hashlib.sha256(source.with_suffix('.meta.npz').read_bytes()).hexdigest())
        cache_api.write_cache(target,states,targets,export_meta)
        results[str(float(scale))]=dict(rows=len(states),source=str(source),trainingCache=str(target),sourceWasPartial=partial)
        print('[AVAILABLE DATA]',scale,'rows',len(states),'source',source,flush=True)
    return results
