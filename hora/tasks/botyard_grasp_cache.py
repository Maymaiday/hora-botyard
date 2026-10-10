"""Versioned 27-column BotYard cache plus aligned controller targets (NumPy only)."""
import hashlib,json,os
from pathlib import Path
import numpy as np
POLICY_NAMES=['FFJ4','FFJ3','FFJ2','LFJ4','LFJ3','LFJ2','MFJ4','MFJ3','MFJ2','RFJ4','RFJ3','RFJ2','THJ4','THJ3','THJ2','THJ1']

def count_tag(count):
    return str(count//1000)+'k' if count%1000==0 else str(count)

def validate_arrays(states,targets):
    states=np.asarray(states,dtype=np.float32);targets=np.asarray(targets,dtype=np.float32)
    if states.ndim!=2 or states.shape[1]!=27 or not len(states):raise ValueError('Cache must be nonempty [N,27]')
    if targets.shape!=(len(states),16):raise ValueError('Targets must be aligned [N,16]')
    if not np.isfinite(states).all() or not np.isfinite(targets).all():raise ValueError('Nonfinite cache value')
    if not np.allclose(np.linalg.norm(states[:,23:27],axis=1),1,atol=1e-3):raise ValueError('Invalid object quaternion xyzw')
    return states,targets

def write_cache(path,states,targets,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    states,targets=validate_arrays(states,targets)
    meta=dict(metadata);meta.update(version=2,rows=len(states),coordinateFrame='environment-local',policyJointNames=POLICY_NAMES)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:np.save(f,states,allow_pickle=False)
    meta['statesSha256']=hashlib.sha256(tmp.read_bytes()).hexdigest()
    side=path.with_suffix('.meta.npz');side_tmp=side.with_name(side.name+'.tmp')
    with side_tmp.open('wb') as f:np.savez_compressed(f,policyTargets=targets,metadata=np.array(json.dumps(meta)))
    os.replace(side_tmp,side);os.replace(tmp,path)
    return meta

def read_cache(path,joint_names,scale,hand_pose,object_asset,hand_asset):
    path=Path(path);states=np.load(path,allow_pickle=False)
    with np.load(path.with_suffix('.meta.npz'),allow_pickle=False) as side:
        targets=side['policyTargets'];meta=json.loads(str(side['metadata'].item()))
    states,targets=validate_arrays(states,targets)
    if meta.get('version')!=2 or meta.get('coordinateFrame')!='environment-local':raise ValueError('Unsupported cache metadata')
    if meta.get('rows')!=len(states):raise ValueError('Row count mismatch')
    if meta.get('jointNames')!=list(joint_names) or meta.get('policyJointNames')!=POLICY_NAMES:raise ValueError('Joint order mismatch')
    if not np.isclose(meta.get('objectScale',-1),scale):raise ValueError('Object scale mismatch')
    if not np.allclose(meta.get('handRootPoseLocal',[]),hand_pose,atol=2e-6):raise ValueError('Fixed hand root pose mismatch')
    for key,current in [('objectAsset',object_asset),('handAsset',hand_asset)]:
        if meta.get(key)!=current:raise ValueError(key+' mismatch')
        digest=hashlib.sha256(Path(current).read_bytes()).hexdigest()
        if meta.get(key+'Sha256')!=digest:raise ValueError(key+' file changed')
    if meta.get('statesSha256')!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError('Cache/target sidecar mismatch')
    return states,targets,meta

def terminal_success(valid_now,valid_history,progress,episode_length,warmup):
    return np.asarray(valid_now,dtype=bool)&np.asarray(valid_history,dtype=bool)&(np.asarray(progress)>=episode_length)&(np.asarray(progress)>warmup)

def contact_field(record,name):
    key=lambda s:s.replace('_','').lower()
    for field in (record.dtype.names or ()):
        if key(field)==key(name):return record[field]
    raise ValueError('Unsupported Isaac contact fields: '+str(record.dtype.names)+'; missing '+name)

def side_contact_summary(contacts,object_body,tip_handles,radius,length):
    touching=set();support=False
    for ct in contacts:
        b0=int(contact_field(ct,'body0'));b1=int(contact_field(ct,'body1'))
        if object_body not in (b0,b1):continue
        if float(contact_field(ct,'lambda'))<=0:continue
        other=b1 if b0==object_body else b0
        if other not in tip_handles:support=True;continue
        point=contact_field(ct,'localPos0' if b0==object_body else 'localPos1')
        xyz=np.array([float(point[k]) for k in ('x','y','z')])
        if abs(xyz[2])<length/2-.001 and abs(np.linalg.norm(xyz[:2])-radius)<.003:touching.add(other)
    return touching,support
