"""Pure geometry and acceptance rules; no Isaac Gym import (Python >= 3.8)."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
POLICY = ['FFJ4','FFJ3','FFJ2','LFJ4','LFJ3','LFJ2','MFJ4','MFJ3','MFJ2','RFJ4','RFJ3','RFJ2','THJ4','THJ3','THJ2','THJ1']
ACTUAL = ['FFJ4','FFJ3','FFJ2','FFJ1','LFJ4','LFJ3','LFJ2','LFJ1','MFJ4','MFJ3','MFJ2','MFJ1','RFJ4','RFJ3','RFJ2','RFJ1','THJ4','THJ3','THJ2','THJ1']
WRIST = ['SEARCH_X','SEARCH_Y','SEARCH_Z','SEARCH_YAW','SEARCH_PITCH','SEARCH_ROLL']
FINGERS = ['ff','lf','mf','rf','th']
def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f: json.dump(data,f,indent=2,allow_nan=False)
    tmp.replace(path)
def transform(xyz, quat=None, rpy=None):
    t=np.eye(4); t[:3,3]=xyz
    t[:3,:3]=Rotation.from_quat(quat).as_matrix() if quat is not None else Rotation.from_euler('xyz',rpy).as_matrix()
    return t
def pose7(t): return np.r_[t[:3,3],Rotation.from_matrix(t[:3,:3]).as_quat()].tolist()
def expand(q):
    d=dict(zip(POLICY,q));d.update({p+'J1':d[p+'J2'] for p in ['FF','LF','MF','RF']})
    return np.array([d[n] for n in ACTUAL])
def smooth(u): return 10*u**3-15*u**4+6*u**5

def wrist_coordinates(anchor, world_pose, previous_rpy=None):
    local=np.linalg.inv(anchor)@world_pose
    rpy=Rotation.from_matrix(local[:3,:3]).as_euler('xyz')
    if previous_rpy is not None:rpy+=2*np.pi*np.round((np.asarray(previous_rpy)-rpy)/(2*np.pi))
    return np.r_[local[:3,3],[rpy[2],rpy[1],rpy[0]]],rpy

def upright_start_check(rows, requested_xyz, radius, length):
    if not rows:return ['no_startup_samples']
    faults=[];expected=np.asarray(requested_xyz)
    for r in rows:
        p=np.asarray(r['object7'][:3]);vel=np.asarray(r['objectVelocity6'])
        if not np.isfinite(np.r_[p,vel]).all():faults.append('nonfinite');continue
        axis=Rotation.from_quat(r['object7'][3:]).as_matrix()[2,2]
        if np.degrees(np.arccos(np.clip(axis,-1,1)))>5:faults.append('cylinder_tipped_before_approach')
        if np.linalg.norm(p[:2]-expected[:2])>.005:faults.append('cylinder_moved_before_approach')
        if p[2]<length/2-.003 or p[2]>length/2+.02:faults.append('invalid_ground_height')
        if np.linalg.norm(vel[:3])>.5 or np.linalg.norm(vel[3:])>3:faults.append('startup_impulse')
    tail=rows[-max(1,len(rows)//4):]
    if max(np.linalg.norm(r['objectVelocity6'][:3]) for r in tail)>.03 or max(np.linalg.norm(r['objectVelocity6'][3:]) for r in tail)>.3:faults.append('cylinder_not_settled')
    return sorted(set(faults))

def make_wrist_urdf(source, destination):
    source=Path(source).resolve();root=ET.parse(source).getroot()
    # Absolute filenames keep original collision meshes when generated URDF lives in result folder.
    for mesh in root.findall('.//mesh'):
        mesh.set('filename',str((source.parent/mesh.get('filename')).resolve()))
    parent='world'
    for i,(name,axis) in enumerate(zip(WRIST, ['1 0 0','0 1 0','0 0 1','0 0 1','0 1 0','1 0 0'])):
        child='search_link_'+str(i)
        link=ET.SubElement(root,'link',name=child); inertial=ET.SubElement(link,'inertial')
        ET.SubElement(inertial,'mass',value='0.02')
        ET.SubElement(inertial,'inertia',ixx='0.0001',iyy='0.0001',izz='0.0001',ixy='0',ixz='0',iyz='0')
        joint=ET.SubElement(root,'joint',name=name,type='prismatic' if i<3 else 'revolute')
        ET.SubElement(joint,'parent',link=parent);ET.SubElement(joint,'child',link=child)
        ET.SubElement(joint,'origin',xyz='0 0 0',rpy='0 0 0');ET.SubElement(joint,'axis',xyz=axis)
        ET.SubElement(joint,'limit',lower='-7' if i>=3 else '-2',upper='7' if i>=3 else '2',effort='500',velocity='5')
        parent=child
    root.find("joint[@name='FOREARM']/parent").set('link',parent)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    ET.ElementTree(root).write(str(destination),encoding='utf-8',xml_declaration=True)
    return destination

class Geometry:
    """FK of the actual independent 20 DOFs, not a mechanical mimic assumption."""
    def __init__(self,path):
        self.path=Path(path).resolve();root=ET.parse(self.path).getroot();self.joints=[];self.limits={};self.mesh={}
        def vec(e,key,default):return [float(x) for x in e.get(key,default).split()] if e is not None else [float(x) for x in default.split()]
        for j in root.findall('joint'):
            o=j.find('origin');axis=vec(j.find('axis'),'xyz','1 0 0');limit=j.find('limit')
            self.joints.append((j.get('name'),j.get('type'),j.find('parent').get('link'),j.find('child').get('link'),transform(vec(o,'xyz','0 0 0'),rpy=vec(o,'rpy','0 0 0')),np.array(axis)))
            if limit is not None and j.get('type')!='fixed':self.limits[j.get('name')]=(float(limit.get('lower')),float(limit.get('upper')))
        if set(self.limits)!=set(ACTUAL):raise ValueError('Expected original fixed-wrist Botyard hand with exactly 20 named DOFs')
        for link in root.findall('link'):
            collision=link.find('collision');mesh=collision.find('geometry/mesh') if collision is not None else None
            if mesh is None:continue
            b=(self.path.parent/mesh.get('filename')).read_bytes();count=int.from_bytes(b[80:84],'little')
            dt=np.dtype([('normal','<f4',3),('vertices','<f4',(3,3)),('attr','<u2')])
            v=np.frombuffer(b,dtype=dt,count=count,offset=84)['vertices'].reshape(-1,3).astype(float)
            scale=vec(mesh,'scale','1 1 1');v*=scale
            o=collision.find('origin');t=transform(vec(o,'xyz','0 0 0'),rpy=vec(o,'rpy','0 0 0'))
            self.mesh[link.get('name')]=v@t[:3,:3].T+t[:3,3]
    def fk(self,q20,base):
        values=dict(zip(ACTUAL,q20));out={'world':base};pending=list(self.joints)
        while pending:
            ready=[j for j in pending if j[2] in out]
            if not ready:raise ValueError('Disconnected hand URDF')
            for name,kind,parent,child,origin,axis in ready:
                motion=np.eye(4)
                if kind!='fixed':motion[:3,:3]=Rotation.from_rotvec(axis*values[name]).as_matrix()
                out[child]=out[parent]@origin@motion
            pending=[j for j in pending if j[3] not in out]
        return out
    def points(self,q20,base,links=None):
        fk=self.fk(q20,base)
        return {n:v@fk[n][:3,:3].T+fk[n][:3,3] for n,v in self.mesh.items() if links is None or n in links}
    def clamp(self,q):
        return np.array([np.clip(v,*self.limits[n]) for n,v in zip(POLICY,q)])

def sample_candidate(seed, index, rng, geo, scale):
    q=np.array(seed['canonicalPolicyPose'],float)
    # Thumb opposition is independent: do not copy Allegro's symmetric finger construction.
    if index:
        q[:12]+=np.tile([0.,rng.uniform(-.10,.10),rng.uniform(.03,.24)],4)
        q[[0,3,6,9]]+=rng.uniform(-.025,.025,4)
        q[12:]+=rng.uniform([-.10,-.15,-.08,-.08],[.10,.15,.12,.12])
    else:q[[2,5,8,11]]+=.08
    close=geo.clamp(q)
    opening=close.copy();opening[[1,4,7,10]]-=.40;opening[[2,5,8,11]]-=.16
    opening[15]-=.20;opening=geo.clamp(opening)
    xy=rng.uniform(-.005,.005,2) if index else np.zeros(2)
    band=float(rng.uniform(.002,.012)*scale) if index else .005*scale
    yaw=float(rng.uniform(-.10,.10)) if index else 0.
    goal=np.array(seed['handRotationRPY'],float);goal[2]+=yaw
    # Seeds use X roll only. This restriction makes the 180-degree path free of Euler singularities.
    if abs(goal[1])>1e-6:raise ValueError('Seed pitch must be zero for this wrist trajectory')
    r0=Rotation.from_euler('xyz',goal).as_matrix()
    local=r0.T@(np.array(seed['objectPosition'])-np.array(seed['handPosition']))
    r_start=Rotation.from_euler('xyz',[goal[0]+np.pi,0,goal[2]]).as_matrix()
    obj=np.array([0.,0.,.04*scale+.001])
    root_pos=obj-r_start@local+np.r_[xy,band]
    start=transform(root_pos,rpy=[goal[0]+np.pi,0,goal[2]])
    points=geo.points(expand(close),start)
    floor=min(float(v[:,2].min()) for v in points.values())
    palm_gap=float(points['palm'][:,2].min()-(obj[2]+.04*scale))
    # Full hand (including forearm) swept clearance, measured relative to the object pivot.
    relative=np.linalg.inv(start)@np.r_[obj,1.];sweep_min=0.
    for u in np.linspace(0,1,25):
        r=goal.copy();r[0]+=np.pi*(1-u);rot=Rotation.from_euler('xyz',r).as_matrix()
        centered=transform(-rot@relative[:3],rpy=r)
        sweep_min=min(sweep_min,min(float(v[:,2].min()) for v in geo.points(expand(close),centered).values()))
    required_pivot=.05-sweep_min
    lift_distance=max(.16,required_pivot-obj[2])
    pre_points=geo.points(expand(opening),transform(root_pos+[0,0,.10],rpy=[goal[0]+np.pi,0,goal[2]]))
    vertex_clearance=1.
    for v in pre_points.values():
        delta=np.stack([np.linalg.norm(v[:,:2]-obj[:2],axis=1)-.04*scale,np.abs(v[:,2]-obj[2])-.04*scale],axis=1)
        dist=np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(delta.max(axis=1),0)
        vertex_clearance=min(vertex_clearance,float(dist.min()))
    return dict(index=index,scale=scale,close16=close.tolist(),open16=opening.tolist(),rootStart7=pose7(start),goalRPY=goal.tolist(),objectStart7=np.r_[obj,[0,0,0,1]].tolist(),gripOffsetZ=band,xyOffset=xy.tolist(),meshFloorClearance=floor,palmGap=palm_gap,pregraspVertexClearance=vertex_clearance,liftDistance=lift_distance,requiredFlipPivotZ=required_pivot)

def contact_summary(contacts, object_body, body_names, radius, length):
    fingers=set();support=[];caps=0;normal_impulse=0.;side_z={}
    def field(c,n):
        try:return c[n]
        except (TypeError,KeyError,IndexError):return getattr(c,n)
    for c in contacts:
        a,b=int(field(c,'body0')),int(field(c,'body1'))
        if object_body not in (a,b):continue
        force=float(field(c,'lambda'))
        if force<=1e-8:continue
        other=b if a==object_body else a;name=body_names.get(other,'unknown_'+str(other))
        p=field(c,'localPos0' if a==object_body else 'localPos1')
        def axis(n):
            try:return float(p[n])
            except (TypeError,IndexError,KeyError):return float(getattr(p,n))
        x,y,z=axis('x'),axis('y'),axis('z');rad=np.hypot(x,y)
        # Margin is below contact_offset; corners/caps cannot count as side support.
        side=abs(z)<length/2-.002 and abs(rad-radius)<.005
        finger=next((prefix for prefix in FINGERS if name.startswith(prefix) and any(part in name for part in ['proximal','middle','distal'])),None)
        if finger and side:
            fingers.add(finger);normal_impulse+=force;side_z.setdefault(finger,[]).append(z)
        elif finger:caps+=1
        else:support.append(name)
    return dict(fingers=sorted(fingers),support=sorted(set(support)),capContacts=caps,contactLambdaSum=normal_impulse,sideContactZByFinger={k:float(np.mean(v)) for k,v in side_z.items()})

def hold_check(rows, minimum_height, max_slip=.008, max_angle=5., require_five=True):
    if not rows:return ['no_samples']
    failures=[];ref=transform(rows[0]['relativeObject7'][:3],quat=rows[0]['relativeObject7'][3:])
    for row in rows:
        rel=transform(row['relativeObject7'][:3],quat=row['relativeObject7'][3:])
        if not np.isfinite(np.r_[row['q20'],row['object7'],row['objectVelocity6']]).all():failures.append('nonfinite');continue
        if row['bottomZ']<minimum_height:failures.append('insufficient_lift')
        if np.linalg.norm(rel[:3,3]-ref[:3,3])>max_slip:failures.append('relative_slip')
        if Rotation.from_matrix(ref[:3,:3].T@rel[:3,:3]).magnitude()>np.deg2rad(max_angle):failures.append('relative_rotation')
        ct=row['contacts']
        if ct['support']:failures.append('external_or_palm_support')
        if ct['capContacts']:failures.append('cap_contact')
        if 'th' not in ct['fingers']:failures.append('no_thumb_side_contact')
        if len(ct['fingers'])<(5 if require_five else 3):failures.append('missing_finger_side_contacts')
        if row['jointViolation']>.01:failures.append('joint_limit')
        if row['wristErrorPosition']>.008 or row['wristErrorAngleDeg']>3:failures.append('wrist_tracking')
    tail=rows[-max(1,len(rows)//4):]
    if max(np.linalg.norm(r['objectVelocity6'][:3]) for r in tail)>.03:failures.append('not_settled_linear')
    if max(np.linalg.norm(r['objectVelocity6'][3:]) for r in tail)>.3:failures.append('not_settled_angular')
    return sorted(set(failures))
