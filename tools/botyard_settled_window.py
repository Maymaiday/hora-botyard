"""Admit a measured settled suffix for SEARCH only; never waive cold-cache acceptance."""
import copy,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'hora/tasks'))
from botyard_pose_contract import evaluate

def settled_candidate(data,seconds=3.):
    rows=data.get('coldStart',{}).get('frames',[]);dt=float(data.get('canonical',{}).get('dt',1/240))
    count=int(round(seconds/dt))
    if len(rows)<count:raise ValueError('Insufficient settled-window samples')
    tail=rows[-count:]
    times=np.array([r['simTime'] for r in tail])
    if not np.allclose(np.diff(times),dt,rtol=1e-5,atol=1e-8):raise ValueError('Settled window is not sampled each physical step')
    converted=[dict(object7=r['object7'],contacts=r['contacts'],jointViolation=r['jointViolation'],sdkVelocity6=r['objectVelocity6']) for r in tail]
    # Orientation may still need whole-hand alignment. All contact, drift,
    # clearance and motion checks must pass in the measured three-second suffix.
    search=evaluate(converted,tail[0]['object7'],float(data['scale']),dt,warmup_seconds=0.,max_tilt=15.)
    if not search['passed']:raise ValueError('No stable five-finger suffix: '+str(search['reasons']))
    gap=np.array([np.array(r['target20'])[[1,2]]-np.array(r['q20'])[[1,2]] for r in tail[-int(round(1/dt)):]])
    if np.mean(np.any(gap>=.005,axis=1))<.95:raise ValueError('Settled suffix lacks forefinger preload proxy')
    out=copy.deepcopy(data);out.pop('canonical',None)
    out['settledSearchWindow']=dict(seconds=seconds,startSimTime=float(times[0]),endSimTime=float(times[-1]),
        sourceWholeTrialCheck=data.get('poseColdStartCheck'),searchCheck=search,acceptedForCache=False)
    out['coldStart']['frames']=tail;out['coldStart']['warmup']=[]
    strict=evaluate(converted,tail[0]['object7'],float(data['scale']),dt,warmup_seconds=0.)
    out['poseColdStartCheck']=strict;out['coldStart']['reasons']=strict['reasons'];out['coldStart']['passed']=False
    out.update(accepted=False,completed=True,stage='settled_search_candidate')
    return out
