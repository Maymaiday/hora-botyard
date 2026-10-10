"""Use the active hora interpreter; logs, heartbeat, Ctrl+C and time limits.

No installation or workspace-copy operations. One command launches one phase.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

def stop(process):
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['anchor','generate','check','train','view'])
    p.add_argument('--viewer',action='store_true');p.add_argument('--count',type=int,default=100)
    p.add_argument('--timeout',type=float,default=None);p.add_argument('--checkpoint')
    p.add_argument('--steps',type=int,default=4096,help='PPO agent steps; default is a short integration run')
    a=p.parse_args();repo=Path(__file__).resolve().parents[1];os.chdir(str(repo))
    if a.count<1 or a.steps<64:p.error('count must be positive and training steps >=64')
    tag=str(a.count//1000)+'k' if a.count%1000==0 else str(a.count)
    base=['pipeline=cpu','sim_device=cpu','rl_device=cpu','num_subscenes=1',
          'headless='+str(not(a.viewer or a.phase=='view')),'task.env.numEnvs=1']
    if a.phase in ('anchor','generate'):
        script='gen_grasp_botyard.py';over=['task=BotyardHandGrasp']
        count=1 if a.phase=='anchor' else a.count
        over+=['task.env.targetGraspCount='+str(count)]
        if a.phase=='anchor':over+=['task.env.graspPoseNoise=0.0','task.env.graspObjectXYNoise=0.0','task.env.graspObjectZNoise=0.0']
    else:
        over=['task=BotyardHandHora','task.env.graspCacheCountTag='+tag]
        if a.phase=='check':script='check_botyard_cache.py'
        else:
            script='train_botyard.py'
            if a.phase=='train':
                proof=repo/'cache/training_replay_v3.json'
                if not proof.exists():p.error('Run check --count %d first'%a.count)
                report=json.loads(proof.read_text());cache=repo/'cache'/('botyard_measured_v3_grasp_'+tag+'_s10.npy')
                if not report.get('passed') or not cache.exists() or report.get('statesSha256')!=hashlib.sha256(cache.read_bytes()).hexdigest() or report.get('targetsSha256')!=hashlib.sha256(cache.with_suffix('.meta.npz').read_bytes()).hexdigest():
                    p.error('Cache replay proof missing/stale; run check for this cache first')
                name='BotyardHandHora/measured_v3_'+time.strftime('%Y%m%d_%H%M%S')
                over+=['train.ppo.output_name='+name,'train.ppo.max_agent_steps='+str(a.steps)]
                print('[OUTPUT]',repo/'outputs'/name,flush=True)
            else:
                if not a.checkpoint:p.error('view requires --checkpoint PATH to best.pth/last.pth')
                cp=Path(a.checkpoint).resolve()
                if not cp.exists():p.error('Checkpoint does not exist: '+str(cp))
                over+=['test=True','checkpoint='+str(cp)]
    timeout=a.timeout if a.timeout is not None else {'anchor':120,'generate':3600,'check':180,'train':3600,'view':120}[a.phase]
    if timeout<=0:p.error('timeout must be positive')
    logs=repo/'botyard_run_logs';logs.mkdir(exist_ok=True)
    log=logs/(a.phase+'_'+time.strftime('%Y%m%d_%H%M%S')+'.log')
    cmd=[sys.executable,'-u',script]+over+base
    print('[RUN] interpreter:',sys.executable,'phase:',a.phase,'limit:',timeout,'seconds',flush=True)
    print('[COMMAND]',cmd,flush=True);print('[LOG]',log,flush=True)
    # Child logs go to disk so neither pipe buffering nor native cleanup can
    # prevent parent heartbeat / cancellation. Tail new bytes every .2 seconds.
    start=time.monotonic();started_wall=time.time();next_heartbeat=start+5;offset=0;reason=None
    with log.open('wb') as output:
        process=subprocess.Popen(cmd,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                with log.open('rb') as f:
                    f.seek(offset);data=f.read();offset=f.tell()
                if data:print(data.decode('utf-8',errors='replace'),end='',flush=True)
                code=process.poll();now=time.monotonic()
                if code is not None:break
                if now>=next_heartbeat:
                    print('[RUNNING]',a.phase,'elapsed',round(now-start,1),'s; log',log.name,flush=True);next_heartbeat=now+5
                if now-start>=timeout:reason='timeout';stop(process);break
                time.sleep(.2)
        except KeyboardInterrupt:reason='interrupted';stop(process)
    with log.open('rb') as f:
        f.seek(offset);data=f.read()
        if data:print(data.decode('utf-8',errors='replace'),end='',flush=True)
    code=process.returncode
    if code==0 and reason is None and a.phase in ('anchor','generate','check'):
        if a.phase=='check':
            proof=repo/'cache/training_replay_v3.json'
            completed=proof.exists() and proof.stat().st_mtime>=started_wall and json.loads(proof.read_text()).get('passed',False)
        else:
            count=1 if a.phase=='anchor' else a.count
            out_tag=str(count//1000)+'k' if count%1000==0 else str(count)
            cache=repo/'cache'/('botyard_measured_v3_grasp_'+out_tag+'_s10.npy')
            completed=cache.exists() and cache.stat().st_mtime>=started_wall and cache.with_suffix('.meta.npz').exists()
        if not completed:
            code=2;reason='exited_before_phase_completed'
    print('[DONE]',a.phase,'exit',code,'reason',reason,'log',log,flush=True)
    if a.phase=='generate' and code==0 and reason is None:
        print('[NEXT] python tools/run_botyard_pipeline.py check --count',a.count,flush=True)
    return 130 if reason=='interrupted' else 124 if reason=='timeout' else code

if __name__=='__main__':sys.exit(main())
