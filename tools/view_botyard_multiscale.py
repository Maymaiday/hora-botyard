"""View a v4 policy with mixed-size reset and constraint telemetry."""
import argparse,sys,time,math
from pathlib import Path
from run_botyard_full_pipeline import run_job,count_tag

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--cache-count',type=int,default=5000)
    p.add_argument('--scales',nargs='+',type=float,default=[1.,.9,.8]);p.add_argument('--seconds',type=float,default=120.)
    p.add_argument('--rl-device',choices=['cpu','cuda:0'],default='cpu');a=p.parse_args()
    if not math.isfinite(a.seconds) or a.seconds<=0:p.error('--seconds must be finite and positive')
    if a.cache_count<=0:p.error('--cache-count must be positive')
    repo=Path(__file__).resolve().parents[1]
    import os
    os.chdir(str(repo));cp=Path(a.checkpoint).resolve()
    if not cp.exists():p.error('Checkpoint not found')
    if not a.scales or len(set(a.scales))!=len(a.scales) or any(s not in (1.,.9,.8) for s in a.scales):p.error('Invalid scales')
    out=repo/'botyard_pipeline_runs'/('view_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True)
    cmd=[sys.executable,'-u','train_botyard.py','task=BotyardHandHora','test=True','checkpoint='+str(cp),
        'pipeline=cpu','sim_device=cpu','rl_device='+a.rl_device,'headless=False','num_subscenes=1',
        'task.env.numEnvs='+str(len(a.scales)),'task.env.randomization.randomizeScaleList='+str(a.scales).replace(' ',''),
        'task.env.graspCacheCountTag='+count_tag(a.cache_count),'task.env.debugMetricsEvery=20',
        'task.env.metricsPath='+str(out/'view_metrics.jsonl')]
    try:
        code=run_job(cmd,out/'view.log',a.seconds)
        # Viewing has no natural end in original HORA; the time limit is expected.
        return 0 if code==124 else code
    except KeyboardInterrupt:return 130

if __name__=='__main__':sys.exit(main())
