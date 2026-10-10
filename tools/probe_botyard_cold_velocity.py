#!/usr/bin/env python3
"""Replay ONE passed lift/flip candidate, record every cold-start physics step."""
import argparse,json,sys,zipfile
from pathlib import Path
from search_botyard_two_stage import unique_out,run

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--replay',type=Path,default=Path('grasp_seed_banks/velocity_reference/s10_000.json'))
    ap.add_argument('--viewer',action='store_true');ap.add_argument('--plot',action='store_true')
    ap.add_argument('--timeout',type=float,default=120.);ap.add_argument('--out-dir',default='cold_velocity_results')
    a=ap.parse_args();source=a.replay.resolve()
    if not source.exists():ap.error('Passed lift/flip report not found: '+str(source))
    old=json.loads(source.read_text());can=old.get('canonical')
    if not can and not old.get('stage2',{}).get('passed'):ap.error('Need a passed lift/flip report, not a new grasp search')
    settings=json.loads((source.parent/'summary.json').read_text()).get('settings',{}) if (source.parent/'summary.json').exists() else {}
    if not can and not all(k in settings for k in ['dt','kp','kd','torque_limit','mass','friction','no_armature']):
        ap.error('Original summary.json is required to reproduce controller settings')
    scale=can['scale'] if can else old['scale']
    out=unique_out(a.out_dir);report=out/'cold_trace.json';status={}
    args=[sys.executable,'-u',str(Path(__file__).with_name('botyard_two_stage_worker.py')),
          '--repo',str(a.repo.resolve()),'--report',str(report),'--replay',str(source),
          '--scale',str(scale),'--cold-trace']
    for arg,key in [('dt','dt'),('kp','kp'),('kd','kd'),('torque_limit','torqueLimit'),('mass','mass'),('friction','friction')]:
        value=can[key] if can else settings[arg];args+=['--'+arg.replace('_','-'),str(value)]
    if (not can['usePhysxArmature'] if can else settings['no_armature']):args+=['--no-armature']
    if a.viewer:args+=['--viewer']
    print('[DIAGNOSTIC ONLY] same candidate, same controller, 0.4s settle + 3s hold; no lift/flip rerun',flush=True)
    try:
        status,data=run(args,out/'cold_trace.log',report,a.timeout)
        diagnostic=data.get('velocityDiagnostic')
        if diagnostic:
            (out/'velocity_summary.json').write_text(json.dumps(diagnostic['summary'],indent=2))
            if a.plot:
                from botyard_velocity_diagnostic import plot_diagnostic
                plot_diagnostic(diagnostic,out/'velocity_comparison.png')
        if status.get('exitCode')!=0 or not status.get('completed'):return 1
        print('[DONE DIAGNOSTIC] acceptance still uses original thresholds; accepted='+str(data.get('accepted')),flush=True)
        return 0
    except KeyboardInterrupt:
        status['interrupted']=True;return 130
    finally:
        (out/'probe_status.json').write_text(json.dumps(dict(sourceReport=str(source),worker=status,thresholdsChanged=False),indent=2))
        archive=out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in out.iterdir():
                if p.is_file():z.write(p,out.name+'/'+p.name)
        print('[RESULT ZIP] '+str(archive),flush=True)

if __name__=='__main__':sys.exit(main())
