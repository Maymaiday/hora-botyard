"""Read an existing training run, restore hidden learning stats and bundle evidence.

Does not launch Gym, modify a policy, or start training.
"""
import argparse,json,re,zipfile
from pathlib import Path

NUM=r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
def read_log(text):
    progress=list(re.finditer(r'\[PPO PROGRESS\]\s+(\d+)\s*/\s*(\d+)',text))
    best=[float(x) for x in re.findall(r'save current best reward:\s*('+NUM+')',text)]
    current=[float(x) for x in re.findall(r'Current Best:\s*('+NUM+')',text)]
    return dict(reportedAgentSteps=int(progress[-1][1]) if progress else None,
        targetSteps=int(progress[-1][2]) if progress else None,
        bestSavedRecentReward=max(best) if best else (max(current) if current else None),
        bestRewardUpdates=best,normalBudgetCompletion='max steps achieved' in text)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1];run=a.run.resolve()
    summary_path=run/'summary.json';log=run/'train.log'
    summary=json.loads(summary_path.read_text())
    if not summary.get('output'):p.error('summary.json has no training output path')
    output=Path(summary['output']);output=output if output.is_absolute() else repo/output
    report=read_log(log.read_text(errors='replace'))
    report.update(run=str(run),output=str(output),sourceSnapshotTiming='current files at diagnosis; saved config/events belong to the completed run',tensorboard={},warnings=[])
    files=[summary_path,log];configs=sorted(output.glob('config_*.yaml'));files+=configs
    if configs:
        import yaml
        cfg=yaml.safe_load(configs[-1].read_text());env=cfg['task']['env'];sim=cfg['task']['sim']
        dt=float(sim['dt'])*int(env['controller']['controlFrequencyInv'])
        step=1/24 if env.get('trainingMode')=='hora_baseline' else float(env['controller']['actionStepRad'])
        report['trainingConfiguration']=dict(trainingMode=env.get('trainingMode','constrained'),
            controlDt=dt,controlHz=1/dt,maximumTargetIncrementRad=step,maximumTargetRateRadS=step/dt,
            forceScale=env.get('forceScale'),forceProbability=env.get('randomForceProbScalar'),
            learningRate=cfg['train']['ppo']['learning_rate'],episodeLength=env['episodeLength'],
            scales=env['randomization']['randomizeScaleList'])
    progress=output/'training_progress.jsonl'
    if progress.exists():
        rows=[json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
        report['progressTail']=rows[-10:];files.append(progress)
        if rows:report['reportedAgentSteps']=rows[-1]['agentSteps'];report['targetSteps']=rows[-1]['targetSteps']
    events=sorted((output/'stage1_tb').glob('events.out.tfevents.*'));files+=events
    if events:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ea=EventAccumulator(str(output/'stage1_tb'),size_guidance={'scalars':0});ea.Reload()
            tags=ea.Tags().get('scalars',[])
            for tag in ['episode_rewards/step','episode_lengths/step','rotation_reward','object_linvel_penalty','pose_diff_penalty','work_done','torques','info/kl','info/last_lr']:
                if tag not in tags:continue
                values=ea.Scalars(tag)
                if not values:continue
                report['tensorboard'][tag]=dict(first=values[0].value,last=values[-1].value,
                    maximum=max(v.value for v in values),minimum=min(v.value for v in values),
                    lastStep=values[-1].step,samples=len(values),tail=[dict(step=v.step,value=v.value) for v in values[-10:]])
        except Exception as exc:report['warnings'].append('TensorBoard extraction unavailable: '+str(exc)+'; raw events included')
    else:report['warnings'].append('No TensorBoard event file found')
    for rel in ['train_botyard.py','hora/tasks/botyard_hand_hora.py','hora/tasks/base/vec_task.py','hora/algo/ppo/ppo.py']:
        path=repo/rel
        if path.exists():files.append(path)
    destination=run/'training_diagnosis.json';destination.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    archive=run/'training_diagnosis.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files+[destination]:
            try:relative=path.relative_to(repo)
            except ValueError:relative=Path('external')/path.parent.name/path.name
            z.write(path,str(relative))
    steps=report['reportedAgentSteps'];print('训练步数：'+('未知' if steps is None else '%.3fM'%(steps/1e6)))
    print('历史最高已保存平均回合 reward：',report['bestSavedRecentReward'])
    for tag,label in [('episode_rewards/step','平均回合 reward'),('episode_lengths/step','平均回合长度（控制步）')]:
        value=report['tensorboard'].get(tag)
        if value:print(label+'：首条 %.3f，末条 %.3f，最高 %.3f'%(value['first'],value['last'],value['maximum']))
    for warning in report['warnings']:print('[说明]',warning)
    print('[报告]',destination);print('[打包]',archive)
    print('读取完成；未运行仿真或训练。')

if __name__=='__main__':main()
