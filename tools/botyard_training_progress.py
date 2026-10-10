"""PPO learning telemetry only. Does not change actions, losses or checkpoints."""
import json,math,time
from pathlib import Path

def number(value):
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (TypeError,ValueError):return None

def shown(value,digits=3):
    return 'NA' if value is None else ('%.'+str(digits)+'f')%value

class TrainingProgress:
    def __init__(self,agent,path,clock=time.monotonic):
        self.agent=agent;self.path=Path(path);self.clock=clock;self.last_time=clock()
        self.last_steps=None;self.best=None
        self.path.parent.mkdir(parents=True,exist_ok=True)
    def after_update(self):
        a=self.agent;now=self.clock();steps=int(a.agent_steps)
        count=len(a.episode_rewards)
        reward=number(a.episode_rewards.get_mean()) if count else None
        length=number(a.episode_lengths.get_mean()) if count else None
        if reward is not None:self.best=reward if self.best is None else max(self.best,reward)
        elapsed=now-self.last_time
        increment=int(a.batch_size) if self.last_steps is None else steps-self.last_steps
        control_dt=float(a.env.dt)*int(a.env.control_freq_inv)
        row=dict(agentSteps=steps,targetSteps=int(a.max_agent_steps),epoch=int(a.epoch_num),
            meanRecentEpisodeReward=reward,bestRecentEpisodeReward=self.best,
            meanRecentEpisodeLengthSteps=length,meanRecentEpisodeSeconds=None if length is None else length*control_dt,
            rewardWindowEpisodes=int(count),learningRate=number(a.last_lr),
            envStepsPerSecond=increment/elapsed if elapsed>0 else None,
            counterSemantics='upstream PPO agent_steps; recent completed-episode reward, not rotation success')
        with self.path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print('[训练] %.3fM / %.3fM | reward=%s | best=%s | 回合=%s步(%ss) | FPS=%s | LR=%s'%(
            steps/1e6,a.max_agent_steps/1e6,shown(reward),shown(self.best),shown(length,1),
            shown(row['meanRecentEpisodeSeconds'],2),shown(row['envStepsPerSecond'],0),shown(row['learningRate'],6)),flush=True)
        self.last_time=now;self.last_steps=steps
        return row
