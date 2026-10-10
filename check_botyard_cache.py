"""Replay all configured scales in the exact parallel training layout."""
import isaacgym
import os,sys,traceback,hashlib,time
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig
import gen_grasp
from hora.tasks.botyard_hand_hora import BotyardHandHoraVecTask
from hora.tasks.botyard_pose_contract import atomic_json
from hora.tasks.botyard_generation_batch import BatchTrace,observe_batch
from hora.utils.reformat import omegaconf_to_dict
from hora.utils.misc import set_seed
from tools.botyard_replay_report import print_summary

@hydra.main(config_name='config',config_path='configs')
def main(config:DictConfig):
    set_seed(config.seed);cfg=omegaconf_to_dict(config.task);cfg['rl_device']=config.rl_device
    cfg['env']['episodeLength']=10000
    cfg['env']['debugMetricsEvery']=0
    print('[阶段] 训练前重放：5 轮，每轮 %.0f 个环境，零动作保持 3.4 秒仿真时间'%cfg['env']['numEnvs'],flush=True)
    env=BotyardHandHoraVecTask(cfg,config.sim_device,config.graphics_device_id,config.headless)
    checks=[]
    for trial in range(5):
        started=time.monotonic()
        env.physics_observer=None;env.reset_buf[:]=1;env.reset();env._refresh_gym()
        starts=env.object_pose[:,:7].cpu().numpy().copy();starts[:,:3]-=env.state_offsets.cpu().numpy()
        steps=round(3.4/(env.dt*env.control_freq_inv))
        trace=BatchTrace(steps*env.control_freq_inv,env.num_envs)
        def observe(task):
            observe_batch(task,trace)
        env.physics_observer=observe;unexpected_reset=np.zeros(env.num_envs,dtype=bool)
        print('[重放] 第 %d/5 轮：开始'% (trial+1),flush=True)
        milestones={round(steps*f):int(100*f) for f in [.25,.5,.75,1.]}
        for step in range(steps):
            _,_,resets,_=env.step(env.zero_actions())
            unexpected_reset|=resets.cpu().numpy().astype(bool)
            if step+1 in milestones:
                print('[重放] 第 %d/5 轮：%d%%，耗时 %.1f 秒'%(trial+1,milestones[step+1],time.monotonic()-started),flush=True)
        batch=[]
        for i in range(env.num_envs):
            check=trace.check(i,starts[i],float(env.env_scales_np[i]),env.dt,.4,2.,.008,.005)
            if unexpected_reset[i]:check['passed']=False;check['reasons'].append('unexpected_training_constraint_reset')
            active=np.arange(1,trace.count+1)*env.dt>.4+1e-9
            contact=trace.contact[:trace.count,i][active]
            check['sideContactSummary']={f:round(float(np.mean((contact&(1<<j))!=0)),4) if len(contact) else 0. for j,f in enumerate(['ff','lf','mf','rf','th'])}
            check.update(env=i,scale=float(env.env_scales_np[i]),trial=trial+1);batch.append(check)
        checks+=batch
        print_summary(batch)
        if not all(c['passed'] for c in batch):break
    e=cfg['env'];fingerprints={}
    for scale in e['randomization']['randomizeScaleList']:
        tag=str(float(scale)).replace('.','');path=Path('cache')/(e['grasp_cache_name']+'_grasp_'+str(e['graspCacheCountTag'])+'_s'+tag+'.npy')
        fingerprints[str(float(scale))]=dict(states=hashlib.sha256(path.read_bytes()).hexdigest(),targets=hashlib.sha256(path.with_suffix('.meta.npz').read_bytes()).hexdigest(),seed=hashlib.sha256(Path(e['botyard']['scaleSeedFiles'][str(float(scale))]).read_bytes()).hexdigest())
    report=dict(passed=len(checks)==5*env.num_envs and all(c['passed'] for c in checks),checks=checks,
                simulationBackend=env.simulation_backend,
                numEnvs=env.num_envs,countTag=str(e['graspCacheCountTag']),fingerprints=fingerprints,
                taskSourceSha256=hashlib.sha256(Path('hora/tasks/botyard_hand_hora.py').read_bytes()).hexdigest(),
                configSha256=hashlib.sha256(Path('configs/task/BotyardHandHora.yaml').read_bytes()).hexdigest())
    atomic_json('cache/training_replay_v4.json',report)
    if not report['passed']:
        print('[停止] 重放未通过，尚未进入 PPO。失败详情：cache/training_replay_v4.json',flush=True)
        raise SystemExit(2)
    print('[检查] 五轮重放通过，可以进入 PPO',flush=True)

if __name__=='__main__':
    code=0
    try:main()
    except SystemExit as exc:code=int(exc.code or 0)
    except KeyboardInterrupt:code=130
    except BaseException:traceback.print_exc();code=1
    sys.stdout.flush();sys.stderr.flush();os._exit(code)
