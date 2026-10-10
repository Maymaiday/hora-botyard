"""Botyard generation entry; original gen_grasp.py remains available."""
import isaacgym
import os
import sys
import traceback
import hydra
from omegaconf import DictConfig
import gen_grasp  # reuse this repository's OmegaConf resolver registration
from hora.tasks import isaacgym_task_map
from hora.utils.reformat import omegaconf_to_dict
from hora.utils.misc import set_seed

@hydra.main(config_name='config',config_path='configs')
def main(config:DictConfig):
    set_seed(config.seed)
    cfg=omegaconf_to_dict(config.task);cfg['rl_device']=config.rl_device
    print('[PHASE] building generation environment',flush=True)
    env=isaacgym_task_map[config.task_name](config=cfg,sim_device=config.sim_device,
            graphics_device_id=config.graphics_device_id,headless=config.headless)
    env.reset()
    print('[PHASE] zero-action holding trials started',flush=True)
    while True:env.step(env.zero_actions())

if __name__=='__main__':
    code=0
    try:main()
    except SystemExit as exc:code=int(exc.code or 0)
    except KeyboardInterrupt:code=130
    except BaseException:traceback.print_exc();code=1
    sys.stdout.flush();sys.stderr.flush()
    # Native Gym cleanup has hung on the user's installation; supervisor owns
    # the process group, and all reports/caches are written before termination.
    os._exit(code)
