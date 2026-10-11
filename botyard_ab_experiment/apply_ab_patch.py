#!/usr/bin/env python3
"""Idempotent insertion of optional recovery shaping into existing BotYard HORA task."""
from pathlib import Path
import shutil, datetime
p=Path('hora/tasks/botyard_hand_hora.py')
if not p.is_file(): raise SystemExit('请在 hora 项目根目录执行。')
s=p.read_text()
if 'AB_RECOVERY_PATCH_V1' in s:
    print('补丁已存在，不重复修改。'); raise SystemExit(0)
needle='        c=self.config[\'env\'][\'constraints\']\n        bad_drift='
if needle not in s: raise SystemExit('源码不匹配：未找到 compute_reward 的 constraints/bad_drift 段落，未修改。')
insert='''        # AB_RECOVERY_PATCH_V1: potential-difference tilt recovery shaping.
        # Compatible with old 96D observation / 16D action checkpoints.
        # This rewards REDUCING tilt, not merely remaining tilted.
        ab_cfg = self.config['env'].get('abRecovery', {})
        recovery_weight = float(ab_cfg.get('weight', 0.0))
        if recovery_weight != 0.0:
            prev_axis_z = 1 - 2*(self.object_rot_prev[:,0]**2 + self.object_rot_prev[:,1]**2)
            prev_tilt = torch.acos(torch.clamp(prev_axis_z.abs(), 0, 1))
            # normalized to 5 degrees. Clamp against numerical spikes.
            recovery_raw = torch.clamp(
                (prev_tilt - tilt) / np.deg2rad(5.0), -1.0, 1.0
            )
            # During reset, the initial prev quaternion should match current pose.
            self.rew_buf[:] += recovery_weight * recovery_raw
            self.extras['ab_recovery_reward'] = (recovery_weight * recovery_raw).mean()
        else:
            self.extras['ab_recovery_reward'] = self.rew_buf.new_zeros(())
'''
s=s.replace(needle,insert+needle,1)
# `self.extras` is reset during step() in original; assignment happens inside compute_reward.
backup=p.with_name(p.name+'.before_ab_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
shutil.copy2(p,backup); p.write_text(s)
print('补丁已应用:',p,'\n备份:',backup)
