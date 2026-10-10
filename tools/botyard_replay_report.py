"""Readable replay diagnostics; can inspect an existing report without Gym."""
import argparse,json
from collections import Counter
from pathlib import Path

LABELS={
    'cylinder_tilt':'圆柱倾斜超过 2°',
    'object_drift':'物体位移超过 8 mm',
    'object_drop_or_ground':'物体下落超过 5 mm 或触地',
    'missing_finger_side_contact':'至少一根手指缺少侧面接触',
    'palm_ground_or_cap_support':'手掌/地面/端盖支撑',
    'pose_motion_not_settled':'末段运动速度超限',
    'tail_pose_drift':'末段位姿漂移超限（1 mm / 1°）',
    'joint_limit':'关节越界',
    'unexpected_training_constraint_reset':'触发训练场景重置',
}

def summarize(checks):
    result={}
    for scale in sorted(set(float(c['scale']) for c in checks),reverse=True):
        group=[c for c in checks if float(c['scale'])==scale]
        failures=[c for c in group if not c['passed']]
        reasons=Counter(r for c in failures for r in set(c.get('reasons',[])))
        result[str(scale)]=dict(tested=len(group),passed=len(group)-len(failures),
            reasons=dict(reasons.most_common()),examples=[dict(env=c.get('env'),trial=c.get('trial'),
            reasons=c.get('reasons',[]),metrics=c.get('metrics',{}),contacts=c.get('sideContactSummary')) for c in failures[:3]])
    return result

def print_summary(checks):
    summary=summarize(checks)
    for scale,data in summary.items():
        print('[检查] 尺寸 '+scale+'：通过 '+str(data['passed'])+'/'+str(data['tested']),flush=True)
        for reason,count in data['reasons'].items():
            print('[原因] '+LABELS.get(reason,reason)+'：'+str(count)+' 个检查失败（原因可重叠）',flush=True)
        for example in data['examples'][:1]:
            metrics=example['metrics'];parts=[]
            if 'cylinder_tilt' in example['reasons']:parts.append('最大倾斜 %.3f°'%metrics.get('maxTiltDeg',float('nan')))
            if 'object_drift' in example['reasons']:parts.append('最大位移 %.3f mm'%(1000*metrics.get('maxDriftM',float('nan'))))
            if 'tail_pose_drift' in example['reasons']:
                parts.append('末段漂移 %.3f mm / %.3f°'%(1000*metrics.get('tailPositionDriftM',float('nan')),metrics.get('tailOrientationDriftDeg',float('nan'))))
            contact=example.get('contacts')
            if contact and 'missing_finger_side_contact' in example['reasons']:
                parts.append('各指侧面接触比例 '+json.dumps(contact,ensure_ascii=False))
            if parts:print('[示例] 环境 '+str(example['env'])+'：'+'；'.join(parts),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',nargs='?',default='cache/training_replay_v4.json')
    a=p.parse_args();data=json.loads(Path(a.report).read_text())
    print('训练前重放：'+('通过' if data.get('passed') else '未通过，未进入 PPO'))
    print('物理设备：'+str(data.get('simulationBackend',{}).get('simDevice','旧报告未记录')))
    print_summary(data.get('checks',[]))
