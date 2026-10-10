"""Run actual reward-method AST with NumPy tensors; no native Gym claim."""
import ast,types,unittest
from pathlib import Path
import numpy as np

class A(np.ndarray):
    def abs(self):return arr(np.abs(self))
    def clone(self):return self.copy()
    def long(self):return self.astype(np.int64)
    def float(self):return self.astype(float)
def arr(x):return np.asarray(x).view(A)
def norm(x,p=2,dim=None):return arr(np.linalg.norm(x,ord=p,axis=dim))
def clamp(x,min=None,max=None):return arr(np.clip(x,min,max))

class BaselineReward(unittest.TestCase):
    def task(self):
        source=Path(__file__).resolve().parents[1]/'hora/tasks/botyard_hand_hora.py'
        cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef))
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='compute_reward')
        tree=ast.Module(body=[method],type_ignores=[]);ast.fix_missing_locations(tree)
        torch=types.SimpleNamespace(norm=norm,clip=clamp,clamp=clamp,
            abs=lambda x:arr(np.abs(x)),
            where=lambda c,a,b:arr(np.where(c,a,b)),acos=lambda x:arr(np.arccos(x)),
            any=lambda x,dim=None:arr(np.any(x,axis=dim)),rad2deg=lambda x:arr(np.rad2deg(x)),
            linalg=types.SimpleNamespace(norm=lambda x,dim:arr(np.linalg.norm(x,axis=dim))))
        namespace=dict(torch=torch,np=np,quat_conjugate=lambda q:q,quat_mul=lambda a,b:a,
            quat_to_axis_angle=lambda q:arr(np.zeros((len(q),3))),
            compute_hand_reward=lambda linear,ls,rotate,rs,pose,ps,torque,ts,work,ws:linear*ls+rotate*rs+pose*ps+torque*ts+work*ws)
        exec(compile(tree,str(source),'exec'),namespace)
        zero=arr(np.zeros((1,20)));q=zero.copy();q[0,19]=.1
        task=types.SimpleNamespace(hora_baseline=True,rot_axis_buf=arr([[0.,0.,0.]]),
            allegro_hand_dof_pos=q,init_pose_buf=zero,pose_weights=arr(np.ones(20)),pose_normalizers=arr(np.ones(20)*.3),
            cur_policy_targets=arr(np.zeros((1,16))),holding_reference=arr(np.zeros((1,16))),target_offset_limits=arr(np.ones(16)*.3),
            torques=arr(np.full((1,20),.02)),dof_vel_finite_diff=arr(np.full((1,20),.3)),
            object_rot=arr([[.70710678,0,0,.70710678]]),object_rot_prev=arr([[0,0,0,1]]),
            object_angvel=arr([[0,0,-.7]]),object_linvel=arr([[.1,.2,.3]]),
            object_pos=arr([[.1,0,.7]]),object_pos_prev=arr([[.1,0,.7]]),object_position_reference=arr([[0,0,.7]]),
            state_offsets=arr([[0,0,0]]),dt=1/240,control_freq_inv=6,reset_z_threshold=.645,
            angvel_clip_max=.5,angvel_clip_min=-.5,object_linvel_penalty_scale=-.3,
            rotate_reward_scale=1.,pose_diff_penalty_scale=-.3,torque_penalty_scale=-.1,work_penalty_scale=-2.,
            rew_buf=arr([0.]),reset_buf=arr([0]),progress_buf=arr([1]),max_episode_length=400,
            extras={},metric_counter=0,on_evaluation=False,evaluate=False,
            config=dict(env=dict(debugMetricsEvery=0,constraints=dict(positionPenaltyScale=-.15,tiltPenaltyScale=-.1,
                maxHorizontalDriftM=.02,maxVerticalDriftM=.03,maxDropM=.02,maxTiltDeg=15.,maxActualPoseOffsetMultiplier=2.,failurePenalty=-2.))))
        task.check_termination=lambda pos:arr((pos[:,2]<.645)|(task.progress_buf>=400))
        task.reward=lambda:namespace['compute_reward'](task,None)
        return task
    def test_author_five_term_reward_and_no_extra_tilt_drift_reset(self):
        task=self.task();task.reward()
        # Independent original-formula reference, generalized to 20 joints.
        expected=.5-.3*.6-.3*.01-.1*(20*.02**2)-2.*(20*.02*.3)**2
        self.assertAlmostEqual(float(task.rew_buf[0]),expected)
        self.assertEqual(task.reset_buf[0],0) # 90-degree tilt and 10cm drift don't add a new reset rule.
        self.assertAlmostEqual(float(task.extras['pose_diff_penalty']),.01)
    def test_original_height_reset_uses_environment_local_frame(self):
        task=self.task();task.state_offsets[0,2]=1.;task.object_pos[0,2]=1.64;task.reward()
        self.assertEqual(task.reset_buf[0],1)

if __name__=='__main__':unittest.main()
