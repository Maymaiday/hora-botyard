"""Execute the real patched stepping/control methods with a deterministic fake backend."""
import ast,types,unittest,copy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def method(path,name):
    tree=ast.parse(path.read_text())
    return [n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name][-1]

def compile_method(node,scope):
    node=copy.deepcopy(node);node.returns=None
    for arg in node.args.args:arg.annotation=None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'patched_actual_method','exec'),scope)
    return scope[node.name]

class Array(np.ndarray):
    def clone(self):return self.copy().view(Array)

def arr(x):return np.array(x,dtype=float).view(Array)

class Feedback(unittest.TestCase):
    def test_cpu_substeps_fetch_before_next_feedback(self):
        step=method(ROOT/'hora/tasks/base/vec_task.py','step')
        loop=next(n for n in step.body if isinstance(n,ast.For))
        function=ast.FunctionDef(name='actual_loop',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[loop],decorator_list=[])
        run=compile_method(function,{})
        class Gym:
            def __init__(self):self.pending=False;self.events=[]
            def simulate(self,sim):self.assert_fetched();self.pending=True;self.events.append('simulate')
            def assert_fetched(self):
                if self.pending:raise AssertionError('PD would read unavailable/stale CPU results')
            def fetch_results(self,sim,blocking):self.pending=False;self.events.append('fetch')
        gym=Gym()
        task=types.SimpleNamespace(gym=gym,sim=object(),device='cpu',control_freq_inv=6,cpu_feedback_each_substep=True,render=lambda:None)
        def update():gym.assert_fetched();gym.events.append('PD')
        task.update_low_level_control=update;run(task)
        self.assertEqual(gym.events,['PD','simulate','fetch']*6);self.assertFalse(gym.pending)
    def test_original_order_is_rejected_by_backend(self):
        step=method(ROOT/'hora/tasks/base/vec_task.py','step');loop=next(n for n in step.body if isinstance(n,ast.For))
        function=ast.FunctionDef(name='actual_loop',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[loop],decorator_list=[])
        run=compile_method(function,{})
        pending=[False]
        def update():
            if pending[0]:raise RuntimeError('No CPU results since simulate')
        task=types.SimpleNamespace(device='cpu',cpu_feedback_each_substep=False,control_freq_inv=6,sim=None,render=lambda:None,update_low_level_control=update,gym=types.SimpleNamespace(simulate=lambda _:pending.__setitem__(0,True)))
        with self.assertRaises(RuntimeError):run(task)
    def test_simulator_velocity_produces_real_damping(self):
        n=method(ROOT/'hora/tasks/botyard_hand_hora.py','update_low_level_control')
        torch=types.SimpleNamespace(clip=lambda x,l,h:np.clip(x,l,h).view(Array))
        run=compile_method(n,{'torch':torch,'gymtorch':types.SimpleNamespace(unwrap_tensor=lambda x:x)})
        forces=[]
        task=types.SimpleNamespace(torque_control=True,allegro_hand_dof_pos=arr([[.2,.3]]),allegro_hand_dof_vel=arr([[1.,-2.]]),cur_targets=arr([[.25,.3]]),p_gain=arr([[3.,3.]]),d_gain=arr([[.1,.1]]),_refresh_gym=lambda:None,sim=None,gym=types.SimpleNamespace(set_dof_actuation_force_tensor=lambda sim,x:forces.append(x)))
        run(task);np.testing.assert_allclose(forces[0],[[.05,.2]],atol=1e-12)
    def test_requested_zero_gravity_is_preserved(self):
        n=method(ROOT/'hora/tasks/base/vec_task.py','set_sim_params_up_axis');run=compile_method(n,{'gymapi':types.SimpleNamespace(UP_AXIS_Z=2)})
        params=types.SimpleNamespace(gravity=types.SimpleNamespace(x=0.,y=0.,z=0.))
        self.assertEqual(run(None,params,'z'),2);self.assertEqual(params.gravity.z,0.)
    def test_default_gravity_is_preserved(self):
        n=method(ROOT/'hora/tasks/base/vec_task.py','set_sim_params_up_axis');run=compile_method(n,{'gymapi':types.SimpleNamespace(UP_AXIS_Z=2)})
        params=types.SimpleNamespace(gravity=types.SimpleNamespace(x=0.,y=0.,z=-9.81));run(None,params,'z');self.assertEqual(params.gravity.z,-9.81)

if __name__=='__main__':unittest.main(verbosity=2)
