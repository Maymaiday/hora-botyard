import ast,unittest
from pathlib import Path
import numpy as np

class DiagnosticFlow(unittest.TestCase):
    def setUp(self):
        self.tree=ast.parse((Path(__file__).resolve().parents[1]/'tools/diagnose_botyard_grasp.py').read_text())
    def test_failed_trial_is_observed_without_reset_or_cache_save(self):
        cls=next(n for n in self.tree.body if isinstance(n,ast.ClassDef) and n.name=='DiagnosticTask')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='post_physics_step')
        # Execute the actual method body on a minimal task with a deliberate failure.
        scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'actual_diagnostic_method','exec'),scope)
        class FailedTask:
            progress_buf=np.array([0]);reset_buf=np.array([0]);actions=np.zeros((1,16))
            def __init__(self):self.events=[]
            def _refresh_gym(self):self.events.append('refresh')
            def compute_reward(self,actions):self.events.append('failure');self.reset_buf[:]=1
            def compute_observations(self):self.events.append('observe')
            def reset_idx(self,*args):raise AssertionError('Diagnostic must not reset')
            def _save(self,*args):raise AssertionError('Diagnostic must not save caches')
        task=FailedTask();scope['post_physics_step'](task)
        self.assertEqual(task.events,['refresh','failure','observe']);self.assertEqual(task.progress_buf[0],1);self.assertEqual(task.reset_buf[0],1)
    def test_initial_state_is_recorded_before_first_simulation_step(self):
        main=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        calls=[n for n in ast.walk(main) if isinstance(n,ast.Call)]
        reset=next(n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='reset_idx')
        initial=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='snapshot' and isinstance(n.args[0],ast.Constant) and n.args[0].value==0)
        step=next(n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='step')
        self.assertLess(reset.lineno,initial.lineno);self.assertLess(initial.lineno,step.lineno)
    def test_first_failure_and_full_trace_are_written(self):
        text=(Path(__file__).resolve().parents[1]/'tools/diagnose_botyard_grasp.py').read_text()
        self.assertIn('FIRST_FAILURE',text);self.assertIn('firstFailure',text);self.assertIn('frames',text)
        self.assertIn('out.write_text',text)

if __name__=='__main__':unittest.main(verbosity=2)
