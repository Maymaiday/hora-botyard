import json,subprocess,sys,tempfile,unittest
from pathlib import Path
TOOL=Path(__file__).resolve().parents[1]/'tools/diagnose_botyard_grasp.py'
CHILD=r'''
import ast,json,os,sys
from pathlib import Path
from types import SimpleNamespace
source=Path(sys.argv[1]).read_text();tree=ast.parse(source)
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
cleanup=next(n.finalbody for n in ast.walk(main) if isinstance(n,ast.Try) and any(isinstance(x,ast.Expr) and 'saving report' in ast.get_source_segment(source,x) for x in n.finalbody))
a=SimpleNamespace(steps=60,worker_exit_after_report=sys.argv[3]=='1')
rows=[{'step':int(sys.argv[4])}];header={};first_failure=None;out=Path(sys.argv[2])
e=SimpleNamespace(substep_rows=[],viewer=None,sim=None,gym=SimpleNamespace(destroy_sim=lambda _:print('DESTROY_CALLED',flush=True)))
exec(compile(ast.Module(body=cleanup,type_ignores=[]),'<actual diagnostic cleanup>','exec'))
'''
class WorkerExitTests(unittest.TestCase):
 def run_worker(self,flag,step):
  with tempfile.TemporaryDirectory() as tmp:
   report=Path(tmp)/'report.json'
   p=subprocess.run([sys.executable,'-u','-c',CHILD,str(TOOL),str(report),str(int(flag)),str(step)],capture_output=True,text=True,timeout=3)
   return p,json.loads(report.read_text())
 def test_complete_worker_skips_native_cleanup(self):
  p,d=self.run_worker(True,60)
  self.assertEqual(p.returncode,0,p.stderr);self.assertTrue(d['trialCompleted'])
  self.assertIn('[WORKER_EXIT]',p.stdout);self.assertNotIn('DESTROY_CALLED',p.stdout)
 def test_incomplete_worker_exits_with_failure(self):
  p,d=self.run_worker(True,12)
  self.assertEqual(p.returncode,1,p.stderr);self.assertFalse(d['trialCompleted'])
 def test_without_opt_in_keeps_normal_cleanup(self):
  p,d=self.run_worker(False,60)
  self.assertEqual(p.returncode,0,p.stderr);self.assertIn('DESTROY_CALLED',p.stdout)
if __name__=='__main__':unittest.main()
