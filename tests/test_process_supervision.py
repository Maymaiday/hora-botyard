import importlib.util,json,os,signal,sys,tempfile,time,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('probe',Path(__file__).resolve().parents[1]/'tools/probe_botyard_controller.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

class SupervisionTests(unittest.TestCase):
    def run_case(self,source,timeout=2.,cleanup=.15):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);report=folder/'report.json';log=folder/'run.log'
            status=probe.run_logged([sys.executable,'-u','-c',source,str(report)],log,report,timeout,heartbeat=.1,cleanup_grace=cleanup)
            return status,log.read_text(),report.exists()

    def test_normal_exit_keeps_live_log(self):
        s,log,_=self.run_case("print('VISIBLE_PROGRESS',flush=True)")
        self.assertEqual(s['exitCode'],0);self.assertIsNone(s['stopReason'])
        self.assertIn('VISIBLE_PROGRESS',log)

    def test_stuck_child_ignoring_term_is_killed(self):
        s,log,_=self.run_case("import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('STUCK',flush=True);time.sleep(30)",timeout=.25)
        self.assertEqual(s['stopReason'],'trial_timeout');self.assertEqual(s['exitCode'],-signal.SIGKILL)
        self.assertLess(s['wallSeconds'],3.)

    def test_cleanup_hang_keeps_saved_report(self):
        s,log,exists=self.run_case("import signal,time,sys;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path(sys.argv[1]).write_text('{}');print('REPORT_WRITTEN',flush=True);time.sleep(30)")
        self.assertEqual(s['stopReason'],'cleanup_timeout');self.assertTrue(exists)
        self.assertEqual(s['exitCode'],-signal.SIGKILL)

    def test_ctrl_c_stops_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);pidfile=folder/'pid.txt'
            source="import os,signal,sys,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(.1);os.kill(os.getppid(),signal.SIGINT);time.sleep(30)"
            with self.assertRaises(KeyboardInterrupt):
                probe.run_logged([sys.executable,'-u','-c',source,str(pidfile)],folder/'run.log',folder/'absent.json',2.)
            pid=int(pidfile.read_text())
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)

if __name__=='__main__':unittest.main()
