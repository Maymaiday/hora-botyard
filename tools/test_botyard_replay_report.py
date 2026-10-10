import contextlib,io,unittest
from botyard_replay_report import summarize,print_summary
from run_botyard_full_pipeline import TerminalLog

class ReplayReport(unittest.TestCase):
    def test_counts_are_separate_by_scale_and_reasons_overlap(self):
        checks=[dict(scale=1.,passed=True,reasons=[],env=0),
                dict(scale=.9,passed=False,reasons=['cylinder_tilt','missing_finger_side_contact'],env=1),
                dict(scale=.9,passed=False,reasons=['missing_finger_side_contact'],env=2)]
        summary=summarize(checks)
        self.assertEqual(summary['1.0']['passed'],1)
        self.assertEqual(summary['0.9']['passed'],0)
        self.assertEqual(summary['0.9']['reasons'],dict(missing_finger_side_contact=2,cylinder_tilt=1))
    def test_split_utf8_and_traceback_do_not_pollute_progress(self):
        text='[重放] 第 1/5 轮：25%\n[RUNNING] spam\nTraceback (most recent call last):\n[METRICS] {}\n[原因] 接触缺失\n'
        raw=text.encode('utf-8');out=io.StringIO()
        with contextlib.redirect_stdout(out):
            logger=TerminalLog()
            for i in range(0,len(raw),2):logger.feed(raw[i:i+2])
            logger.feed(b'',final=True)
        self.assertEqual(out.getvalue(),'[重放] 第 1/5 轮：25%\n[原因] 接触缺失\n')
    def test_empty_report_and_old_report_without_contact_details(self):
        self.assertEqual(summarize([]),{})
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            print_summary([dict(scale=.9,passed=False,reasons=['tail_pose_drift'],env=7,
                metrics=dict(tailPositionDriftM=.0004,tailOrientationDriftDeg=1.2))])
        self.assertIn('0.400 mm / 1.200°',out.getvalue())

if __name__=='__main__':unittest.main()
