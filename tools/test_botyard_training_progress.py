import contextlib,io,json,tempfile,types,unittest
from pathlib import Path
from botyard_training_progress import TrainingProgress
from report_botyard_training import read_log
from run_botyard_full_pipeline import TerminalLog

class Meter:
    def __init__(self,value=0,count=0):self.value=value;self.count=count
    def __len__(self):return self.count
    def get_mean(self):return self.value

class ProgressTests(unittest.TestCase):
    def test_no_finished_episode_is_na_and_negative_best_is_preserved(self):
        a=types.SimpleNamespace(agent_steps=8192,max_agent_steps=10000000,epoch_num=1,batch_size=4096,
            episode_rewards=Meter(),episode_lengths=Meter(),last_lr=.005,env=types.SimpleNamespace(dt=1/240,control_freq_inv=6))
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()) as out:
            times=iter([0.,2.,4.,6.]);logger=TrainingProgress(a,Path(d)/'progress.jsonl',clock=lambda:next(times))
            row=logger.after_update();self.assertIsNone(row['meanRecentEpisodeReward']);self.assertIsNone(row['bestRecentEpisodeReward'])
            a.episode_rewards=Meter(-5,100);a.episode_lengths=Meter(40,100);a.agent_steps+=4096
            row=logger.after_update();self.assertEqual(row['bestRecentEpisodeReward'],-5)
            self.assertEqual(row['meanRecentEpisodeSeconds'],1.)
            a.episode_rewards.value=-8;a.agent_steps+=4096;row=logger.after_update()
            self.assertEqual(row['bestRecentEpisodeReward'],-5);self.assertEqual(row['meanRecentEpisodeReward'],-8)
            rows=[json.loads(s) for s in (Path(d)/'progress.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),3);self.assertIn('0.016M / 10.000M',out.getvalue())
    def test_old_ppo_lines_pass_filter_and_rich_lines_avoid_duplicate_progress(self):
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            logger=TerminalLog();logger.feed(b'Agent Steps: 0001M | Current Best: -2.00\n')
            logger.feed('[训练] 1.001M | reward=-2 | best=-2\n'.encode())
            logger.feed(b'Agent Steps: 0001M | Current Best: -2.00\nsave current best reward: -1.50\n',final=True)
        self.assertEqual(out.getvalue().count('Agent Steps:'),1)
        self.assertIn('save current best reward: -1.50',out.getvalue())
    def test_existing_log_recovers_real_progress_and_negative_reward(self):
        result=read_log('[PPO PROGRESS] 9998336 / 10000000\nAgent Steps: 0009M | Current Best: -1.20\nsave current best reward: -2.00\nsave current best reward: -1.20\nmax steps achieved\n')
        self.assertEqual(result['reportedAgentSteps'],9998336)
        self.assertEqual(result['bestSavedRecentReward'],-1.2)
        self.assertTrue(result['normalBudgetCompletion'])

if __name__=='__main__':unittest.main()
