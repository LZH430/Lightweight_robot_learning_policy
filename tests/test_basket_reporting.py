import argparse,json,sys,tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import unittest
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from deployment_script.evaluate_basket_criteria import BasketEntryTracker,RULE_VERSION,RULE,summarize_basket_results,write_json_atomic
from deployment_script.evaluate_basket import run_trials

class ReportingTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def record(self,name,success,started=True,status='completed',group='a'):
  report={'status':status,'success':success,'evaluation_started':started,'basket_evaluation':{'rule_version':RULE_VERSION,'criterion':RULE},'evaluation_group':{'id':group,'checkpoint':group}}
  write_json_atomic(self.root/name/'summary.json',report)
 def test_latches_partial_entry_and_preserves_first_time(self):
  t=BasketEntryTracker({'interior_min':[-.1,-.1,0.],'interior_max':[.1,.1,.2]},length=.35,radius=.018)
  self.assertFalse(t.update([0,0,.5,1,0,0,0],.1))
  self.assertTrue(t.update([0,0,.3,1,0,0,0],.2))  # centre above rim; lower part inside
  self.assertTrue(t.update([2,2,2,1,0,0,0],.3))
  self.assertEqual(t.first_entry_time,.2)
  self.assertFalse(t.metadata()['requires_full_containment'])
 def test_counts_only_final_eligible_trials(self):
  self.record('success',True);self.record('failed_grasp',False);self.record('init_failure',False,status='failed')
  self.record('load_error',None,False,'failed');self.record('active',False,True,'started')
  write_json_atomic(self.root/'legacy/summary.json',{'success':None,'status':'completed'})
  r=summarize_basket_results(self.root)
  self.assertEqual((r['successes'],r['total_trials'],r['setup_errors'],r['pending']),(1,3,1,1))
  self.assertEqual(r['ignored_legacy_summaries'],1)
  self.assertEqual(summarize_basket_results(self.root)['total_trials'],3)
 def test_groups_different_models(self):
  self.record('a',True,group='adaptive');self.record('b',False,group='finetune')
  r=summarize_basket_results(self.root);rates={g['id']:g['basket_success_rate'] for g in r['groups']}
  self.assertEqual(rates,{'adaptive':1.,'finetune':0.})
 def test_concurrent_atomic_reports_and_recounts(self):
  def write(i):self.record('trial_'+str(i),bool(i%2));return summarize_basket_results(self.root)
  with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(write,range(12)))
  r=summarize_basket_results(self.root);self.assertEqual((r['successes'],r['total_trials']),(6,12))
  self.assertEqual(json.loads((self.root/'basket_success_rate.json').read_text())['total_trials'],12)
 def test_unreadable_reports_are_not_task_failures(self):
  (self.root/'bad').mkdir();(self.root/'bad/summary.json').write_text('{bad')
  write_json_atomic(self.root/'list/summary.json',[])
  r=summarize_basket_results(self.root);self.assertEqual(r['total_trials'],0);self.assertEqual(len(r['unreadable_summaries']),2)
 def test_batch_handles_success_failure_setup_error_and_native_exit(self):
  runner=self.root/'fake_runner.py'
  runner.write_text('''import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output');a,_=p.parse_known_args();out=Path(a.output);out.mkdir(parents=True)
n=int(out.name.rsplit('_',1)[1])
if n==6:sys.exit(1)
r={'status':'completed','evaluation_started':True,'success':n==1,'basket_evaluation':{'rule_version':1},'evaluation_group':{'id':'same','checkpoint':'test'}}
if n==3:r['status']='failed'
if n==4:r.update(status='failed',evaluation_started=False,success=None)
if n==5:r.update(status='started',success=True)
(out/'summary.json').write_text(json.dumps(r))
if n==5:sys.exit(1)
''')
  cfg=self.root/'config.json';cfg.write_text('{}')
  args=argparse.Namespace(config=str(cfg),sim_repo=str(self.root),output=str(self.root/'batch'),episodes=6,seconds=30.,observation_hz=30.,action_hz=15.,execute_steps=1,camera='front_camera',headless=True)
  r=run_trials(args,runner=runner)
  self.assertEqual((r['total_trials'],r['successes'],r['setup_errors']),(4,2,2))
  self.assertEqual(r['basket_success_rate'],.5)
  native=json.loads((self.root/'batch/episode_0005/summary.json').read_text())
  self.assertEqual(native['status'],'failed');self.assertTrue(native['success'])

 def test_batch_records_subprocess_launch_errors(self):
  cfg=self.root/'config.json';cfg.write_text('{}')
  args=argparse.Namespace(config=str(cfg),sim_repo=str(self.root),output=str(self.root/'batch'),episodes=2,seconds=30.,observation_hz=30.,action_hz=15.,execute_steps=1,camera='front_camera',headless=True)
  with patch('deployment_script.evaluate_basket.subprocess.run',side_effect=OSError('launch blocked')):
   result=run_trials(args)
  self.assertEqual((result['total_trials'],result['setup_errors']),(0,2))
  self.assertIn('Could not launch',json.loads((self.root/'batch/episode_0001/summary.json').read_text())['error'])

if __name__=='__main__':unittest.main()
