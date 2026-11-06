import unittest,random,pathlib,json,time,tempfile
import numpy as np
from types import SimpleNamespace
import generate_training_data as g
import campaign_policies as p
from agent_export import load_agent
original_rollout=g.rollout_targets
p.install()

class CampaignTests(unittest.TestCase):
 def test_balanced_caches_and_league_evaluation(self):
  import run_campaign as campaign
  import benchmark_agents as b
  old_cam=campaign.CAM
  with tempfile.TemporaryDirectory() as temp:
   campaign.CAM=pathlib.Path(temp)
   try:
    arrays={s:np.ones((200,217),dtype=np.float32) for s in ('train','validation','test')}
    new={s:np.ones((200,217),dtype=np.float32)*2 for s in arrays}
    meta={'identity':{'fixture':True},'window':50,'dimension':210,'source_config':{},'splits':{}}
    campaign.combine_cache('selfplay',meta,arrays,meta,new,200)
    data=np.load(campaign.CAM/'runs/selfplay/cache/train.npy')
    self.assertEqual(int((data[:,0]==1).sum()),100);self.assertEqual(int((data[:,0]==2).sum()),100)
   finally:campaign.CAM=old_cam
  prior=b.choose;b.choose=p.choose
  try:
   agent=load_agent(next((p.ROOT/'large_experiments/models'/p.POOL[0]).glob('*.py')))()
   result=b.play_match(agent,None,[{'family':'learned:'+p.POOL[1]},p.sample_spec(p.HOLDOUT[0],random.Random(5))],5,.02,1729,0)
   self.assertTrue(all(np.isfinite(x) for x in result))
  finally:b.choose=prior
 def test_type_accounting_is_exact_and_idempotent(self):
  import gzip
  from generate_campaign_lane import account_committed
  with tempfile.TemporaryDirectory() as temp:
   out=pathlib.Path(temp);folder=out/'batch-00000000000000000000-00000000000000000002';folder.mkdir()
   records=[{'private_metadata':{'strategies':[{'family':'learned:'+p.POOL[0]}]*3},'mlp_examples':[{},{}]},
            {'private_metadata':{'strategies':[{'family':'defect'},{'family':'cooperate'},{'family':'random'}]},'mlp_examples':[{}]}]
   with gzip.open(folder/'train.jsonl.gz','wt') as f:
    for r in records:f.write(json.dumps(r)+'\n')
   (folder/'commit.json').write_text(json.dumps({'end_id':2,'shards':[{'file':folder.name+'/train.jsonl.gz','split':'train'}]}))
   counts=account_committed(out);self.assertEqual(counts['games'],2);self.assertEqual(counts['examples'],3)
   self.assertEqual(counts['by_composition']['three_learned_identical']['examples'],2)
   self.assertEqual(counts,account_committed(out))
   for dimension in ('by_composition','by_focal_family','by_opponent_pair','by_split'):
    self.assertEqual(sum(v['examples'] for v in counts[dimension].values()),3)
 def test_paired_rollout_matches_original(self):
  a=SimpleNamespace(rollouts=4,rollout_horizon=8,discount=1.,noise=.02,fixed_rounds=None,mean_rounds=100,std_rounds=10,min_rounds=80,max_rounds=200,continuation='tft_both')
  specs=[p.sample_spec('generous_tft',random.Random(1)),p.sample_spec('memory_one',random.Random(2))]
  past=[[0,0,0],[1,1,0],[0,1,1]]
  old=original_rollout(a,past,specs,77);new=p.rollout_targets(a,past,specs,77)
  for x,y in zip(old,new):np.testing.assert_allclose(x,y,rtol=1e-12,atol=1e-12)
 def test_all_new_policies_and_holdouts_valid(self):
  rng=random.Random(7)
  for f in p.TRAIN_NEW+p.HOLDOUT:
   s=p.sample_spec(f,rng)
   for n in (0,1,10,100):
    h=[[rng.randrange(2) for _ in range(n)] for _ in range(3)]
    self.assertIn(p.choose(s,*h,rng),(0,1))
  self.assertFalse(set(p.TRAIN_NEW)&set(p.HOLDOUT))
 def test_frozen_neural_actions_match_standalone_exports(self):
  rng=random.Random(19)
  histories=[[[rng.randrange(2) for _ in range(n)] for _ in range(3)] for n in (0,1,5,50,100)]
  for tag in p.POOL:
   source=next((p.ROOT/'large_experiments/models'/tag).glob('*.py'))
   agent=load_agent(source)()
   actual=p.choose_batch({'family':'learned:'+tag},histories,[rng]*len(histories))
   self.assertEqual(actual,[agent.play(*h) for h in histories])
 def test_learned_counterfactual_labels_finite_and_deterministic(self):
  a=SimpleNamespace(rollouts=4,rollout_horizon=5,discount=1.,noise=.02,fixed_rounds=None,mean_rounds=100,std_rounds=10,min_rounds=80,max_rounds=200,continuation='league')
  specs=[{'family':'learned:'+p.POOL[0]},{'family':'learned:'+p.POOL[2]}]
  x=p.rollout_targets(a,[[0,0,0],[1,0,1]],specs,67);y=p.rollout_targets(a,[[0,0,0],[1,0,1]],specs,67)
  self.assertEqual(x,y);self.assertTrue(all(0<=v<=8 for v in x[0]));self.assertTrue(np.isfinite(np.array(x[1])).all())

if __name__=='__main__':unittest.main(verbosity=2)
