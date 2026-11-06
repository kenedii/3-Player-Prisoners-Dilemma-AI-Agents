import argparse,pathlib,time,json,os,gzip,collections
from types import SimpleNamespace
import generate_training_data as g
import campaign_policies as p

def account_committed(output,folders=None):
 path=output/'type_counts.json'
 c=json.loads(path.read_text()) if path.exists() else {'last_counted_end':0,'games':0,'examples':0,'by_composition':{},'by_focal_family':{},'by_opponent_pair':{},'by_split':{},'by_learned_model_seat':{}}
 for folder in sorted(output.glob('batch-*')) if folders is None else folders:
  commit=json.loads((folder/'commit.json').read_text())
  if commit['end_id']<=c['last_counted_end']:continue
  for shard in commit['shards']:
   with gzip.open(output/shard['file'],'rt') as stream:
    for line in stream:
     r=json.loads(line);n=len(r['mlp_examples']);families=[s['family'] for s in r['private_metadata']['strategies']]
     learned=[f for f in families if f.startswith('learned:')]
     category=('three_learned_identical' if len(set(learned))==1 else 'three_learned_mixed') if len(learned)==3 else ('all_scripted' if not learned else str(len(learned))+'_learned')
     keys={'by_composition':category,'by_focal_family':families[0],'by_opponent_pair':' | '.join(families[1:]),'by_split':shard['split']}
     for field,key in keys.items():
      cell=c[field].setdefault(key,{'games':0,'examples':0});cell['games']+=1;cell['examples']+=n
     for seat,f in enumerate(families):
      if f.startswith('learned:'):
       cell=c['by_learned_model_seat'].setdefault(str(seat)+':'+f,{'games':0,'examples':0});cell['games']+=1;cell['examples']+=n
     c['games']+=1;c['examples']+=n
  c['last_counted_end']=commit['end_id']
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(c));tmp.replace(path)
 return c

def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=pathlib.Path,required=True);a.add_argument('--arm',required=True)
 a.add_argument('--worker',type=int,required=True);a.add_argument('--deadline',type=float,required=True);a.add_argument('--seed',type=int,default=2026101900)
 a.add_argument('--batch',type=int,default=8);a.add_argument('--rollouts',type=int,default=8);a.add_argument('--examples',type=int,default=8)
 args=a.parse_args();p.install();g.rollout_targets=p.rollout_targets
 families=list(p.BASE_FAMILIES+p.TRAIN_NEW)
 if args.arm!='diverse':families+=list(p.LEARNED)*7  # Half learned, half scripted, sampled independently per seat.
 settings=SimpleNamespace(output=args.output,episodes=None,start_id=None,seed=args.seed+args.worker*100003,
  shard_episodes=args.batch,noise=.02,fixed_rounds=None,mean_rounds=100,std_rounds=10,min_rounds=80,max_rounds=200,
  families=families,test_only_families=[],exploration=[0,.05,.15,.3],gzip_level=1,window=50,
  examples_per_episode=args.examples,rollouts=args.rollouts,rollout_horizon=20,
  continuation='league' if args.arm=='policy_improvement' else 'tft_both',discount=1.)
 args.output.mkdir(parents=True,exist_ok=True)
 with g.dataset_lock(args.output):
  m=g.open_dataset(settings)
  account_committed(args.output)
  while time.time()<args.deadline:
   if __import__('shutil').disk_usage(args.output).free<80*2**30:
    print('Disk reserve reached; ending this generation lane cleanly',flush=True);break
   start=m['next_episode_id'];g.append_batch(settings,m,start,start+args.batch)
   account_committed(args.output,[args.output/f'batch-{start:020d}-{m["next_episode_id"]:020d}'])
   print(json.dumps({'episodes':sum(m['split_counts'].values()),'examples':sum(m['example_counts'].values()),'arm':args.arm,'worker':args.worker}),flush=True)
 print('GENERATION_LANE_COMPLETE',flush=True)

if __name__=='__main__':main()
