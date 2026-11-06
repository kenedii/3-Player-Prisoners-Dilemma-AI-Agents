"""Frozen policy league and explicit three-player scripted adaptations."""
import random,pathlib
import numpy as np
import torch
import generate_training_data as g
from model_training import SmallModel

BASE_FAMILIES=tuple(g.FAMILIES)
TRAIN_NEW=('forgiving_majority','finite_punisher','gradual_retaliator','soft_reciprocator','opportunist')
HOLDOUT=('cooldown_holdout','adaptive_holdout','probe_holdout')
POOL=('gru16_seed1234','gru32_seed3456','mlp64_32_seed2345')
LEARNED=tuple('learned:'+p for p in POOL)
_choose=g.choose;_sample=g.sample_spec;MODELS={};ROOT=pathlib.Path(__file__).resolve().parent

def sample_spec(family,rng):
 if family.startswith('learned:'):return {'family':family}
 if family in TRAIN_NEW+HOLDOUT:
  return {'family':family,'window':rng.choice([3,7,12,20]),'threshold':rng.uniform(.25,.65),
          'forgive':rng.uniform(.1,.5),'punish':rng.choice([2,3,5]),'probe_every':rng.randint(7,19)}
 return _sample(family,rng)

def load_model(name):
 if name not in MODELS:
  tag=name.split(':',1)[-1];architecture=tag.rsplit('_seed',1)[0]
  model=SmallModel(architecture,50)
  saved=torch.load(ROOT/'large_experiments/models'/tag/'best.pt',map_location='cpu',weights_only=True)
  model.load_state_dict(saved['model']);model.eval();MODELS[name]=model
 return MODELS[name]

def choose(spec,own,first,second,rng):
 family=spec['family'];t=len(own)
 if family.startswith('learned:'):
  return choose_batch(spec,[(own,first,second)],[rng])[0]
 if family not in TRAIN_NEW+HOLDOUT:return _choose(spec,own,first,second,rng)
 if not t:return 0
 w=min(t,spec['window']);rate=(sum(first[-w:])+sum(second[-w:]))/(2*w)
 if family=='forgiving_majority':return int(rate>spec['threshold'] and rng.random()>spec['forgive'])
 if family=='soft_reciprocator':return int(rng.random()<max(.03,min(.97,.1+.8*(first[-1]+second[-1])/2)))
 if family=='opportunist':
  return int(rate<.15 and t%spec['probe_every']==0 or rate>.7)
 if family=='finite_punisher':
  recent=list(zip(own[-spec['punish']:],first[-spec['punish']:],second[-spec['punish']:]))
  return int(any(a==0 and (b or c) for a,b,c in recent))
 if family=='gradual_retaliator':
  offences=sum(a==0 and (b or c) for a,b,c in zip(own,first,second))
  last=max((i for i,(a,b,c) in enumerate(zip(own,first,second)) if a==0 and (b or c)),default=-1000)
  return int(t-last<=min(8,1+offences//3))
 if family=='cooldown_holdout':
  return int(sum(first[-3:])+sum(second[-3:])>=4 and t%6 not in (0,1))
 if family=='adaptive_holdout':
  rewards=[g.PAYOFF[a][b][c] for a,b,c in zip(own[-10:],first[-10:],second[-10:])]
  return own[-1] if sum(rewards)/len(rewards)>=4.5 else 1-own[-1]
 if family=='probe_holdout':
  if t%17 in (0,1):return 1
  return int(rate>.45 and not (own[-1] and t%17==2))
 raise ValueError(family)

def choose_batch(spec,histories,rngs):
 if not spec['family'].startswith('learned:'):return [choose(spec,*h,r) for h,r in zip(histories,rngs)]
 if not histories:return []
 x=np.asarray([g.encode_input(list(zip(*h)),50) for h in histories],dtype=np.float32)
 with torch.inference_mode():out=load_model(spec['family'])(torch.from_numpy(x))
 return (out[:,1]>out[:,0]).long().tolist()

def install():
 torch.set_num_threads(1)
 g.sample_spec=sample_spec;g.choose=choose

def rollout_targets(args,past,opponents,seed):
 """Paired C/D counterfactuals, batching frozen neural decisions across draws."""
 draws=args.rollouts;elapsed=len(past);states=[];lengths=[];stepseeds=[];continuations=[]
 for draw in range(draws):
  rng=random.Random(g.episode_seed(seed,draw));steps=min(args.rollout_horizon,g.surviving_length(args,elapsed,rng)-elapsed)
  lengths.append(steps)
  family=rng.choice(LEARNED) if args.continuation=='league' else args.continuation
  continuations.append(sample_spec(family,rng));stepseeds.append([rng.getrandbits(64) for _ in range(steps)])
  for initial in (0,1):states.append([[row[s] for row in past] for s in range(3)])
 totals=np.zeros((draws,2));denom=np.zeros(draws)
 for step in range(max(lengths)):
  indices=[i for i in range(draws*2) if step<lengths[i//2]]
  streams={i:[random.Random(g.episode_seed(stepseeds[i//2][step],s)) for s in range(4)] for i in indices}
  actions={i:[None,None,None] for i in indices}
  for seat in range(3):
   grouped={}
   for i in indices:
    spec=continuations[i//2] if seat==0 else opponents[seat-1]
    if seat==0 and step==0:actions[i][seat]=i%2;continue
    key=str(sorted(spec.items()));grouped.setdefault(key,(spec,[]))[1].append(i)
   for spec,group in grouped.values():
    histories=[(states[i][seat],states[i][(seat+1)%3],states[i][(seat+2)%3]) for i in group]
    selected=choose_batch(spec,histories,[streams[i][seat] for i in group])
    for i,a in zip(group,selected):actions[i][seat]=a
  for i in indices:
   noisy=[a ^ int(streams[i][3].random()<args.noise) for a in actions[i]]
   weight=args.discount**step;totals[i//2,i%2]+=weight*g.PAYOFF[noisy[0]][noisy[1]][noisy[2]]
   if i%2==0:denom[i//2]+=weight
   for seat in range(3):states[i][seat].append(noisy[seat])
 values=totals/denom[:,None];means=values.mean(0)
 se=values.std(0,ddof=1)/draws**.5;gap=(values[:,1]-values[:,0]).std(ddof=1)/draws**.5
 return means.tolist(),se.tolist(),float(gap)
