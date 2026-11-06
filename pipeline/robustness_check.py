"""Fresh paired, population-shift evaluation of unmodified exported PD agents.
Standard library only. Run explicitly; never imported by an agent.
"""
import concurrent.futures as futures
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parent
RESTORED=ROOT
import generate_training_data as g
from agent_export import load_agent
from prison import PAYOFF

OUT=ROOT/'robustness_check'
SEED=202610304913728
REPLICATES=12
MATCHES=16
NOISE=.02
CAM=RESTORED/'campaigns/diverse_selfplay_20261019'
PATHS={
 'policy_improvement_gru16': CAM/'runs/policy_improvement/models/gru16_seed1234/gru16_seed1234.py',
 'diverse_gru16': CAM/'runs/diverse/models/gru16_seed2345/gru16_seed2345.py',
 'original_gru16': RESTORED/'large_experiments/models/gru16_seed1234/gru16_seed1234.py',
}
CANDIDATES=['policy_improvement_gru16','diverse_gru16','two_strikes']
NEW=['handshake','hysteresis','probe_repair','payoff_comparer']
OLD=['cooldown_holdout','adaptive_holdout','probe_holdout']
POPS={
 'balanced_known': list(g.FAMILIES),
 'cooperative': ['cooperate','generous_tft','contrite_tft','tft_both','two_strikes','wsls'],
 'retaliatory': ['grim','tft_any','tft_one','two_strikes','majority','wsls'],
 'exploitative': ['defect','random','memory_one','switch','periodic','probe_repair'],
 'new_behaviours': NEW,
 'learned_mixed': ['learned:original_gru16','learned:policy_improvement_gru16','learned:diverse_gru16'],
 'previous_holdouts': OLD,
}
CLASSES={}

def seed_for(*parts):
 return int.from_bytes(hashlib.sha256(':'.join(map(str,(SEED,)+parts)).encode()).digest()[:8],'big')

def spec(family,rng):
 if family in g.FAMILIES:return g.sample_spec(family,rng)
 if family in OLD:return {'family':family,'window':rng.choice([3,7,12,20])}
 if family.startswith('learned:'):return {'family':family}
 return {'family':family,'window':rng.choice([4,8,15]),'low':rng.uniform(.1,.3),
         'high':rng.uniform(.45,.75),'period':rng.randint(9,23),'margin':rng.uniform(0,.7),
         'pattern':rng.choice([[0,1,0],[0,0,1],[1,0,0]])}

def action(s,own,first,second,rng):
 f=s['family'];t=len(own)
 if f in g.FAMILIES:return g.choose(s,own,first,second,rng)
 if not t:return s['pattern'][0] if f=='handshake' else 0
 w=min(t,s.get('window',10));rate=(sum(first[-w:])+sum(second[-w:]))/(2*w)
 if f=='cooldown_holdout':return int(sum(first[-3:])+sum(second[-3:])>=4 and t%6 not in (0,1))
 if f=='adaptive_holdout':
  rewards=[PAYOFF[a][b][c] for a,b,c in zip(own[-10:],first[-10:],second[-10:])]
  return own[-1] if statistics.fmean(rewards)>=4.5 else 1-own[-1]
 if f=='probe_holdout':
  if t%17 in (0,1):return 1
  return int(rate>.45 and not (own[-1] and t%17==2))
 if f=='handshake':
  if t<3:return s['pattern'][t]
  partner=all(sum(a!=b for a,b in zip(h[:3],s['pattern']))<=1 for h in (first,second))
  return int(rate>.6) if partner else int(rng.random()<.85)
 if f=='hysteresis':return int(rate>s['low']) if own[-1] else int(rate>s['high'])
 if f=='probe_repair':
  phase=t%s['period']
  if phase in (0,1):return 1
  if phase in (2,3,4):return 0
  return int(rate>.5)
 if f=='payoff_comparer':
  recent=list(zip(own[-w:],first[-w:],second[-w:]))
  mine=statistics.fmean(PAYOFF[a][b][c] for a,b,c in recent)
  others=statistics.fmean((PAYOFF[b][c][a]+PAYOFF[c][a][b])/2 for a,b,c in recent)
  return int(mine+s['margin']<others or rate>.65)
 raise ValueError(f)

def get_class(name):
 if name not in CLASSES:CLASSES[name]=load_agent(PATHS[name])
 return CLASSES[name]

def schedule():
 rows=[]
 for pop,families in POPS.items():
  for rep in range(REPLICATES):
   for match in range(MATCHES):
    seed=seed_for(pop,rep,match);rng=random.Random(seed)
    first=rng.choice(families)
    # New and old holdout suites always include at least one diagnostic family.
    # Half pair it with a known family; half with the same population.
    second=rng.choice(list(g.FAMILIES) if pop in ('new_behaviours','previous_holdouts','learned_mixed') and match%2 else families)
    opponents=[spec(first,rng),spec(second,rng)]
    if match%2:opponents.reverse()
    rounds=round(rng.gauss(100,10))
    while not 80<=rounds<=200:rounds=round(rng.gauss(100,10))
    rows.append({'population':pop,'replicate':rep,'match':match,'seed':seed,
                 'seat':(rep+match)%3,'rounds':rounds,'opponents':opponents})
 return rows

def play(candidate,row,trace=False,reuse=None,noise=NOISE):
 seat=row['seat'];hist=[[],[],[]];tot=[0,0,0];latency=0.;steps=[];cooperations=0
 agents={seat:(reuse if reuse is not None else get_class(candidate)())} if candidate in PATHS else {}
 specs={seat:{'family':'two_strikes'},(seat+1)%3:row['opponents'][0],(seat+2)%3:row['opponents'][1]}
 for s in range(3):
  f=specs[s]['family']
  if s!=seat and f.startswith('learned:'):agents[s]=get_class(f.split(':',1)[1])()
 rngs=[random.Random(seed_for(row['seed'],'policy',s)) for s in range(3)]
 channel=random.Random(seed_for(row['seed'],'channel'))
 for turn in range(row['rounds']):
  intended=[]
  for s in range(3):
   own,first,second=hist[s],hist[(s+1)%3],hist[(s+2)%3]
   if s in agents:
    start=time.perf_counter();a=agents[s].play(own,first,second)
    if s==seat:latency=max(latency,time.perf_counter()-start)
   else:a=action(specs[s],own,first,second,rngs[s])
   if type(a) is not int or a not in (0,1):raise ValueError('Invalid action')
   intended.append(a)
  executed=[a^int(channel.random()<noise) for a in intended]
  rewards=[PAYOFF[executed[s]][executed[(s+1)%3]][executed[(s+2)%3]] for s in range(3)]
  for s in range(3):tot[s]+=rewards[s];hist[s].append(executed[s])
  cooperations+=executed[seat]==0
  if trace:steps.append({'turn':turn,'intended':intended,'executed':executed,'rewards':rewards})
 return {'candidate':candidate,**row,'score':tot[seat]/row['rounds'],
         'other_scores':[tot[(seat+1)%3]/row['rounds'],tot[(seat+2)%3]/row['rounds']],
         'cooperation':cooperations/row['rounds'],'max_play_ms':latency*1000,**({'trace':steps} if trace else {})}

def worker(job):
 candidate,rep,rows=job
 target=OUT/'checkpoints'/f'{candidate}_rep{rep:02d}.json'
 if target.exists():return candidate,rep,'cached'
 result=[play(candidate,row) for row in rows]
 tmp=target.with_suffix('.pending');tmp.write_text(json.dumps(result),encoding='utf-8');tmp.replace(target)
 return candidate,rep,len(result)

def ci(values):
 mean=statistics.fmean(values);se=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else 0
 # Student t, 11 degrees of freedom, for the predeclared 12 schedule replicates.
 assert len(values)==REPLICATES
 return {'mean':mean,'ci95_low':mean-2.200985160082949*se,'ci95_high':mean+2.200985160082949*se}

def summarize():
 data=[]
 for path in sorted((OUT/'checkpoints').glob('*.json')):data.extend(json.loads(path.read_text()))
 assert len(data)==len(CANDIDATES)*len(POPS)*REPLICATES*MATCHES
 index={(r['candidate'],r['population'],r['replicate'],r['match']):r for r in data}
 summary=[]
 for pop in list(POPS)+['equal_weight_all_populations']:
  selected=list(POPS) if pop=='equal_weight_all_populations' else [pop]
  for candidate in CANDIDATES:
   scores=[];deltas=[];vswinner=[]
   for rep in range(REPLICATES):
    batch=[index[candidate,p,rep,m]['score'] for p in selected for m in range(MATCHES)]
    baseline=[index['two_strikes',p,rep,m]['score'] for p in selected for m in range(MATCHES)]
    incumbent=[index['policy_improvement_gru16',p,rep,m]['score'] for p in selected for m in range(MATCHES)]
    scores.append(statistics.fmean(batch));deltas.append(statistics.fmean(a-b for a,b in zip(batch,baseline)))
    vswinner.append(statistics.fmean(a-b for a,b in zip(batch,incumbent)))
   summary.append({'population':pop,'candidate':candidate,'matches':len(selected)*MATCHES*REPLICATES,
                   **ci(scores),**{'delta_two_strikes_'+k:v for k,v in ci(deltas).items()},
                   **{'delta_incumbent_'+k:v for k,v in ci(vswinner).items()}})
 with (OUT/'summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 # Preserve reproducible full round traces for the incumbent's five largest shortfalls to Two Strikes.
 worst=sorted([r for r in data if r['candidate']=='policy_improvement_gru16'],
              key=lambda r:r['score']-index['two_strikes',r['population'],r['replicate'],r['match']]['score'])[:5]
 traces=[]
 for row in worst:
  scenario={k:row[k] for k in ('population','replicate','match','seed','seat','rounds','opponents')}
  traces.extend(play(c,scenario,True) for c in CANDIDATES)
 (OUT/'worst_case_traces.json').write_text(json.dumps(traces))
 timing={c:max(r['max_play_ms'] for r in data if r['candidate']==c) for c in CANDIDATES}
 (OUT/'validation.json').write_text(json.dumps({'complete':True,'total_matches':len(data),
     'all_actions_valid':True,'max_play_ms_under_parallel_load':timing,'all_observed_calls_under_one_second':all(t<1000 for t in timing.values())},indent=2))
 print('COMPLETE',json.dumps(timing),flush=True)
 for r in summary:
  print(r['population'],r['candidate'],round(r['mean'],4),'delta vs Two Strikes',round(r['delta_two_strikes_mean'],4),flush=True)

def selfcheck():
 # Deterministic payoff reference and opponent-seat symmetry at zero noise.
 row={'population':'test','replicate':0,'match':0,'seed':123,'seat':0,'rounds':12,'opponents':[{'family':'cooperate'},{'family':'cooperate'}]}
 assert play('two_strikes',row,noise=0)['score']==6
 row['opponents']=[{'family':'defect'},{'family':'defect'}]
 assert abs(play('two_strikes',row,noise=0)['score']-20/12)<1e-12
 # Same channel seeds and match recreation must produce identical actions/scores.
 for c in CANDIDATES:
  a=play(c,row,True);b=play(c,row,True)
  assert a['trace']==b['trace']
  if c in PATHS:
   reused=get_class(c)();play(c,row,reuse=reused)
   assert play(c,row,True,reuse=reused)['trace']==a['trace']

def main():
 OUT.mkdir(exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True)
 rows=schedule();selfcheck()
 config={'seed':SEED,'noise':NOISE,'replicates':REPLICATES,'matches_per_population_per_replicate':MATCHES,
  'populations':POPS,'candidates':CANDIDATES,'rounds':'Rounded normal mean 100 sd 10, rejection truncated to [80,200]',
  'selection':'Candidates and populations fixed before fresh results; no retraining',
  'uncertainty':'Paired Student-t intervals across 12 independent complete schedule replicates; exploratory, no multiplicity adjustment',
  'aggregation':'Mean match payoff, equal match weight; all-populations summary gives each stress population equal weight, not an estimate of class frequencies',
  'old_holdouts':'Previously inspected families; fresh seeds only. They are a diagnostic, not a new untouched test.',
  'novel_opponents':'Custom handshakes, hysteresis, probe/repair, relative-payoff responses; absent from original training family list',
  'limitations':'Opponent populations are hand-designed; no prediction of performance against unknown agents. Learned policies are fixed snapshots.',
  'hashes':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in PATHS.items()},
  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 cfgpath=OUT/'protocol.json'
 if cfgpath.exists():assert json.loads(cfgpath.read_text())==config,'Protocol changed; use a new output directory'
 else:cfgpath.write_text(json.dumps(config,indent=2));(OUT/'schedule.json').write_text(json.dumps(rows))
 print(f'Starting {len(rows)*len(CANDIDATES):,} matches; 6 worker processes; self-checks passed.',flush=True)
 jobs=[(c,rep,[r for r in rows if r['replicate']==rep]) for rep in range(REPLICATES) for c in CANDIDATES]
 with futures.ProcessPoolExecutor(max_workers=6) as pool:
  pending=[pool.submit(worker,job) for job in jobs]
  for i,f in enumerate(futures.as_completed(pending),1):print(f'Progress {i}/{len(jobs)}:',f.result(),flush=True)
 summarize()

if __name__=='__main__':main()
