"""Resumable 24h data collection followed by controlled training and independent evaluation."""
import pathlib,json,time,sys,os,subprocess,hashlib,shutil,collections,csv,statistics,traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import campaign_policies as p
from experiment_data import prepare_cache,load_cache,write_json

ROOT=pathlib.Path(__file__).resolve().parent
CAM=ROOT/'campaigns/diverse_selfplay_20261019'
ARMS=('control','diverse','selfplay','policy_improvement')
NEW_ARMS=ARMS[1:]
MODELS=['gru16','gru32','mlp64_32'];SEEDS=[1234,2345]
ENV={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
STATE={}

def state(**kwargs):
 STATE.update(kwargs);STATE['updated']=time.time();write_json(CAM/'state.json',STATE)

def run_checked(command,log,timeout=None):
 log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('a') as stream:
  process=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT)
  begin=time.time()
  while process.poll() is None:
   if time.time()>STATE['hard_deadline'] or timeout and time.time()-begin>timeout:
    process.terminate()
    try:process.wait(20)
    except subprocess.TimeoutExpired:process.kill();process.wait()
    raise TimeoutError('Campaign time budget reached; completed checkpoints preserved')
   updates={}
   if STATE.get('step')==3:
    updates['progress_done']=len(list((CAM/'runs').glob('*/models/*/training.json')))
    paths=list((CAM/'runs').glob('*/models/*/events.jsonl'))
    if paths:
     latest=max(paths,key=lambda f:f.stat().st_mtime)
     with latest.open('rb') as stream:
      stream.seek(max(0,latest.stat().st_size-10000));lines=stream.read().splitlines()
     try:
      event=json.loads(lines[-1]);updates['active_training']={'model':latest.parent.name,'arm':latest.parents[2].name,'event':event.get('event'),'epoch':event.get('epoch'),'batch':event.get('batch'),'loss':event.get('loss')}
     except (ValueError,IndexError):pass
   state(**updates);time.sleep(10)
  if process.returncode:raise RuntimeError(f'{pathlib.Path(command[1]).name} exited {process.returncode}; see {log.relative_to(CAM)}')

def aggregate_counts():
 result={}
 for arm in NEW_ARMS:
  total={'games':0,'examples':0,'by_composition':{},'by_focal_family':{},'by_opponent_pair':{},'by_split':{},'by_learned_model_seat':{}}
  for file in (CAM/'lanes').glob(arm+'_*/type_counts.json'):
   try:c=json.loads(file.read_text())
   except (FileNotFoundError,ValueError):continue
   for k in ('games','examples'):total[k]+=c[k]
   for field in ('by_composition','by_focal_family','by_opponent_pair','by_split','by_learned_model_seat'):
    for key,cell in c[field].items():
     x=total[field].setdefault(key,{'games':0,'examples':0});x['games']+=cell['games'];x['examples']+=cell['examples']
  result[arm]=total
 write_json(CAM/'generation_counts.json',result)
 with (CAM/'generation_counts.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['arm','dimension','type','games','labeled_examples'])
  for arm,c in result.items():
   w.writerow([arm,'total','all',c['games'],c['examples']])
   for field in ('by_composition','by_focal_family','by_opponent_pair','by_split','by_learned_model_seat'):
    for key,cell in c[field].items():w.writerow([arm,field,key,cell['games'],cell['examples']])
 return result

def generate():
 state(stage='New-data generation',step=1,detail='Collecting diverse games and frozen-policy self-play for 24 hours; no new model training during this stage.',progress_total=86400,progress_unit='seconds',progress_done=0)
 definitions=[('diverse',0),('diverse',1)]+[('selfplay',i) for i in range(2,6)]+[('policy_improvement',i) for i in range(6,10)]
 running={};retries=collections.Counter();finished=set()
 try:
  while time.time()<STATE['generation_deadline']:
   for arm,i in definitions:
    key=f'{arm}_{i}';lane=CAM/'lanes'/key;lane.mkdir(parents=True,exist_ok=True)
    if key in finished:continue
    proc=running.get(key)
    if proc and proc.poll() is not None:
     if proc.returncode==0:finished.add(key);continue
     retries[key]+=1
     if retries[key]>2:raise RuntimeError(f'Generation lane {key} failed repeatedly; inspect lane console.log')
     proc=None
    if proc is None:
     command=[sys.executable,str(ROOT/'generate_campaign_lane.py'),'--output',str(lane),'--arm',arm,'--worker',str(i),'--deadline',str(STATE['generation_deadline']),'--batch','16','--rollouts','16','--examples','8']
     with (lane/'console.log').open('a') as f:running[key]=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
   counts=aggregate_counts();state(counts=counts,progress_done=min(86400,int(time.time()-STATE['started'])))
   if shutil.disk_usage(ROOT).free<80*2**30:
    state(detail='Disk reserve reached; generation lanes finish their current atomic batch. Training remains scheduled after the 24-hour boundary.')
   time.sleep(20)
 finally:
  # Cooperative deadline is checked between small atomic batches; allow final commits.
  for proc in running.values():
   if proc.poll() is None:
    try:proc.wait(timeout=300)
    except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=30)
  state(counts=aggregate_counts())
 for arm in NEW_ARMS:
  if not list((CAM/'lanes').glob(arm+'_*/manifest.json')):raise RuntimeError('No committed data for '+arm)

def merge_lanes(arm):
 dest=CAM/'datasets'/arm;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'manifest.json').exists():return dest
 manifests=list((CAM/'lanes').glob(arm+'_*/manifest.json'));base=json.loads(manifests[0].read_text())
 merged={**base,'shards':[],'split_counts':{},'example_counts':{}}
 for file in manifests:
  m=json.loads(file.read_text());lane=file.parent.name
  for field in ('split_counts','example_counts'):
   for split,n in m[field].items():merged[field][split]=merged[field].get(split,0)+n
  for shard in m['shards']:
   rel=lane+'/'+shard['file'];target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
   if not target.exists():os.link(file.parent/shard['file'],target)
   merged['shards'].append({**shard,'file':rel})
 merged['config']['output']=str(dest);merged['campaign_lane_manifests']=[str(f.relative_to(CAM)) for f in manifests]
 write_json(dest/'manifest.json',merged);return dest

def combine_cache(arm,base_meta,base_arrays,new_meta,new_arrays,train_rows):
 out=CAM/'runs'/arm/'cache';out.mkdir(parents=True,exist_ok=True)
 if (out/'cache.json').exists():return
 meta={**base_meta,'identity':{'campaign':CAM.name,'arm':arm,'base_identity':base_meta['identity'],
   'new_identity':new_meta['identity'] if new_meta else None,'training_rows':train_rows,'replay_fraction':0 if arm=='control' else .5},'splits':{}}
 rng=np.random.default_rng(8471)
 for split in ('train','validation','test'):
  if split=='train':n=train_rows
  elif arm=='control':n=min(20000,len(base_arrays[split]))
  else:n=min(20000,2*len(new_arrays[split]),len(base_arrays[split]))
  if n<100:raise RuntimeError(f'Too little data in {arm}/{split}: {n}')
  if arm=='control':
   selected=rng.choice(len(base_arrays[split]),n,replace=False);values=np.asarray(base_arrays[split][selected],dtype=np.float32)
  else:
   half=n//2;selected=rng.choice(len(base_arrays[split]),n-half,replace=False);other=rng.choice(len(new_arrays[split]),half,replace=False)
   values=np.concatenate([base_arrays[split][selected],new_arrays[split][other]]);rng.shuffle(values)
  np.save(out/(split+'.npy'),values);meta['splits'][split]={'file':split+'.npy','rows':n}
  np.save(out/(split+'_base_indices.npy'),selected)
  if arm!='control':np.save(out/(split+'_new_indices.npy'),other)
 write_json(out/'cache.json',meta)

def prepare():
 state(stage='Freeze and pack datasets',step=2,detail='Freezing manifests, balancing training budgets, and retaining 50% original-data replay in every new-data arm.',progress_done=0,progress_total=4,progress_unit='datasets')
 refreshed=ROOT/'campaigns/home_refresh_20261019/training_data'
 refresh_status=ROOT/'campaigns/home_refresh_20261019/refresh.status'
 source=refreshed if refresh_status.exists() and refresh_status.read_text().startswith('EXIT=0') else ROOT/'training_data'
 datasets={arm:merge_lanes(arm) for arm in NEW_ARMS}
 min_n=min(json.loads((d/'manifest.json').read_text())['example_counts'].get('train',0) for d in datasets.values())
 train_rows=min(500000,2*min_n)
 if train_rows<10000:raise RuntimeError(f'Insufficient newly generated data for meaningful comparison: {min_n} train labels in smallest arm')
 prepare_cache(source,CAM/'raw_cache/control',{'train':max(500000,train_rows),'validation':20000,'test':20000},8471)
 bm,ba=load_cache(CAM/'raw_cache/control')
 cache_info={'base_source':str(source),'equal_training_rows_per_model':train_rows,'original_data_replay_fraction':.5,'source_dataset_counts':{}}
 for j,arm in enumerate(ARMS):
  nm=na=None
  if arm!='control':
   d=datasets[arm];manifest=json.loads((d/'manifest.json').read_text());cache_info['source_dataset_counts'][arm]=manifest['example_counts']
   prepare_cache(d,CAM/'raw_cache'/arm,{'train':min_n,'validation':20000,'test':20000},8471)
   nm,na=load_cache(CAM/'raw_cache'/arm)
  combine_cache(arm,bm,ba,nm,na,train_rows);state(progress_done=j+1)
 write_json(CAM/'data_design.json',cache_info)

def train():
 state(stage='Train controlled models',step=3,detail='Four dataset arms x three architectures x two seeds = 24 new models; CUDA training sequential, all exports checked.',progress_done=0,progress_total=24,progress_unit='models')
 for arm in ARMS:
  output=CAM/'runs'/arm
  command=[sys.executable,str(ROOT/'run_model_experiments.py'),'train','--suite','large','--output',str(output),'--models',*MODELS,'--train-seeds',*map(str,SEEDS),'--epochs','15','--patience','3','--classical-jobs','4','--batch-log-every','10','--trace-level','matches','--no-continue-on-error']
  state(detail=f'Training {arm}: GRU16, GRU32, MLP64/32, two seeds each. Detailed metrics retained locally.')
  run_checked(command,output/'train_console.log')
  failures=json.loads((output/'training_failures.json').read_text());assert not failures
  state(progress_done=len(list((CAM/'runs').glob('*/models/*/training.json'))))

def all_candidates():
 candidates=[]
 for arm in ARMS:
  for item in json.loads((CAM/'runs'/arm/'agent_index.json').read_text()):
   candidates.append({'name':arm+'__'+item['name'],'path':item['path'],'arm':arm})
 for tag in p.POOL:
  path=next((ROOT/'large_experiments/models'/tag).glob('*.py'))
  candidates.append({'name':'original__'+tag,'path':str(path),'arm':'original'})
 candidates.extend({'name':'script:'+f,'family':f,'arm':'script'} for f in ('two_strikes','defect','grim'))
 return candidates

def evaluation(candidates,stage_name,families,repeats,noises,seed):
 directory=CAM/stage_name;directory.mkdir(exist_ok=True)
 config=json.loads((ROOT/'training_data/manifest.json').read_text())['config']
 jobs=[]
 for i,candidate in enumerate(candidates):
  result=directory/f'candidate_{i:03d}.json';jobfile=directory/f'job_{i:03d}.json'
  job={'candidate':candidate,'config':config,'repeats':repeats,'seed':seed,'noises':noises,'timing_repeats':10,
       'log_directory':str(directory/f'candidate_{i:03d}_logs'),'trace_level':'matches','families':list(families),'result_path':str(result)}
  if jobfile.exists() and json.loads(jobfile.read_text())!=job:raise RuntimeError('Evaluation settings changed; refusing incompatible resume')
  write_json(jobfile,job);jobs.append((i,jobfile,result))
 write_json(directory/'settings.json',{'candidates':candidates,'families':list(families),'repeats':repeats,'noises':noises,'seed':seed})
 active={};retries=collections.Counter()
 try:
  while any(not result.exists() for _,_,result in jobs):
   if time.time()>STATE['hard_deadline']:raise TimeoutError('72-hour budget reached; partial evaluation rows retained')
   for i,(proc,stream) in list(active.items()):
    if proc.poll() is not None:
     stream.close();del active[i]
     if proc.returncode:
      retries[i]+=1
      if retries[i]>2:raise RuntimeError(f'{stage_name} candidate {i} failed repeatedly; inspect its worker log')
   for i,file,result in jobs:
    if len(active)>=4:break
    if result.exists() or i in active:continue
    stream=(directory/f'worker_{i:03d}.log').open('a');proc=subprocess.Popen([sys.executable,str(ROOT/'campaign_eval.py'),str(file)],cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT);active[i]=(proc,stream)
   rows=len(list(directory.glob('candidate_*_logs/noise*_row[0-9][0-9].json')))
   state(progress_done=rows,progress_total=len(candidates)*len(families)*len(noises),progress_unit='opponent rows',
         detail=f'{len([1 for _,_,f in jobs if f.exists()])}/{len(candidates)} candidates finished; four CPU evaluation workers. Each row covers {len(families)} opponent pairs x {repeats} repetitions.')
   time.sleep(20)
 finally:
  for proc,stream in active.values():
   if proc.poll() is None:
    proc.terminate()
    try:proc.wait(20)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
   stream.close()
 results=[json.loads(result.read_text()) for _,_,result in jobs if result.exists()]
 import benchmark_agents as b
 from campaign_eval import ci_stats
 b.ci_stats=ci_stats
 b.write_reports(results,directory)
 return results

def select_finalists(results,candidates):
 if (CAM/'finalists.json').exists():return json.loads((CAM/'finalists.json').read_text())
 by_name={c['name']:c for c in candidates};selected=[]
 for arm in ARMS:
  entries=[r for r in results if by_name[r['name']]['arm']==arm]
  entries.sort(key=lambda r:next(s['mean'] for s in r['summaries'] if s['noise']==.02),reverse=True)
  selected.extend(by_name[r['name']] for r in entries[:2])
 selected.extend(c for c in candidates if c['arm']=='original' and any(t in c['name'] for t in ('gru16_seed1234','gru32_seed3456')))
 selected.extend(c for c in candidates if c['arm']=='script')
 write_json(CAM/'finalists.json',selected);return selected

def finish_report(results):
 import benchmark_agents as b
 from campaign_eval import ci_stats
 b.ci_stats=ci_stats
 rows=[]
 reference=next(r for r in results if r['name']=='script:two_strikes')
 for r in results:
  for s in r['summaries']:
   ref=next(x for x in reference['summaries'] if x['noise']==s['noise'])
   paired=b.ci_stats([a-bb for a,bb in zip(s['replicate_means'],ref['replicate_means'])])
   rows.append({'candidate':r['name'],'noise':s['noise'],'subset':'all_opponent_pairs','mean':s['mean'],
                'delta_vs_two_strikes':paired['mean'],'delta_ci95_low':paired['ci95_low'],'delta_ci95_high':paired['ci95_high']})
   for subset,predicate in [('unseen_family_pairs',lambda x:x['opponent1'] in p.HOLDOUT or x['opponent2'] in p.HOLDOUT),
                            ('learned_opponent_pairs',lambda x:x['opponent1'].startswith('learned:') or x['opponent2'].startswith('learned:'))]:
    own=[x for x in r['pairs'] if x['noise']==s['noise'] and predicate(x)]
    refs=[x for x in reference['pairs'] if x['noise']==s['noise'] and predicate(x)]
    a=np.array([x['replicate_scores'] for x in own]).mean(0);bb=np.array([x['replicate_scores'] for x in refs]).mean(0);ci=b.ci_stats((a-bb).tolist())
    rows.append({'candidate':r['name'],'noise':s['noise'],'subset':subset,'mean':float(a.mean()),'delta_vs_two_strikes':ci['mean'],'delta_ci95_low':ci['ci95_low'],'delta_ci95_high':ci['ci95_high']})
 with (CAM/'paired_generalization_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 primary=sorted([r for r in rows if r['noise']==.02 and r['subset']=='all_opponent_pairs'],key=lambda x:x['mean'],reverse=True)
 lines=['# Diverse opponents and frozen-league self-play campaign','',
  'Four arms: original scripted-data control; expanded scripted families; mixed frozen-league self-play; and league continuation targets.',
  'All newly trained models use equal training-example budgets and two seeds. New-data arms retain 50% original-data replay. Generation lasted approximately 24 hours.',
  'Screening selected two candidates per arm on a separate seed and known-family schedule. Final evaluation uses a new seed and adds three policy families absent from all training and screening.',
  'The league uses frozen original-run checkpoints. This is one offline policy-improvement iteration, not online reinforcement learning or training against future checkpoints.',
  'Counterfactual targets use 16 paired Monte Carlo draws, 20-step horizons; they are continuation-policy values, not optimal Q values.',
  '', '| Candidate | 2% noise payoff | Paired improvement vs two_strikes | 95% interval |','|---|---:|---:|---:|']
 for r in primary:lines.append(f"| {r['candidate']} | {r['mean']:.4f} | {r['delta_vs_two_strikes']:+.4f} | [{r['delta_ci95_low']:.4f}, {r['delta_ci95_high']:.4f}] |")
 lines+=['','See generation_counts.csv for exact games/labels by arm, composition, focal family, opponent pair, split, and learned model/seat. Category dimensions overlap; do not add across dimensions.',
  'See paired_generalization_results.csv for held-out-family and learned-opponent subsets. Timings are hardware-specific; performance against unknown agents is not established by this benchmark.',
  'The final dataset seed and evaluation seeds must not be repurposed as training data in a later experiment that claims them as untouched holdouts.']
 (CAM/'REPORT.md').write_text('\n'.join(lines))
 top=primary[0];return f"Best observed at 2% noise: {top['candidate']} = {top['mean']:.4f}; paired difference vs two_strikes {top['delta_vs_two_strikes']:+.4f}. Full reports and type counts saved."

def main():
 CAM.mkdir(parents=True,exist_ok=True)
 if (CAM/'state.json').exists():STATE.update(json.loads((CAM/'state.json').read_text()))
 else:
  now=time.time();STATE.update(started=now,generation_deadline=now+86400,hard_deadline=now+72*3600,status='running',step=1,stage='Preflight')
 if STATE['status']=='complete':return
 state(status='running')
 try:
  import torch
  assert torch.cuda.is_available();assert torch.ones(4,device='cuda').sum().item()==4
  if not (CAM/'plan.json').exists():
   plan={'generation_hours':24,'maximum_campaign_hours':72,'new_models':24,'arms':ARMS,'architectures':MODELS,'seeds':SEEDS,
    'new_scripted_families':p.TRAIN_NEW,'heldout_families':p.HOLDOUT,'frozen_league':p.POOL,'screen_seed':2026102101,'final_seed':2026102201,
    'final_repeats':8,'final_noise':[0,.02,.05],'primary_noise':.02,'generation_workers':10,
    'rollouts':16,'rollout_horizon':20,'train_examples_cap':500000,'replay_fraction':.5,
    'code_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')},
    'pool_sha256':{tag:hashlib.sha256((ROOT/'large_experiments/models'/tag/'best.pt').read_bytes()).hexdigest() for tag in p.POOL}}
   write_json(CAM/'plan.json',plan)
  plan=json.loads((CAM/'plan.json').read_text())
  for tag,digest in plan['pool_sha256'].items():assert hashlib.sha256((ROOT/'large_experiments/models'/tag/'best.pt').read_bytes()).hexdigest()==digest,'Frozen league checkpoint changed'
  if time.time()<STATE['generation_deadline']:generate()
  prepare();train()
  candidates=all_candidates();state(stage='Screen candidate models',step=4,detail='Independent screening seed; reserved unseen-family policies remain excluded.',progress_done=0)
  screened=evaluation(candidates,'screening',p.BASE_FAMILIES+p.TRAIN_NEW+p.LEARNED,2,[.02],2026102101)
  finalists=select_finalists(screened,candidates)
  state(stage='Final generalization evaluation',step=5,detail='Frozen finalists; fresh paired schedule; unseen policies, learned opponents, and 0/2/5% noise.',progress_done=0)
  results=evaluation(finalists,'final_evaluation',p.BASE_FAMILIES+p.TRAIN_NEW+p.LEARNED+p.HOLDOUT,plan['final_repeats'],[0.,.02,.05],2026102201)
  summary=finish_report(results);state(status='complete',result_summary=summary,detail='All planned training, evaluation, export parity checks, and reports completed.',progress_done=STATE.get('progress_total',0))
 except BaseException as exc:
  (CAM/'failure_traceback.log').write_text(traceback.format_exc())
  state(status='budget_stopped' if isinstance(exc,TimeoutError) else 'failed',error=str(exc),detail='Completed datasets, models, and evaluation row checkpoints preserved. See failure_traceback.log.')
  raise

if __name__=='__main__':main()
