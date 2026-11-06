"""Create reviewable figures and report after the fresh evaluation completes."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'robustness_check'
summary=json.loads((OUT/'summary.json').read_text())
validation=json.loads((OUT/'validation.json').read_text())
protocol=json.loads((OUT/'protocol.json').read_text())
assert validation['complete']
names=['policy_improvement_gru16','diverse_gru16','two_strikes']
labels=['Policy-improvement GRU16','Diverse GRU16','Two Strikes']
colors=['#007C91','#8C5BB5','#D17A00']
pops=list(protocol['populations'])+['equal_weight_all_populations']
titles=['Balanced known','Cooperative','Retaliatory','Exploitative','New behaviours','Learned mixed','Previous holdouts','Equal-weight average']
index={(r['population'],r['candidate']):r for r in summary}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.facecolor':'white'})
fig,axes=plt.subplots(1,2,figsize=(15,8.3))
fig.subplots_adjust(left=.17,right=.98,top=.81,bottom=.2,wspace=.38)
fig.suptitle('Fresh robustness check at 2% noise',x=.04,y=.97,ha='left',fontsize=23,weight='bold')
fig.text(.04,.915,'4,032 matches · 12 independent paired schedule replicates · Unmodified standalone Python agents',fontsize=11)
for j,name in enumerate(names):
 for i,pop in enumerate(pops):
  r=index[pop,name];y=len(pops)-1-i+(1-j)*.19
  axes[0].errorbar(r['mean'],y,xerr=[[r['mean']-r['ci95_low']],[r['ci95_high']-r['mean']]],fmt=['o','s','^'][j],color=colors[j],ms=6,capsize=3,label=labels[j] if i==0 else None)
  if j<2:
   m,lo,hi=[r['delta_two_strikes_'+k] for k in ('mean','ci95_low','ci95_high')]
   axes[1].errorbar(m,len(pops)-1-i+(.1 if j==0 else -.1),xerr=[[m-lo],[hi-m]],fmt=['o','s'][j],color=colors[j],ms=6,capsize=3)
for ax in axes:
 ax.set_yticks(list(reversed(range(len(pops)))),titles)
 ax.set_ylim(-.5,len(pops)-.5)
 ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
 ax.axhline(.5,color='#B7C0C8',lw=1)
axes[0].set_title('Average payoff by population',loc='left',pad=15)
axes[0].set_xlabel('Mean payoff per round (higher is better)')
axes[1].set_title('Paired advantage over Two Strikes',loc='left',pad=15)
axes[1].set_xlabel('Difference in payoff per round')
axes[1].axvline(0,color='#697582',lw=1)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.09),ncol=3,frameon=False)
fig.text(.04,.055,'Whiskers: 95% Student-t intervals across paired schedule replicates; exploratory comparisons without multiplicity adjustment.',fontsize=10)
fig.text(.04,.025,'Stress populations have equal weight in the average. Their real-world prevalence is unknown, so this is not a ranking prediction.',fontsize=10)
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'robustness_comparison.{ext}',dpi=180)
plt.close(fig)

lines=['# Fresh robustness check at 2% action noise','',
 'Candidates were fixed before these results: policy-improvement GRU16, diverse GRU16, and Two Strikes. No training or original evaluation files were changed.','',
 '4,032 fresh matches: 1,344 per candidate, 192 per population per candidate, arranged as 12 independent schedule replicates of 16 matches. Opponent specifications, match lengths, seats, and independent action-noise draws are paired across candidates. Match lengths use a rounded normal (mean 100, standard deviation 10) rejection-truncated to 80–200 rounds. This matches the target game setting.','',
 'Scores are equal-weight mean match payoffs. Confidence intervals use Student t across the 12 schedule replicates. These exploratory intervals have no multiple-comparison correction.','',
 '| Population | Policy-improvement GRU16 | Diverse GRU16 | Two Strikes | Diverse − incumbent (95% paired CI) |',
 '|---|---:|---:|---:|---|']
for pop,title in zip(pops,titles):
 rr=[index[pop,c] for c in names];r=rr[1]
 lines.append('| '+title+' | '+' | '.join(f'{x["mean"]:.4f}' for x in rr)+' | '+f'{r["delta_incumbent_mean"]:+.4f} [{r["delta_incumbent_ci95_low"]:+.4f}, {r["delta_incumbent_ci95_high"]:+.4f}] |')
lines+=['','## Paired differences versus Two Strikes','',
 '| Population | Policy-improvement difference (95% CI) | Diverse difference (95% CI) |','|---|---|---|']
for pop,title in zip(pops,titles):
 vals=[]
 for c in names[:2]:
  r=index[pop,c];vals.append(f'{r["delta_two_strikes_mean"]:+.4f} [{r["delta_two_strikes_ci95_low"]:+.4f}, {r["delta_two_strikes_ci95_high"]:+.4f}]')
 lines.append('| '+title+' | '+' | '.join(vals)+' |')
lines+=['','## Opponent populations','',
 '- Balanced known: the 16 original scripted training families, sampled uniformly.',
 '- Cooperative: cooperators, generous/contrite TFT, both-opponent TFT, Two Strikes, and win-stay/lose-switch.',
 '- Retaliatory: grim, any-opponent/one-opponent TFT, Two Strikes, majority, and win-stay/lose-switch.',
 '- Exploitative: defect, random, memory-one, switches, periodic defections, and probe/repair.',
 '- New behaviours: noisy handshakes (cooperate after an accepted signal, otherwise mostly defect), hysteresis (different entry/exit retaliation thresholds), periodic probing followed by cooperation to repair, and responses comparing own payoff with opponents’ payoffs. Half of these pairs mix a new family with an original family.',
 '- Learned mixed: frozen original, incumbent, and diverse GRU16 policies; half pair one learned policy with an original scripted family, half pair two learned policies. No adaptive online learning.',
 '- Previous holdouts: the earlier cooldown, adaptive, and probing families, using fresh random seeds. These have already been inspected and are diagnostic, not a fresh unseen-family test.',
 '', '## Validation and reproducibility','',
 '- Harness self-checks covered deterministic reference payoffs, repeatability, valid integer actions, and fresh-instance versus reused-instance behaviour.',
 f'- All {validation["total_matches"]:,} matches completed; every action was valid.',
 '- Maximum observed neural-agent decision time under six-process evaluation load: '+', '.join(f'{c}: {t:.2f} ms' for c,t in validation['max_play_ms_under_parallel_load'].items() if c!='two_strikes')+'. Baseline timing was not measured; its zero in validation.json is a placeholder.',
 '- Model-source SHA256 values and the evaluation-script hash are stored in protocol.json. schedule.json stores the full predeclared match schedule. Per-match results are in checkpoints/.',
 '- worst_case_traces.json records all rounds for each candidate on the incumbent’s five largest single-match shortfalls to Two Strikes. These are selected failure examples, not representative averages.',
 '', '## Findings and recommendation','',
 'Diverse GRU16 has the highest equal-weight mean (4.6325), compared with Two Strikes (4.6036) and the incumbent policy-improvement GRU16 (4.5941). Its paired gain over the incumbent is +0.0385 [0.0074, 0.0696]. Its gain over Two Strikes is +0.0289 [-0.0082, 0.0661], which does not establish an overall baseline advantage. All intervals are exploratory.',
 '',
 'Diverse GRU16 is the stronger existing candidate for the chosen mixture of stress populations, particularly cooperative and new-behaviour populations. The incumbent retains an advantage against the fixed learned-policy population and exploitative population. No candidate dominates across populations.',
 '',
 'A concrete failure is noise-triggered retaliation that fails to restore cooperation. In retaliatory replicate 5, match 15, a first-round opponent channel flip is followed by retaliation and persistent mutual defection. The incumbent scores 1.8929, diverse scores 1.4643, and Two Strikes scores 5.9286 on the same 112-round scenario. Both neural agents fail in this selected example. In new-behaviours replicate 3, match 5, against TFT-any plus hysteresis, the incumbent scores 2.0291 while diverse scores 5.8058 and Two Strikes 5.8252. These are selected examples, not population averages.',
 '',
 'If submitting an existing neural model now, prefer diverse GRU16 based on this stress-test convention, while preserving the incumbent. If time permits, a bounded follow-up should target recovery after accidental defection and sustained retaliation, and retain cooperative, exploitative, and learned-opponent coverage. Repeating the same training distribution at larger scale is not directly supported by these failures. Any changed agent needs a fresh locked confirmation schedule; these stress tests are now development evidence.',
 '', '## Interpretation limits','',
 'This evaluates resilience to several chosen opponent distributions. Equal weighting is a stress-test convention, not a claim about any real population of agents. A family can produce behaviours similar to training even if its exact rule is new. This does not establish universal superiority.','']
(OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print('\n'.join(lines[:18]))
