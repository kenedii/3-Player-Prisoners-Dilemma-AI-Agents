"""Cross-seed aggregates, joined training metrics, and an offline HTML report."""

import csv
import html
import json
from pathlib import Path
import statistics

from experiment_data import write_json


def finish_reports(results, root, report):
    root, report = Path(root), Path(report)
    aggregates = {}
    training = []
    for result in results:
        name = result["name"]
        if name.startswith("script:"):
            continue
        stats_path = root / "models" / name / "training.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            training.append({"candidate": name, "architecture": stats["architecture"], "seed": stats["seed"],
                "stored_parameters_or_numeric_fields": stats["parameters"],
                "training_rows": stats.get("training_examples", stats.get("fit", {}).get("training_rows")),
                "fit_or_training_seconds": stats.get("training_seconds", stats.get("fit", {}).get("fit_seconds")),
                "export_bytes": stats.get("export_source_bytes"),
                "validation_q_rmse": stats["validation"]["q_rmse_payoff_units"],
                "test_q_rmse": stats["test"]["q_rmse_payoff_units"],
                "test_teacher_regret": stats["test"]["teacher_regret_payoff_units"],
                "test_joint_nll": stats["test"]["joint_nll"],
                "test_joint_accuracy": stats["test"]["joint_accuracy"],
                "export_parity_passed": stats["export_parity"]["passed"]})
        for summary in result["summaries"]:
            key = (name.rsplit("_seed", 1)[0], summary["noise"])
            aggregates.setdefault(key, []).append(summary["mean"])
    rows = [{"architecture": name, "noise": noise, "seeds_completed": len(scores),
             "mean_score_across_seeds": statistics.fmean(scores),
             "stddev_across_seeds": statistics.stdev(scores) if len(scores) > 1 else None,
             "minimum_seed_score": min(scores), "maximum_seed_score": max(scores)}
            for (name, noise), scores in aggregates.items()]
    for filename, data in (("architecture_summary.csv", rows), ("training_metrics.csv", training)):
        with (report / filename).open("w", encoding="utf-8", newline="") as stream:
            if data:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader(); writer.writerows(data)
    scores = []
    for result in results:
        deployment = result.get("deployment")
        for summary in result["summaries"]:
            scores.append({"candidate": result["name"], "noise": summary["noise"],
                "mean": summary["mean"], "ci_low": summary["ci95_low"], "ci_high": summary["ci95_high"],
                "matches": summary["matches"], "max_play_ms": deployment["max_play_ms"] if deployment else None,
                "source_bytes": deployment["source_bytes"] if deployment else None,
                "observed_under_one_second": deployment["observed_under_one_second"] if deployment else None})
    write_json(report / "performance_summary.json", scores)
    payload = json.dumps(scores).replace("<", "\\u003c")
    page = '''<!doctype html><meta charset="utf-8"><title>PD experiment comparison</title>
<style>body{font:15px system-ui;margin:30px;max-width:1300px;background:#101923;color:#e6edf3}
table{border-collapse:collapse;width:100%}th,td{padding:9px;text-align:left;border-bottom:1px solid #344050}
th{cursor:pointer;background:#1b2b3b}input{padding:10px;width:330px;margin-bottom:20px}a{color:#8bcfff}
.bad{color:#ff998d}small{color:#aebccc}</style>
<h1>Simulated agent performance</h1><p>Absolute mean payoff per round. Click a column to sort; filter by candidate or noise.</p>
<p><small>Intervals describe stochastic match schedules at a fixed model. Cross-seed variation is in architecture_summary.csv.
Known scripted families are used; these scores do not establish performance against unknown agents.</small></p>
<input id="filter" placeholder="Filter candidates or noise"><p><a href="comparison.md">Summary</a> ·
<a href="by_strategy.csv">By strategy</a> · <a href="opponent_pairs.csv">Every pair</a> ·
<a href="training_metrics.csv">Training metrics</a> · <a href="architecture_summary.csv">Across seeds</a></p>
<table><thead><tr id="header"></tr></thead><tbody id="body"></tbody></table>
<script>const data=PAYLOAD;const keys=['candidate','noise','mean','ci_low','ci_high','matches','max_play_ms','source_bytes','observed_under_one_second'];
let order='mean',asc=false;const header=document.getElementById('header');for(const k of keys){const th=document.createElement('th');
th.textContent=k;th.onclick=()=>{asc=order===k?!asc:true;order=k;render()};header.append(th)}
function render(){const filter=document.getElementById('filter').value.toLowerCase();const rows=data.filter(r=>JSON.stringify(r).toLowerCase().includes(filter));
rows.sort((a,b)=>{let x=a[order],y=b[order];return (typeof x==='string'?x.localeCompare(y):(x??-1)-(y??-1))*(asc?1:-1)});
const body=document.getElementById('body');body.replaceChildren();for(const row of rows){const tr=document.createElement('tr');
if(row.observed_under_one_second===false)tr.className='bad';for(const k of keys){const td=document.createElement('td');
const v=row[k];td.textContent=v===null?'—':typeof v==='number'&&!Number.isInteger(v)?v.toFixed(5):String(v);tr.append(td)}body.append(tr)}}
document.getElementById('filter').oninput=render;render();</script>'''.replace("PAYLOAD", payload)
    (report / "comparison.html").write_text(page, encoding="utf-8")
