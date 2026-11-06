"""Offline CPU benchmarks using the actual standalone exported agents.

No torch or numpy dependency.
"""

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
import tracemalloc
from types import SimpleNamespace

from agent_export import load_agent
from generate_training_data import FAMILIES, choose, episode_seed, sample_length, sample_spec
from prison import PAYOFF
from experiment_logging import event, resources


def ci_stats(values):
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "se": se, "ci95_low": mean - 1.96 * se,
            "ci95_high": mean + 1.96 * se, "n": len(values)}


def play_match(agent, focal_spec, opponents, rounds, noise, seed, seat, trace=None):
    history = [[], [], []]
    policy_rngs = [random.Random(episode_seed(seed, s)) for s in range(3)]
    channel = random.Random(episode_seed(seed, 3))
    specs = {seat: focal_spec, (seat + 1) % 3: opponents[0], (seat + 2) % 3: opponents[1]}
    total = [0, 0, 0]
    cooperations = 0
    if trace is not None:
        trace.update(intended_actions=[], executed_actions=[], round_rewards=[],
                     focal_outputs=[], focal_latency_ms=[], channel_flips=[])
    for _ in range(rounds):
        actions = []
        intended = []
        for s in range(3):
            begin = time.perf_counter() if s == seat and trace is not None else None
            if s == seat and agent is not None:
                action = agent.play(history[s], history[(s + 1) % 3], history[(s + 2) % 3])
                if type(action) is not int or action not in (0, 1):
                    raise ValueError("Agent must return a Python integer 0 or 1")
            else:
                action = choose(specs[s], history[s], history[(s + 1) % 3], history[(s + 2) % 3], policy_rngs[s])
            if s == seat and trace is not None:
                trace["focal_latency_ms"].append((time.perf_counter() - begin) * 1000)
                trace["focal_outputs"].append(getattr(agent, "last_outputs", None) if agent is not None else None)
            intended.append(action)
            actions.append(action ^ int(channel.random() < noise))
        cooperations += int(actions[seat] == 0)
        for s in range(3):
            total[s] += PAYOFF[actions[s]][actions[(s + 1) % 3]][actions[(s + 2) % 3]]
            history[s].append(actions[s])
        if trace is not None:
            trace["intended_actions"].append(intended)
            trace["executed_actions"].append(actions)
            trace["channel_flips"].append([a ^ b for a, b in zip(intended, actions)])
            trace["round_rewards"].append([PAYOFF[actions[s]][actions[(s + 1) % 3]][actions[(s + 2) % 3]] for s in range(3)])
    return (total[seat] / rounds,
            total[(seat + 1) % 3] / rounds, total[(seat + 2) % 3] / rounds,
            cooperations / rounds)


def deployment_profile(path, repeats=100):
    # Account for imported weight objects and inference scratch allocations.
    tracemalloc.start()
    begin = time.perf_counter()
    cls = load_agent(path)
    load_seconds = time.perf_counter() - begin
    agent = cls()
    _, load_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rng = random.Random(4551)
    patterns = [[], [[0, 0, 0]] * 200, [[1, 1, 1]] * 200,
                [[t % 2, (t // 2) % 2, (t // 3) % 2] for t in range(200)],
                [[rng.randrange(2) for _ in range(3)] for _ in range(200)],
                [[0, 0, 0]] * 1000]
    histories = [list(zip(*p)) if p else [[], [], []] for p in patterns]
    for h in histories:
        agent.play(*h)
    times = []
    for _ in range(repeats):
        for h in histories:
            begin = time.perf_counter()
            agent.play(*h)
            times.append(time.perf_counter() - begin)
    tracemalloc.start()
    agent.play(*histories[-1])
    _, inference_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ordered = sorted(times)
    return {"source_bytes": Path(path).stat().st_size, "load_seconds": load_seconds,
            "traced_load_peak_bytes": load_peak, "traced_inference_scratch_peak_bytes": inference_peak,
            "median_play_ms": statistics.median(times) * 1000,
            "p99_play_ms": ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))] * 1000,
            "max_play_ms": max(times) * 1000,
            "observed_under_one_second": max(times) < 1,
            "timing_calls": len(times), "history_lengths": [len(h[0]) for h in histories],
            "memory_note": "tracemalloc allocations, not full process RSS; timings are hardware-specific"}


def evaluate_candidate(candidate, config, repeats, seed, noises, timing_repeats,
                       log_directory=None, trace_level="turns"):
    name = candidate["name"]
    cls = load_agent(candidate["path"]) if "path" in candidate else None
    agent = cls() if cls else None  # Reuse across matches to exercise stateless reset behavior.
    profile = deployment_profile(candidate["path"], timing_repeats) if cls else None
    config = SimpleNamespace(**config)
    logdir = Path(log_directory) if log_directory else None
    if logdir:
        logdir.mkdir(parents=True, exist_ok=True)
        event(logdir / "events.jsonl", "evaluation_start", candidate=name, resources=resources())
    pairs, summaries = [], []
    for noise_index, noise in enumerate(noises):
        repeat_totals = [0.0] * repeats
        for first_index, first in enumerate(FAMILIES):
            checkpoint = logdir / f"noise{noise_index:02d}_row{first_index:02d}.json" if logdir else None
            if checkpoint and checkpoint.exists():
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                for row in saved:
                    for i, score in enumerate(row["replicate_scores"]):
                        repeat_totals[i] += score / len(FAMILIES) ** 2
                pairs.extend(saved)
                continue
            row_results = []
            trace_stream = None
            trace_pending = logdir / f"noise{noise_index:02d}_row{first_index:02d}.matches.pending.gz" if logdir else None
            if trace_pending and trace_level != "none":
                trace_stream = gzip.open(trace_pending, "wt", encoding="utf-8", compresslevel=1)
            try:
                for second_index, second in enumerate(FAMILIES):
                    scores, cooperation, opp1, opp2 = [], [], [], []
                    for repeat in range(repeats):
                        # IDs do not depend on repeat count: expanding a schedule keeps its existing scenarios.
                        scenario = ((noise_index * len(FAMILIES) + first_index) * len(FAMILIES) + second_index) * 1000000 + repeat
                        match_seed = episode_seed(seed, scenario)
                        rng = random.Random(match_seed)
                        opponents = [sample_spec(first, rng), sample_spec(second, rng)]
                        rounds = sample_length(config, rng)
                        focal_rng = random.Random(episode_seed(match_seed, 99))
                        focal = sample_spec(candidate.get("family", "cooperate"), focal_rng)
                        trace = {} if trace_stream and trace_level == "turns" else None
                        positional = (agent, focal, opponents, rounds, noise, match_seed, repeat % 3)
                        result = play_match(*positional, trace=trace) if trace is not None else play_match(*positional)
                        scores.append(result[0]); opp1.append(result[1]); opp2.append(result[2]); cooperation.append(result[3])
                        repeat_totals[repeat] += result[0] / len(FAMILIES) ** 2
                        if trace_stream:
                            trace_stream.write(json.dumps({"candidate": name, "seed": match_seed, "repeat": repeat,
                                "focal_seat": repeat % 3, "rounds": rounds, "noise": noise,
                                "opponent_specs": opponents, "focal_spec": focal if agent is None else None,
                                "scores": list(result[:3]), "cooperation_rate": result[3], **(trace or {})}, separators=(",", ":")) + "\n")
                    row_results.append({"candidate": name, "noise": noise, "opponent1": first, "opponent2": second,
                                          **ci_stats(scores), "opponent1_score": statistics.fmean(opp1),
                                          "opponent2_score": statistics.fmean(opp2), "cooperation_rate": statistics.fmean(cooperation),
                                          "replicate_scores": scores})
            finally:
                if trace_stream:
                    trace_stream.close()
            if checkpoint:
                from experiment_data import write_json
                if trace_stream:
                    trace_pending.replace(logdir / f"noise{noise_index:02d}_row{first_index:02d}.matches.jsonl.gz")
                write_json(checkpoint, row_results)
                event(logdir / "events.jsonl", "opponent_row_complete", noise=noise, opponent=first,
                      resources=resources())
            pairs.extend(row_results)
            print(f"{name}: noise={noise:.2f}, opponent rows {first_index + 1}/{len(FAMILIES)} complete", flush=True)
        summaries.append({"candidate": name, "noise": noise, **ci_stats(repeat_totals),
                          "matches": repeats * len(FAMILIES) ** 2,
                          "replicate_means": repeat_totals})
    return {"name": name, "pairs": pairs, "summaries": summaries, "deployment": profile}


def candidate_job(job):
    if job.get("log_directory"):
        from experiment_logging import capture_console
        with capture_console(Path(job["log_directory"]) / "console.log"):
            return evaluate_candidate(**job)
    return evaluate_candidate(**job)


def write_reports(results, output):
    output = Path(output)
    results = sorted(results, key=lambda r: r["name"])
    pairs = [{k: v for k, v in row.items() if k != "replicate_scores"}
             for result in results for row in result["pairs"]]
    summary = [{k: v for k, v in row.items() if k != "replicate_means"}
               for result in results for row in result["summaries"]]
    baseline = next((r for r in results if r["name"] == "script:defect"), None)
    for row in summary:
        row.update(delta_vs_always_defect=None, delta_ci95_low=None, delta_ci95_high=None)
        if baseline:
            reference = next(s for s in baseline["summaries"] if s["noise"] == row["noise"])
            own = next(s for r in results if r["name"] == row["candidate"]
                       for s in r["summaries"] if s["noise"] == row["noise"])
            difference = ci_stats([a - b for a, b in zip(own["replicate_means"], reference["replicate_means"])])
            row.update(delta_vs_always_defect=difference["mean"],
                       delta_ci95_low=difference["ci95_low"], delta_ci95_high=difference["ci95_high"])
    grouped = []
    for result in results:
        for noise in sorted({p["noise"] for p in result["pairs"]}):
            for family in FAMILIES:
                # First opponent fixed, second uniformly mixed across all 16.
                relevant = [p for p in result["pairs"] if p["noise"] == noise and p["opponent1"] == family]
                both = next(p for p in relevant if p["opponent2"] == family)
                grouped.append({"candidate": result["name"], "noise": noise, "opponent_family": family,
                                "score_vs_two_of_family": both["mean"],
                                "score_with_family_and_mixed_other": statistics.fmean(p["mean"] for p in relevant)})
    for filename, rows in (("leaderboard.csv", summary), ("opponent_pairs.csv", pairs), ("by_strategy.csv", grouped)):
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    profiles = {r["name"]: r["deployment"] for r in results if r["deployment"] is not None}
    (output / "deployment.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    lines = ["# Exported-agent benchmark", "",
             "Absolute mean payoff per round; equal weight for all 256 ordered scripted-opponent pairs.",
             "95% intervals are normal approximations across independent complete schedule replicates.",
             "These are known-family tests with new parameter/noise seeds, not evidence of unseen-family generalization.", "",
             "| Agent | Noise | Mean score | Approx. 95% interval | Matches |", "|---|---:|---:|---|---:|"]
    for row in sorted(summary, key=lambda r: (r["noise"], -r["mean"])):
        lines.append(f"| {row['candidate']} | {row['noise']:.2f} | {row['mean']:.4f} | "
                     f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | {row['matches']} |")
    lines += ["", "Model rankings use exported pure-Python policies. See by_strategy.csv for same-family and mixed-family scores,",
              "opponent_pairs.csv for all pair results, and deployment.json for source size, timing and allocation measurements.",
              "A model that misses the one-second-per-move budget is too slow on this machine.",
              "Do not repeatedly tune on this final benchmark; reserve a fresh seed for confirmation."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
