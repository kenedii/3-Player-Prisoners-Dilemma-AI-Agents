"""Offline three-player PD episode generator. Standard library only.

Does nothing on import. Run explicitly to generate gzip JSONL shards.
Actions: C=0, D=1. Histories contain executed (post-noise) actions.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import math
import tempfile
from contextlib import contextmanager

from prison import PAYOFF


FAMILIES = (
    "cooperate", "defect", "random", "tft_any", "tft_both",
    "tft_one", "generous_tft", "contrite_tft", "grim", "two_strikes",
    "majority", "wsls", "alternator", "periodic", "switch", "memory_one",
)


def sample_spec(family, rng):
    """Explicit three-player adaptations; these are not canonical 2-player TFT."""
    spec = {"family": family}
    if family == "random":
        spec["p_defect"] = rng.choice([0.1, 0.25, 0.5, 0.75, 0.9])
    elif family == "tft_one":
        spec["opponent"] = rng.randrange(2)
    elif family == "generous_tft":
        spec["forgive"] = rng.uniform(0.05, 0.4)
    elif family == "majority":
        spec.update(window=rng.choice([3, 5, 10, 20]), threshold=rng.uniform(0.3, 0.7))
    elif family == "wsls":
        spec["aspiration"] = rng.choice([3, 5, 6])
    elif family == "periodic":
        spec.update(period=rng.randint(3, 12), phase=rng.randrange(3))
    elif family == "switch":
        spec.update(at=rng.randint(15, 90), first=rng.randrange(2))
    elif family == "memory_one":
        spec.update(initial=rng.random(), p_defect=[rng.betavariate(0.5, 0.5) for _ in range(8)])
    return spec


def choose(spec, own, first, second, rng):
    """Compute an intended action from public history, without hidden state."""
    family = spec["family"]
    t = len(own)
    if family == "cooperate":
        return 0
    if family == "defect":
        return 1
    if family == "random":
        return int(rng.random() < spec["p_defect"])
    if family == "alternator":
        return t % 2
    if family == "periodic":
        return int((t + spec["phase"]) % spec["period"] == 0)
    if family == "switch":
        return spec["first"] if t < spec["at"] else 1 - spec["first"]
    if family == "memory_one":
        p = spec["initial"] if not t else spec["p_defect"][4 * own[-1] + 2 * first[-1] + second[-1]]
        return int(rng.random() < p)
    if not t:
        return 0
    if family == "tft_any":
        return int(first[-1] == 1 or second[-1] == 1)
    if family == "tft_both":
        return int(first[-1] == 1 and second[-1] == 1)
    if family == "tft_one":
        return (first, second)[spec["opponent"]][-1]
    if family == "generous_tft":
        return int((first[-1] or second[-1]) and rng.random() >= spec["forgive"])
    if family == "contrite_tft":
        # Forgive an immediate response to our executed D after mutual cooperation.
        if t >= 2 and own[-1] == 1 and first[-2] == second[-2] == 0:
            return 0
        return int(first[-1] or second[-1])
    if family == "grim":
        return int(1 in first or 1 in second)
    if family == "two_strikes":
        return int(t >= 2 and ((first[-1] == first[-2] == 1) or (second[-1] == second[-2] == 1)))
    if family == "majority":
        n = min(t, spec["window"])
        return int((sum(first[-n:]) + sum(second[-n:])) / (2 * n) > spec["threshold"])
    if family == "wsls":
        reward = PAYOFF[own[-1]][first[-1]][second[-1]]
        return own[-1] if reward >= spec["aspiration"] else 1 - own[-1]
    raise ValueError("Unknown family: " + family)


def episode_seed(seed, episode_id):
    digest = hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def sample_length(args, rng):
    if args.fixed_rounds is not None:
        return args.fixed_rounds
    # Rejection sampling, rather than clipping, of rounded normal samples.
    while True:
        length = round(rng.gauss(args.mean_rounds, args.std_rounds))
        if args.min_rounds <= length <= args.max_rounds:
            return length


SUMMARY_FEATURES = [
    "elapsed_rounds_div_100", "self_cooperation_rate", "opp1_cooperation_rate",
    "opp2_cooperation_rate", "self_defection_streak_div_100",
    "opp1_defection_streak_div_100", "opp2_defection_streak_div_100",
    "opponents_both_cooperate_rate", "opponents_both_defect_rate",
    "self_mean_payoff_div_8",
]


def encode_input(past, window):
    """Deployment-compatible features: ONLY executed rounds before this turn.

    Layout: oldest-to-newest flattened triples, then one mask per round,
    then the ten summary features. Padding is on the left.
    """
    t = len(past)
    recent = past[-window:]
    missing = window - len(recent)
    vector = [0] * (3 * missing) + [a for row in recent for a in row]
    vector += [0] * missing + [1] * len(recent)
    rates, streaks = [], []
    for seat in range(3):
        rates.append(sum(row[seat] == 0 for row in past) / t if t else 0.0)
        streak = 0
        for row in reversed(past):
            if row[seat] != 1:
                break
            streak += 1
        streaks.append(streak / 100)
    both_c = sum(row[1] == row[2] == 0 for row in past) / t if t else 0.0
    both_d = sum(row[1] == row[2] == 1 for row in past) / t if t else 0.0
    payoff = sum(PAYOFF[row[0]][row[1]][row[2]] for row in past) / (8 * t) if t else 0.0
    return vector + [t / 100] + rates + streaks + [both_c, both_d, payoff]


def surviving_length(args, elapsed, rng):
    """Sample stopping conditional on the match still being alive, not its saved end."""
    if args.fixed_rounds is not None:
        return args.fixed_rounds
    while True:
        length = sample_length(args, rng)
        if length > elapsed:
            return length


def rollout_targets(args, past, opponent_specs, seed):
    """Privileged Monte Carlo teacher, fixed continuation; NOT optimal Q*.

    Re-simulate BOTH initial actions. A common per-step random seed pairs noise
    and stochastic opponent decisions across branches. Opponents subsequently
    react to each branch's own history. Hidden specs are teacher-only metadata.
    """
    values = [[], []]
    elapsed = len(past)
    for draw in range(args.rollouts):
        draw_seed = episode_seed(seed, draw)
        setup_rng = random.Random(draw_seed)
        length = surviving_length(args, elapsed, setup_rng)
        steps = min(args.rollout_horizon, length - elapsed)
        continuation = sample_spec(args.continuation, setup_rng)
        step_seeds = [setup_rng.getrandbits(64) for _ in range(steps)]
        for initial in (0, 1):
            history = [[row[s] for row in past] for s in range(3)]
            total = weight_total = 0.0
            weight = 1.0
            for step, step_seed in enumerate(step_seeds):
                streams = [random.Random(episode_seed(step_seed, s)) for s in range(4)]
                own = initial if step == 0 else choose(continuation, *history, streams[0])
                actions = [own,
                           choose(opponent_specs[0], history[1], history[2], history[0], streams[1]),
                           choose(opponent_specs[1], history[2], history[0], history[1], streams[2])]
                noisy = [a ^ int(streams[3].random() < args.noise) for a in actions]
                total += weight * PAYOFF[noisy[0]][noisy[1]][noisy[2]]
                weight_total += weight
                weight *= args.discount
                for s in range(3):
                    history[s].append(noisy[s])
            values[initial].append(total / weight_total)
    means = [sum(v) / len(v) for v in values]
    def stderr(samples):
        n = len(samples)
        mean = sum(samples) / n
        return math.sqrt(sum((v - mean) ** 2 for v in samples) / (n * (n - 1)))
    gaps = [d - c for c, d in zip(*values)]
    return means, [stderr(v) for v in values], stderr(gaps)


def make_examples(args, record):
    """Sample turns uniformly without replacement; keep all samples in episode split."""
    rng = random.Random(episode_seed(record["seed"], 0x123456))
    rounds = record["executed_actions"]
    turns = sorted(rng.sample(range(len(rounds)), min(args.examples_per_episode, len(rounds))))
    examples = []
    specs = record["private_metadata"]["strategies"]
    for t in turns:
        q, se, gap_se = rollout_targets(args, rounds[:t], specs[1:], episode_seed(record["seed"], t))
        joint_class = 2 * rounds[t][1] + rounds[t][2]
        one_hot = [int(j == joint_class) for j in range(4)]
        examples.append({"turn": t, "x": encode_input(rounds[:t], args.window),
                         "y": q + one_hot, "q_standard_error": se,
                         "defect_minus_cooperate_standard_error": gap_se})
    return examples


def simulate(args, episode_id, families):
    seed = episode_seed(args.seed, episode_id)
    rng = random.Random(seed)
    specs = [sample_spec(rng.choice(families), rng) for _ in range(3)]
    # Separate action RNG per seat and independent channel RNG.
    policy_rngs = [random.Random(rng.getrandbits(64)) for _ in range(3)]
    channel_rng = random.Random(rng.getrandbits(64))
    length = sample_length(args, rng)
    exploration = rng.choice(args.exploration)
    history = [[], [], []]
    intended, executed, rewards, focal_probabilities = [], [], [], []
    for _ in range(length):
        actions = []
        for seat in range(3):
            prng = policy_rngs[seat]
            base = choose(specs[seat], history[seat], history[(seat + 1) % 3], history[(seat + 2) % 3], prng)
            action = base
            if seat == 0:
                # Epsilon exploration supplies both focal actions in diverse histories.
                if prng.random() < exploration:
                    action = prng.randrange(2)
                # Conditional on sampled base; not a marginalized propensity for stochastic families.
                focal_probabilities.append((1 - exploration) * int(action == base) + exploration / 2)
            actions.append(action)
        noisy = [a ^ int(channel_rng.random() < args.noise) for a in actions]
        score = [PAYOFF[noisy[s]][noisy[(s + 1) % 3]][noisy[(s + 2) % 3]] for s in range(3)]
        intended.append(actions)
        executed.append(noisy)
        rewards.append(score)
        for seat in range(3):
            history[seat].append(noisy[seat])
    return {
        "episode_id": episode_id, "seed": seed,
        "private_metadata": {"strategies": specs, "sampled_rounds": length, "focal_epsilon": exploration},
        "intended_actions": intended, "executed_actions": executed,
        "rewards": rewards, "focal_action_probability_given_sampled_base": focal_probabilities,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("training_data"))
    parser.add_argument("--episodes", type=int, default=None,
                        help="Optional limit for this run; default generates continuously until Ctrl+C.")
    parser.add_argument("--start-id", type=int, default=None, help="Optional initial ID for a NEW dataset; existing datasets continue automatically.")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--shard-episodes", type=int, default=1000)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--fixed-rounds", type=int)
    parser.add_argument("--mean-rounds", type=float, default=100)
    parser.add_argument("--std-rounds", type=float, default=10)
    parser.add_argument("--min-rounds", type=int, default=80)
    parser.add_argument("--max-rounds", type=int, default=200)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--test-only-families", nargs="*", choices=FAMILIES, default=[])
    parser.add_argument("--exploration", type=float, nargs="+", default=[0.0, 0.1, 0.3, 1.0])
    parser.add_argument("--gzip-level", type=int, choices=range(1, 10), default=1)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--examples-per-episode", type=int, default=8,
                        help="Sampled turns receiving MLP features and rollout labels; 0 disables.")
    parser.add_argument("--rollouts", type=int, default=16, help="Paired Monte Carlo draws per labeled turn.")
    parser.add_argument("--rollout-horizon", type=int, default=20)
    parser.add_argument("--continuation", choices=FAMILIES, default="tft_both",
                        help="Focal policy AFTER the forced initial action in each teacher branch.")
    parser.add_argument("--discount", type=float, default=1.0)
    args = parser.parse_args()
    if (args.episodes is not None and args.episodes <= 0) or args.shard_episodes <= 0 or (args.start_id is not None and args.start_id < 0):
        parser.error("episodes and shard-episodes must be positive; start-id must be nonnegative")
    if not 0 <= args.noise <= 1 or any(not 0 <= e <= 1 for e in args.exploration):
        parser.error("noise and exploration probabilities must be in [0,1]")
    if args.fixed_rounds is not None and args.fixed_rounds <= 0:
        parser.error("fixed-rounds must be positive")
    if args.std_rounds <= 0 or not 1 <= args.min_rounds <= args.max_rounds:
        parser.error("require positive std-rounds and 1 <= min-rounds <= max-rounds")
    if not set(args.families) - set(args.test_only_families):
        parser.error("At least one family must be available for training")
    if args.window <= 0 or args.examples_per_episode < 0 or args.rollouts < 2 or args.rollout_horizon <= 0:
        parser.error("require window>0, examples-per-episode>=0, rollouts>=2, rollout-horizon>0")
    if not 0 < args.discount <= 1:
        parser.error("discount must be in (0,1]")
    return args


def save_manifest(output, manifest):
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(output / "manifest.json")


@contextmanager
def dataset_lock(output):
    """OS lock releases on process exit, including crashes; lock file may remain."""
    import sys
    with (output / ".writer.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another generator is writing this dataset. Use a separate output directory.") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Another generator is writing this dataset. Use a separate output directory.") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def new_manifest(args):
    manifest = {"schema_version": 3, "config": {**vars(args), "output": str(args.output)},
                "payoff": PAYOFF, "split_counts": {}, "shards": [],
                "length_assumption": "std=10, bounds=[80,200] are configurable assumptions",
                "complete": True, "next_episode_id": args.start_id or 0}
    manifest["mlp_schema"] = {
        "input_dimension": 4 * args.window + 10,
        "input_layout": "3*window flattened executed actions; window masks; 10 summaries",
        "summary_features": SUMMARY_FEATURES,
        "output_order": ["Q_cooperate", "Q_defect", "P_CC", "P_CD", "P_DC", "P_DD"],
        "prediction_target": "Observed post-noise joint opponent outcome, one-hot; train softmax cross entropy",
        "q_target": "Expected normalized discounted future payoff over truncated rollout under fixed continuation",
        "teacher_information": "Privileged exact simulated opponent specs; not optimal control or an inferred type posterior",
        "stopping": "Fresh total length conditional on surviving to sampled turn; never use saved episode end",
        "q_range": [0, 8],
    }
    manifest["example_counts"] = {}
    return manifest


def apply_commit(manifest, commit):
    if commit["start_id"] != manifest["next_episode_id"]:
        raise RuntimeError("Non-contiguous batch IDs; dataset needs inspection.")
    manifest["shards"].extend(commit["shards"])
    for field in ("split_counts", "example_counts"):
        for split, count in commit[field].items():
            manifest[field][split] = manifest[field].get(split, 0) + count
    manifest["next_episode_id"] = commit["end_id"]


def open_dataset(args):
    path = args.output / "manifest.json"
    if not path.exists():
        if any(args.output.glob("*.jsonl.gz")) or any(args.output.glob("batch-*")):
            raise RuntimeError("Dataset files exist without a manifest; refusing to overwrite.")
        manifest = new_manifest(args)
        save_manifest(args.output, manifest)
        return manifest
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in (2, 3):
        raise RuntimeError("Unsupported dataset schema. Use a new output directory.")
    ignored = {"output", "episodes", "start_id", "shard_episodes", "gzip_level"}
    mismatches = [k for k, v in vars(args).items()
                  if k not in ignored and manifest["config"].get(k) != v]
    if mismatches:
        raise RuntimeError("Settings differ from existing dataset: " + ", ".join(mismatches)
                           + ". Reuse its settings or choose a new --output directory.")
    if manifest["payoff"] != PAYOFF:
        raise RuntimeError("Payoff table differs from existing dataset.")
    if "next_episode_id" not in manifest:
        # One-time migration of datasets generated by the previous script.
        if not manifest.get("complete"):
            raise RuntimeError("Legacy dataset is incomplete; use a new output directory.")
        highest = manifest["config"].get("start_id") or 0
        for shard in manifest["shards"]:
            with gzip.open(args.output / shard["file"], "rt", encoding="utf-8") as stream:
                for line in stream:
                    highest = max(highest, json.loads(line)["episode_id"] + 1)
        manifest["next_episode_id"] = highest
        manifest["schema_version"] = 3
    # Recover committed batches if a process exited between directory rename and manifest update.
    for folder in sorted(args.output.glob("batch-*")):
        commit = json.loads((folder / "commit.json").read_text(encoding="utf-8"))
        if commit["end_id"] <= manifest["next_episode_id"]:
            continue
        apply_commit(manifest, commit)
    if args.start_id is not None and args.start_id != manifest["next_episode_id"]:
        raise RuntimeError("For an existing dataset omit --start-id so IDs continue automatically.")
    manifest["complete"] = True
    save_manifest(args.output, manifest)
    return manifest


def append_batch(args, manifest, start_id, end_id):
    """Publish closed gzip shards together via a directory rename, then checkpoint."""
    stage = Path(tempfile.mkdtemp(prefix=".pending-", dir=args.output))
    folder_name = f"batch-{start_id:020d}-{end_id:020d}"
    commit = {"start_id": start_id, "end_id": end_id,
              "shards": [], "split_counts": {}, "example_counts": {}}
    train_families = [f for f in args.families if f not in args.test_only_families]
    handles, shard_counts = {}, {}
    interrupted = False
    try:
        for episode_id in range(start_id, end_id):
            split_rng = random.Random(episode_seed(args.seed ^ 0xABCDEF, episode_id))
            u = split_rng.random()
            split = "train" if u < 0.8 else "validation" if u < 0.9 else "test"
            families = train_families if split != "test" else list(dict.fromkeys(args.families + args.test_only_families))
            if split not in handles:
                filename = f"{split}.jsonl.gz"
                handles[split] = gzip.open(stage / filename, "xt", encoding="utf-8", compresslevel=args.gzip_level)
                shard_counts[split] = {"file": f"{folder_name}/{filename}", "split": split, "episodes": 0}
                commit["shards"].append(shard_counts[split])
            record = simulate(args, episode_id, families)
            record["mlp_examples"] = make_examples(args, record)
            commit["example_counts"][split] = commit["example_counts"].get(split, 0) + len(record["mlp_examples"])
            handles[split].write(json.dumps(record, separators=(",", ":")) + "\n")
            commit["split_counts"][split] = commit["split_counts"].get(split, 0) + 1
            shard_counts[split]["episodes"] += 1
    except KeyboardInterrupt:
        interrupted = True
        commit["end_id"] = start_id + sum(commit["split_counts"].values())
        if commit["end_id"] == start_id:
            raise
    finally:
        for handle in handles.values():
            handle.close()
    if interrupted:
        folder_name = f"batch-{start_id:020d}-{commit['end_id']:020d}"
        for shard in commit["shards"]:
            shard["file"] = f"{folder_name}/{Path(shard['file']).name}"
    (stage / "commit.json").write_text(json.dumps(commit, indent=2), encoding="utf-8")
    stage.rename(args.output / folder_name)
    apply_commit(manifest, commit)
    save_manifest(args.output, manifest)
    return interrupted


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with dataset_lock(args.output):
        manifest = open_dataset(args)
        start = manifest["next_episode_id"]
        stop = start + args.episodes if args.episodes is not None else None
        amount = f"{args.episodes} episodes" if stop is not None else "episodes continuously (Ctrl+C to stop)"
        print(f"Adding {amount} to {args.output}; first ID {start}", flush=True)
        try:
            batch_start = start
            while stop is None or batch_start < stop:
                end = batch_start + args.shard_episodes
                if stop is not None:
                    end = min(end, stop)
                interrupted = append_batch(args, manifest, batch_start, end)
                end = manifest["next_episode_id"]
                progress = str(end - start) if stop is None else f"{end - start}/{args.episodes}"
                print(f"Saved {progress} new episodes; "
                      f"dataset total {sum(manifest['split_counts'].values())}", flush=True)
                batch_start = end
                if interrupted:
                    print("Stopped. Completed episodes are saved; rerun to keep adding data.", flush=True)
                    break
        except KeyboardInterrupt:
            print("Stopped. Completed batches are saved; rerun to add more episodes.", flush=True)


if __name__ == "__main__":
    main()
