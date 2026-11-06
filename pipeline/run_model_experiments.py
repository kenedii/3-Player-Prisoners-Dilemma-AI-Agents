"""Server experiment entry point: prepare, train, evaluate, or all.

Run --help for options. Importing this module does not start training or games.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import random
import time
import hashlib
import traceback
from datetime import datetime, timezone
import shutil

import numpy as np

from agent_export import export_source, load_agent, parameter_count
from experiment_data import batches, load_cache, prepare_cache, write_json
from experiment_catalog import ARCHITECTURES, NEURAL, CLASSICAL, STANDARD, describe_suite
from experiment_logging import event, capture_console, environment, resources


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("all", "prepare", "train", "evaluate"), default="all")
    parser.add_argument("--suite", choices=("standard", "large"), default="standard")
    parser.add_argument("--data", type=Path, default=Path("training_data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", nargs="+", choices=ARCHITECTURES)
    parser.add_argument("--max-train", type=int, default=2000000, help="Labeled turns; 0 uses all available train examples.")
    parser.add_argument("--max-validation", type=int, default=200000)
    parser.add_argument("--max-test", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--train-seeds", type=int, nargs="+")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gap-weight", type=float, default=4.0)
    parser.add_argument("--aux-weight", type=float, default=0.05)
    parser.add_argument("--device", default="cuda", help="cuda by default; explicit cpu is available for server debugging.")
    parser.add_argument("--eval-repeats", type=int, help="Independent matches per ordered opponent-family pair.")
    parser.add_argument("--eval-seed", type=int, default=9864321)
    parser.add_argument("--eval-noise", type=float, nargs="+")
    parser.add_argument("--eval-workers", type=int, default=1, help="CPU processes; affects exported-agent benchmarks only.")
    parser.add_argument("--timing-repeats", type=int, default=100)
    parser.add_argument("--skip-baselines", action="store_true", help="Quick debugging only; omits the 16 scripted focal agents.")
    parser.add_argument("--classical-max-train", type=int, default=250000,
                        help="Base examples per CPU model before symmetry doubling; 0 uses full cache.")
    parser.add_argument("--classical-jobs", type=int, default=4)
    parser.add_argument("--batch-log-every", type=int, default=1)
    parser.add_argument("--trace-level", choices=("turns", "matches", "none"), default="turns")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Write plan and environment only; no training or games.")
    args = parser.parse_args()
    args.models = args.models or list(ARCHITECTURES if args.suite == "large" else STANDARD)
    args.train_seeds = args.train_seeds or ([1234, 2345, 3456] if args.suite == "large" else [1234])
    args.eval_repeats = args.eval_repeats if args.eval_repeats is not None else (20 if args.suite == "large" else 10)
    args.eval_noise = args.eval_noise or ([0.0, .02, .05] if args.suite == "large" else [.02])
    args.output = args.output or Path("large_experiments" if args.suite == "large" else "model_experiments")
    if any(getattr(args, key) <= 0 for key in ("epochs", "patience", "batch_size", "eval_repeats", "eval_workers", "timing_repeats")):
        parser.error("Epoch/batch/repeat/worker counts must be positive")
    if any(getattr(args, key) < 0 for key in ("max_train", "max_validation", "max_test", "gap_weight", "aux_weight", "classical_max_train")):
        parser.error("Sample caps and loss weights cannot be negative")
    if args.learning_rate <= 0 or any(not 0 <= n <= 1 for n in args.eval_noise):
        parser.error("Require positive learning rate and noise probabilities in [0,1]")
    if args.classical_jobs <= 0 or args.batch_log_every <= 0 or args.eval_repeats >= 1000000:
        parser.error("Positive CPU jobs/log interval and fewer than one million repeats required")
    if len(set(args.models)) != len(args.models) or len(set(args.train_seeds)) != len(args.train_seeds) or len(set(args.eval_noise)) != len(args.eval_noise):
        parser.error("Models, seeds and noise settings must not contain duplicates")
    if any(not 0 <= seed < 2**32 for seed in args.train_seeds):
        parser.error("Training seeds must be unsigned 32-bit integers")
    return args


def tensor_batch(block, dimension, device):
    import torch
    tensor = torch.from_numpy(block).to(device)
    x = tensor[:, :dimension]
    targets = tensor[:, dimension:dimension + 6]
    return x, targets[:, :2] / 8, targets[:, 2:].argmax(dim=1), tensor[:, dimension + 6]


def loss_terms(prediction, q, joint, args):
    import torch.nn.functional as F
    value = F.mse_loss(prediction[:, :2], q)
    gap = F.mse_loss(prediction[:, 1] - prediction[:, 0], q[:, 1] - q[:, 0])
    auxiliary = F.cross_entropy(prediction[:, 2:], joint)
    return value + args.gap_weight * gap + args.aux_weight * auxiliary


def offline_metrics(model, array, dimension, args, device):
    import torch
    model.eval()
    totals = {"loss": 0.0, "value_sq": 0.0, "gap_sq": 0.0, "agreement": 0.0,
              "teacher_regret": 0.0, "joint_nll": 0.0, "joint_accuracy": 0.0,
              "confident_correct": 0.0, "confident_count": 0.0}
    count = 0
    with torch.inference_mode():
        for block in batches(array, args.batch_size):
            x, q, joint, se = tensor_batch(block, dimension, device)
            prediction = model(x)
            n = len(block)
            gap = 8 * (q[:, 1] - q[:, 0])
            predicted_gap = 8 * (prediction[:, 1] - prediction[:, 0])
            action = (predicted_gap > 0).long()
            optimal = (gap > 0).long()
            correct = action == optimal
            confident = gap.abs() > 1.96 * se
            values = {
                "loss": loss_terms(prediction, q, joint, args) * n,
                "value_sq": ((8 * (prediction[:, :2] - q)) ** 2).sum() / 2,
                "gap_sq": ((predicted_gap - gap) ** 2).sum(),
                "agreement": correct.sum(),
                "teacher_regret": (q.max(1).values - q.gather(1, action[:, None]).squeeze(1)).sum() * 8,
                "joint_nll": torch.nn.functional.cross_entropy(prediction[:, 2:], joint, reduction="sum"),
                "joint_accuracy": (prediction[:, 2:].argmax(1) == joint).sum(),
                "confident_correct": (correct & confident).sum(), "confident_count": confident.sum(),
            }
            for key, value in values.items():
                totals[key] += float(value.item())
            count += n
    return {"examples": count, "loss": totals["loss"] / count,
            "q_rmse_payoff_units": math.sqrt(totals["value_sq"] / count),
            "gap_rmse_payoff_units": math.sqrt(totals["gap_sq"] / count),
            "teacher_action_agreement": totals["agreement"] / count,
            "teacher_regret_payoff_units": totals["teacher_regret"] / count,
            "joint_nll": totals["joint_nll"] / count,
            "joint_accuracy": totals["joint_accuracy"] / count,
            "confident_examples": int(totals["confident_count"]),
            "confident_action_agreement": (totals["confident_correct"] / totals["confident_count"]
                                           if totals["confident_count"] else None)}


def export_parity(model, path, arrays, meta):
    """Executed on server: PyTorch versus exported source, plus public-history encoder."""
    import torch
    cls = load_agent(path)
    agent = cls()
    d = meta["dimension"]
    array = arrays["validation"]
    rng = np.random.default_rng(8787)
    indices = rng.integers(0, len(array), size=min(64, len(array)))
    features = np.array(array[indices, :d], copy=True)
    from generate_training_data import encode_input
    public_rng = random.Random(989)
    histories = [[[public_rng.randrange(2) for _ in range(3)] for _ in range(n)]
                 for n in (0, 1, 7, 50, 199, 1000)]
    generated = np.asarray([encode_input(h, meta["window"]) for h in histories], dtype=np.float32)
    features = np.concatenate((features, generated))
    model = model.cpu().eval()
    with torch.inference_mode():
        expected = model(torch.from_numpy(features)).numpy()
    actual = np.asarray([agent.predict(x.tolist()) for x in features])
    max_error = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, atol=2e-5, rtol=2e-5):
        raise AssertionError(f"Export parity failed: maximum error {max_error}")
    expected_actions = expected[:, 1] > expected[:, 0]
    actual_actions = actual[:, 1] > actual[:, 0]
    ambiguous = np.abs(expected[:, 1] - expected[:, 0]) < 4e-5
    if np.any((expected_actions != actual_actions) & ~ambiguous):
        raise AssertionError("Export changes non-tied action decisions")
    for h in histories:
        own, first, second = ([list(v) for v in zip(*h)] if h else [[], [], []])
        copies = [own[:], first[:], second[:]]
        a = agent.play(own, first, second)
        assert a == agent.play(own, second, first)
        assert [own, first, second] == copies
    return {"max_output_absolute_error": max_error, "compared_inputs": len(features),
            "near_tied_inputs": int(ambiguous.sum()), "passed": True}


def train_neural_models(args):
    try:
        import torch
        from model_training import SmallModel
    except ImportError as exc:
        raise SystemExit("Install server dependencies first; see requirements.txt") from exc
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Install a GPU-compatible PyTorch build; not silently falling back to CPU.")
    torch.set_num_threads(4)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}; torch={torch.__version__}; CUDA={torch.version.cuda}", flush=True)
        # Fail early on unsupported GPU architecture instead of hours into parsing/training.
        torch.ones((2, 2), device=device).square().sum().item()
    meta, arrays = load_cache(args.output / "cache")
    settings = {k: getattr(args, k) for k in ("epochs", "patience", "batch_size", "learning_rate", "gap_weight", "aux_weight")}
    index = []
    for name in args.models:
        for seed in args.train_seeds:
            tag = f"{name}_seed{seed}"
            directory = args.output / "models" / tag
            directory.mkdir(parents=True, exist_ok=True)
            stats_path = directory / "training.json"
            events = directory / "events.jsonl"
            export_path = directory / f"{tag}.py"
            if stats_path.exists():
                saved = json.loads(stats_path.read_text(encoding="utf-8"))
                if saved["settings"] != settings or saved["cache_identity"] != meta["identity"]:
                    raise ValueError(f"Existing run {tag} has different settings; use a new --output directory.")
                if not saved.get("export_parity", {}).get("passed") or not export_path.exists():
                    raise ValueError(f"Incomplete export for {tag}; use a new output directory.")
                print(f"Skipping finished model {tag}", flush=True)
                index.append({"name": tag, "path": str(export_path.resolve())})
                continue
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            model = SmallModel(name, meta["window"]).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
            best, stale, history = math.inf, 0, []
            first_epoch = 0
            latest_path = directory / "latest.pt"
            if latest_path.exists():
                latest = torch.load(latest_path, map_location=device, weights_only=True)
                if latest["settings"] != settings or latest["cache_identity"] != meta["identity"]:
                    raise ValueError("Interrupted checkpoint has incompatible settings")
                model.load_state_dict(latest["model"])
                optimizer.load_state_dict(latest["optimizer"])
                first_epoch, best, stale, history = latest["next_epoch"], latest["best"], latest["stale"], latest["history"]
                event(events, "resume_epoch", next_epoch=first_epoch + 1)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            event(events, "neural_training_start", architecture=name, seed=seed,
                  train_examples=len(arrays["train"]), resources=resources())
            begin = time.perf_counter()
            for epoch in range(first_epoch, args.epochs):
                if stale >= args.patience:
                    break
                model.train()
                running, rows = 0.0, 0
                epoch_begin = time.perf_counter()
                for batch_index, block in enumerate(batches(arrays["train"], args.batch_size, seed=seed + epoch)):
                    batch_begin = time.perf_counter()
                    x, q, joint, _ = tensor_batch(block, meta["dimension"], device)
                    optimizer.zero_grad(set_to_none=True)
                    predictions = model(x)
                    loss = loss_terms(predictions, q, joint, args)
                    if not torch.isfinite(loss).item():
                        raise RuntimeError("Nonfinite training loss")
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    running += loss.item() * len(block)
                    rows += len(block)
                    if batch_index % args.batch_log_every == 0:
                        with torch.no_grad():
                            weight_norm = math.sqrt(sum(float(p.square().sum().item()) for p in model.parameters()))
                        event(events, "training_batch", epoch=epoch + 1, batch=batch_index, examples=len(block),
                              rows_in_epoch=rows, loss=float(loss.item()), gradient_norm=float(gradient_norm.item()),
                              q_mse_normalized=float(torch.nn.functional.mse_loss(predictions[:, :2], q).item()),
                              gap_mse_normalized=float(torch.nn.functional.mse_loss(predictions[:, 1] - predictions[:, 0], q[:, 1] - q[:, 0]).item()),
                              joint_cross_entropy=float(torch.nn.functional.cross_entropy(predictions[:, 2:], joint).item()),
                              predicted_defection_rate=float((predictions[:, 1] > predictions[:, 0]).float().mean().item()),
                              target_gap_mean_payoff=float((8 * (q[:, 1] - q[:, 0])).mean().item()),
                              predicted_gap_mean_payoff=float((8 * (predictions[:, 1] - predictions[:, 0])).mean().item()),
                              weight_l2_norm=weight_norm, learning_rate=optimizer.param_groups[0]["lr"],
                              seconds=time.perf_counter() - batch_begin,
                              cuda_allocated_bytes=torch.cuda.memory_allocated(device) if device.type == "cuda" else 0,
                              resources=resources())
                validation = offline_metrics(model, arrays["validation"], meta["dimension"], args, device)
                history.append({"epoch": epoch + 1, "train_loss": running / rows, "validation": validation,
                                "epoch_seconds": time.perf_counter() - epoch_begin})
                event(events, "epoch_complete", **history[-1])
                write_json(directory / "epochs.json", history)
                print(f"{tag} epoch {epoch + 1}: train={running / rows:.6f}, val={validation['loss']:.6f}, "
                      f"teacher regret={validation['teacher_regret_payoff_units']:.5f}", flush=True)
                if validation["loss"] < best - 1e-7:
                    best, stale = validation["loss"], 0
                    temporary = directory / "best.pt.tmp"
                    torch.save({"model": model.state_dict(), "epoch": epoch + 1}, temporary)
                    temporary.replace(directory / "best.pt")
                else:
                    stale += 1
                latest_temporary = directory / "latest.pt.tmp"
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "next_epoch": epoch + 1, "best": best, "stale": stale, "history": history,
                            "settings": settings, "cache_identity": meta["identity"]}, latest_temporary)
                latest_temporary.replace(latest_path)
            checkpoint = torch.load(directory / "best.pt", map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            validation = offline_metrics(model, arrays["validation"], meta["dimension"], args, device)
            test = offline_metrics(model, arrays["test"], meta["dimension"], args, device)
            parameters = model.export_parameters()
            export_path.write_text(export_source(parameters, meta["window"], tag), encoding="utf-8")
            parity = export_parity(model, export_path, arrays, meta)
            result = {"architecture": name, "seed": seed, "parameters": parameter_count(parameters),
                      "settings": settings, "cache_identity": meta["identity"],
                      "best_epoch": checkpoint["epoch"], "validation": validation, "test": test,
                      "export_parity": parity, "training_seconds": time.perf_counter() - begin,
                      "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                      "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
            result.update(training_examples=len(arrays["train"]),
                          cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                          export_source_bytes=export_path.stat().st_size)
            write_json(stats_path, result)
            event(events, "model_complete", result=result, resources=resources())
            index.append({"name": tag, "path": str(export_path.resolve())})
            write_json(args.output / "agent_index.json", index)
            del model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_json(args.output / "agent_index.json", index)


def train_classical_model(args, name, seed, meta, arrays):
    import joblib
    import classical_training as classical
    tag = f"{name}_seed{seed}"
    directory = args.output / "models" / tag
    directory.mkdir(parents=True, exist_ok=True)
    settings = {"classical_max_train": args.classical_max_train, "gap_weight": args.gap_weight,
                "aux_weight": args.aux_weight, "spec": CLASSICAL[name]}
    stats_path = directory / "training.json"
    export_path = directory / f"{tag}.py"
    if stats_path.exists():
        saved = json.loads(stats_path.read_text(encoding="utf-8"))
        if saved["settings"] != settings or saved["cache_identity"] != meta["identity"]:
            raise ValueError("Existing classical model has incompatible settings")
        if not saved["export_parity"]["passed"] or not export_path.exists():
            raise ValueError("Existing classical model export incomplete")
        return {"name": tag, "path": str(export_path.resolve())}
    estimator, parameters, fitted_info = classical.fit_classical(args, name, seed, arrays["train"], meta, directory)
    joblib.dump(estimator, directory / "fitted_model.joblib", compress=3)
    export_path.write_text(export_source(parameters, meta["window"], tag), encoding="utf-8")
    agent = load_agent(export_path)()
    rng = np.random.default_rng(45)
    selected = rng.choice(len(arrays["validation"]), min(128, len(arrays["validation"])), replace=False)
    x = np.array(arrays["validation"][selected, :meta["dimension"]], copy=True)
    expected = classical.predict(estimator, x, meta["window"], parameters["gap_scale"], parameters["aux_scale"])
    actual = np.asarray([agent.predict(row.tolist()) for row in x])
    max_error = float(np.max(np.abs(expected - actual)))
    if not np.allclose(actual, expected, atol=5e-5, rtol=5e-5):
        raise AssertionError(f"Classical export parity failed; maximum error {max_error}")
    result = {"architecture": name, "seed": seed, "settings": settings,
              "cache_identity": meta["identity"], "parameters": parameter_count(parameters),
              "complexity_note": "tree parameter field counts stored numeric fields, not neural trainable weights",
              "fit": fitted_info, "export_source_bytes": export_path.stat().st_size,
              "export_parity": {"passed": True, "max_output_absolute_error": max_error},
              "validation": classical.offline_metrics(estimator, arrays["validation"], meta, args, parameters),
              "test": classical.offline_metrics(estimator, arrays["test"], meta, args, parameters)}
    write_json(stats_path, result)
    event(directory / "events.jsonl", "model_complete", result=result, resources=resources())
    return {"name": tag, "path": str(export_path.resolve())}


def train_models(args):
    from experiment_logging import redact
    meta, arrays = load_cache(args.output / "cache")
    index, failures = [], []
    for name in args.models:
        for seed in args.train_seeds:
            tag = f"{name}_seed{seed}"
            directory = args.output / "models" / tag
            directory.mkdir(parents=True, exist_ok=True)
            try:
                with capture_console(directory / "console.log"):
                    if name in CLASSICAL:
                        entry = train_classical_model(args, name, seed, meta, arrays)
                    else:
                        local = argparse.Namespace(**{**vars(args), "models": [name], "train_seeds": [seed]})
                        train_neural_models(local)
                        entry = json.loads((args.output / "agent_index.json").read_text())[0]
                index.append(entry)
            except Exception as exc:
                failure = {"model": tag, "error_type": type(exc).__name__,
                           "message": redact(str(exc)), "traceback": redact(traceback.format_exc())}
                failures.append(failure)
                write_json(directory / "failure.json", failure)
                event(args.output / "events.jsonl", "training_failure", **failure)
                print(f"Failed {tag}: {failure['message']}", flush=True)
                if not args.continue_on_error:
                    raise
            finally:
                # A neural helper may have written a one-model index; restore the complete index.
                write_json(args.output / "agent_index.json", index)
                write_json(args.output / "training_failures.json", failures)
    if not index:
        raise RuntimeError("All model fits failed. Inspect model failure.json files; no completion notification sent.")


def evaluate_models(args):
    from benchmark_agents import candidate_job, write_reports
    from generate_training_data import FAMILIES
    meta, _ = load_cache(args.output / "cache")
    index = json.loads((args.output / "agent_index.json").read_text(encoding="utf-8"))
    for entry in index:
        entry["source_sha256"] = hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
    if not args.skip_baselines:
        index += [{"name": f"script:{f}", "family": f} for f in FAMILIES]
    report = args.output / f"evaluation_seed{args.eval_seed}"
    report.mkdir(parents=True, exist_ok=True)
    settings = {"repeats": args.eval_repeats, "seed": args.eval_seed, "noises": args.eval_noise,
                "timing_repeats": args.timing_repeats, "config": meta["source_config"]}
    comparison_settings = {**settings, "candidates": index}
    comparison_settings.update(trace_level=args.trace_level, benchmark_version=2)
    settings_path = report / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != comparison_settings:
        raise ValueError("Existing evaluation has different settings; choose a different --eval-seed.")
    write_json(settings_path, comparison_settings)
    results, jobs, failures = [], [], []
    for i, candidate in enumerate(index):
        saved = report / f"candidate_{i:03d}.json"
        if saved.exists():
            results.append(json.loads(saved.read_text(encoding="utf-8")))
        else:
            jobs.append((i, {"candidate": candidate, **settings,
                             "log_directory": str((report / f"candidate_{i:03d}_logs").resolve()),
                             "trace_level": args.trace_level}))
    print(f"CPU benchmark: {len(index)} candidates, {len(FAMILIES)**2} ordered opponent pairs, "
          f"{args.eval_repeats} repetitions, {len(args.eval_noise)} noise setting(s). "
          "Pure-Python GRU inference may make this stage much slower than GPU training.", flush=True)
    if args.eval_workers == 1:
        for i, job in jobs:
            print(f"Evaluating {job['candidate']['name']}", flush=True)
            try:
                result = candidate_job(job)
                write_json(report / f"candidate_{i:03d}.json", result)
                results.append(result)
            except Exception as exc:
                failures.append({"model": job["candidate"]["name"], "error_type": type(exc).__name__, "message": str(exc)})
                event(report / "events.jsonl", "evaluation_failure", **failures[-1], traceback=traceback.format_exc())
                if not args.continue_on_error:
                    raise
    else:
        with ProcessPoolExecutor(max_workers=args.eval_workers) as pool:
            futures = {pool.submit(candidate_job, job): i for i, job in jobs}
            names = {i: job["candidate"]["name"] for i, job in jobs}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    result = future.result()
                    write_json(report / f"candidate_{i:03d}.json", result)
                    results.append(result)
                    print(f"Finished {result['name']}", flush=True)
                except Exception as exc:
                    failures.append({"model": names[i], "error_type": type(exc).__name__, "message": str(exc)})
                    event(report / "events.jsonl", "evaluation_failure", **failures[-1], traceback=traceback.format_exc())
                    if not args.continue_on_error:
                        raise
    write_json(report / "evaluation_failures.json", failures)
    if not results:
        raise RuntimeError("No candidates evaluated successfully; inspect evaluation logs")
    write_reports(results, report)
    from detailed_reports import finish_reports
    finish_reports(results, args.output, report)
    event(report / "events.jsonl", "evaluation_complete", successful=len(results), failed=len(failures))
    print(f"Comparison saved to {report / 'comparison.md'}", flush=True)
    return report


def execute(args):
    plan = describe_suite(args.models, args.train_seeds, args.eval_repeats, args.eval_noise,
                          0 if args.skip_baselines else 16)
    write_json(args.output / "plan.json", plan)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        print("Plan only: no fitting or games.", flush=True)
        return
    if args.stage in ("all", "train") and any(m in NEURAL for m in args.models):
        # Validate the server environment before potentially expensive JSON packing.
        try:
            import torch
        except ImportError as exc:
            raise SystemExit("Install server dependencies first; see requirements.txt") from exc
        if args.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise SystemExit("CUDA is unavailable; install a GPU-compatible PyTorch build before training.")
            torch.ones(2, device=args.device).square().sum().item()
            props = torch.cuda.get_device_properties(args.device)
            event(args.output / "events.jsonl", "gpu_preflight", name=props.name,
                  total_memory_bytes=props.total_memory, compute_capability=torch.cuda.get_device_capability(args.device),
                  supported_architectures=torch.cuda.get_arch_list(), torch_version=str(torch.__version__),
                  cuda_version=torch.version.cuda)
    if args.stage in ("all", "prepare"):
        prepare_cache(args.data, args.output / "cache",
                      {"train": args.max_train, "validation": args.max_validation, "test": args.max_test}, args.seed)
    if args.stage in ("all", "train"):
        train_models(args)
    if args.stage in ("all", "evaluate"):
        report = evaluate_models(args)
        training_failures_path = args.output / "training_failures.json"
        training_failures = json.loads(training_failures_path.read_text()) if training_failures_path.exists() else []
        evaluation_failures = json.loads((report / "evaluation_failures.json").read_text())
        write_json(report / "completion.json", {"status": "complete_with_failures" if training_failures or evaluation_failures else "complete",
                   "utc": datetime.now(timezone.utc).isoformat(), "training_failures": training_failures,
                   "evaluation_failures": evaluation_failures})


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    session = args.output / "sessions" / stamp
    session.mkdir(parents=True, exist_ok=True)
    write_json(session / "arguments.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    write_json(session / "environment.json", environment())
    code = session / "code"
    code.mkdir()
    hashes = {}
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, code / source.name)
        hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(session / "code_sha256.json", hashes)
    with capture_console(session / "console.log"):
        event(args.output / "events.jsonl", "session_start", stage=args.stage, session=stamp)
        try:
            execute(args)
        except BaseException as exc:
            event(args.output / "events.jsonl", "session_failed", error_type=type(exc).__name__,
                  traceback=traceback.format_exc())
            traceback.print_exc()
            raise
        event(args.output / "events.jsonl", "session_end", session=stamp)


if __name__ == "__main__":
    main()
