"""Small synthetic checks only. No dataset training or full benchmarks.

python -m unittest -v test_model_experiments
PyTorch parity checks run when torch is installed (e.g. on the server).
"""

import ast
import gzip
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
import io
from urllib import error
from unittest.mock import patch

import numpy as np

from agent_export import export_source, parameter_count, swap_index
from experiment_data import batches, prepare_cache, load_cache
from generate_training_data import encode_input
from prison import PAYOFF


def reference_forward(parameters, x, window):
    x = np.asarray(x, dtype=np.float64)
    if parameters["kind"] == "gru":
        size = parameters["hidden_size"]
        hidden = np.zeros(size)
        for t in range(window):
            if not x[3 * window + t]:
                continue
            step = np.r_[x[3 * t:3 * t + 3], x[3 * window + t]]
            gi = np.asarray(parameters["weight_ih"]) @ step + parameters["bias_ih"]
            gh = np.asarray(parameters["weight_hh"]) @ hidden + parameters["bias_hh"]
            reset, update = (1 / (1 + np.exp(-(gi[:size] + gh[:size]))),
                             1 / (1 + np.exp(-(gi[size:2 * size] + gh[size:2 * size]))))
            new = np.tanh(gi[2 * size:] + reset * gh[2 * size:])
            hidden = (1 - update) * new + update * hidden
        x = np.r_[hidden, x[-10:]]
    for i, layer in enumerate(parameters["layers"]):
        x = np.asarray(layer["weight"]) @ x + layer["bias"]
        if i < len(parameters["layers"]) - 1:
            x = np.maximum(x, 0)
    return x


class ExportTests(unittest.TestCase):
    def test_dense_and_gru_against_numpy(self):
        rng = np.random.default_rng(42)
        for kind in ("dense", "gru"):
            window, hidden = 3, 2
            width = 4 * window + 10 if kind == "dense" else hidden + 10
            p = {"kind": kind, "layers": [
                {"weight": rng.normal(0, .1, (5, width)).tolist(), "bias": rng.normal(0, .1, 5).tolist()},
                {"weight": rng.normal(0, .1, (6, 5)).tolist(), "bias": rng.normal(0, .1, 6).tolist()}]}
            if kind == "gru":
                p.update(hidden_size=hidden,
                         weight_ih=rng.normal(0, .1, (3 * hidden, 4)).tolist(),
                         weight_hh=rng.normal(0, .1, (3 * hidden, hidden)).tolist(),
                         bias_ih=rng.normal(0, .1, 3 * hidden).tolist(),
                         bias_hh=rng.normal(0, .1, 3 * hidden).tolist())
            source = export_source(p, window, "synthetic")
            tree = ast.parse(source)
            imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
            imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            self.assertEqual(imported, {"prison", "math"})
            self.assertFalse(any(isinstance(n, ast.Name) and n.id in ("open", "exec", "eval", "__import__") for n in ast.walk(tree)))
            namespace = {}
            exec(compile(source, "<synthetic-agent>", "exec"), namespace)
            agent = namespace["CompactAgent"]()
            for past in ([], [[0, 1, 0]], [[1, 0, 1], [0, 1, 0], [0, 0, 0], [1, 1, 0]]):
                x = encode_input(past, window)
                sx = [x[i] for i in swap_index(window)]
                expected = (reference_forward(p, x, window) + reference_forward(p, sx, window)[[0, 1, 2, 4, 3, 5]]) / 2
                np.testing.assert_allclose(agent.predict(x), expected, atol=1e-12)
                actual = agent.predict(sx)
                np.testing.assert_allclose(actual[:2], expected[:2], atol=1e-12)
                h = [list(v) for v in zip(*past)] if past else [[], [], []]
                before = [v[:] for v in h]
                self.assertIs(type(agent.play(*h)), int)
                self.assertEqual(h, before)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is installed on server only")
    def test_actual_pytorch_export(self):
        import torch
        from model_training import SmallModel
        rng = random.Random(75)
        from experiment_catalog import NEURAL
        for name in NEURAL:
            window = 5
            model = SmallModel(name, window).eval()
            p = model.export_parameters()
            self.assertEqual(parameter_count(p), sum(v.numel() for v in model.parameters()))
            namespace = {}
            exec(export_source(p, window, name), namespace)
            agent = namespace["CompactAgent"]()
            for t in (0, 1, 4, 5, 12):
                past = [[rng.randrange(2) for _ in range(3)] for _ in range(t)]
                x = np.asarray(encode_input(past, window), dtype=np.float32)
                with torch.inference_mode():
                    expected = model(torch.from_numpy(x[None, :])).numpy()[0]
                np.testing.assert_allclose(agent.predict(x.tolist()), expected, atol=2e-6, rtol=2e-6)

    def test_tree_and_boosting_exports(self):
        # Test an exact threshold, float32 rounding, forest averaging, and boosting sums.
        for kind in ("forest", "boosting"):
            leaves = [[.6, -.1, .2, .1, .1, .1], [.7, .2, .1, .1, .1, .2]]
            nodes = [[0, .5, 1, 2, None], [-1, -2, -1, -1, leaves[0]], [-1, -2, -1, -1, leaves[1]]]
            p = {"kind": kind, "float32_inputs": True, "postprocess": "mean_gap_prob", "gap_scale": 2., "aux_scale": .5}
            if kind == "forest":
                p["trees"] = [nodes, nodes]
            else:
                p.update(base=[0.] * 6, learning_rate=.5,
                         trees=[{"output": i, "nodes": [[0, .5, 1, 2, None],
                                  [-1, -2, -1, -1, [leaves[0][i] * 2]],
                                  [-1, -2, -1, -1, [leaves[1][i] * 2]]]} for i in range(6)])
            namespace = {}
            exec(export_source(p, 2, "test"), namespace)
            agent = namespace["CompactAgent"]()
            for value, expected_leaf in ((.4, leaves[0]), (.5, leaves[0]), (.5000000001, leaves[0]), (.6, leaves[1])):
                x = [value] + [0.] * 17
                out = agent.predict(x)
                self.assertAlmostEqual(out[0], expected_leaf[0] - expected_leaf[1] / 4)
                self.assertAlmostEqual(out[1], expected_leaf[0] + expected_leaf[1] / 4)
                self.assertTrue(np.isfinite(out).all())

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is installed on server only")
    def test_actual_sklearn_exports(self):
        from classical_training import make_estimator, encode_targets, export_parameters, predict
        from experiment_catalog import CLASSICAL
        rng = np.random.default_rng(15)
        x = rng.random((192, 18)).astype(np.float32)
        labels = np.eye(4)[rng.integers(0, 4, len(x))]
        targets = np.c_[rng.random((len(x), 2)) * 8, labels]
        y = encode_targets(targets, 2., .25)
        for name in CLASSICAL:
            fitted = make_estimator(name, 5, 1)
            fitted.fit(x.astype(np.float64), y.astype(np.float64))
            p = export_parameters(fitted, name, 2., .25)
            namespace = {}
            exec(export_source(p, 2, name), namespace)
            agent = namespace["CompactAgent"]()
            expected = predict(fitted, x[:9], 2, 2., .25)
            actual = np.asarray([agent.predict(row.tolist()) for row in x[:9]])
            np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-5, err_msg=name)


class CacheTests(unittest.TestCase):
    def test_snapshot_caps_and_split_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, cache = root / "data", root / "cache"
            data.mkdir()
            manifest = {"schema_version": 3, "payoff": PAYOFF, "config": {"window": 2},
                        "mlp_schema": {"input_dimension": 18}, "example_counts": {}, "shards": []}
            for split, identity in (("train", 0), ("validation", 1), ("test", 2)):
                filename = split + ".jsonl.gz"
                manifest["shards"].append({"file": filename, "split": split})
                manifest["example_counts"][split] = 4
                with gzip.open(data / filename, "wt", encoding="utf-8") as stream:
                    for i in range(4):
                        x = [float(identity)] * 18
                        record = {"episode_id": 10 * identity + i, "mlp_examples": [
                            {"x": x, "y": [6, 5, 1, 0, 0, 0], "defect_minus_cooperate_standard_error": .01}]}
                        stream.write(json.dumps(record) + "\n")
            (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            limits = {"train": 3, "validation": 2, "test": 2}
            prepare_cache(data, cache, limits, 5)
            meta, arrays = load_cache(cache)
            self.assertEqual(meta["dimension"], 18)
            for split, identity in (("train", 0), ("validation", 1), ("test", 2)):
                self.assertEqual(len(arrays[split]), limits[split])
                self.assertTrue((arrays[split][:, 0] == identity).all())
            blocks = list(batches(arrays["train"], 2, seed=7, block_rows=2))
            self.assertEqual(sum(map(len, blocks)), 3)
            self.assertEqual(prepare_cache(data, cache, limits, 5)["identity"], meta["identity"])
            with self.assertRaises(ValueError):
                prepare_cache(data, cache, {**limits, "train": 4}, 5)
            # Close memmaps before TemporaryDirectory cleanup on Windows.
            for array in arrays.values():
                array._mmap.close()


class BenchmarkTests(unittest.TestCase):
    def test_common_scenarios_and_reports_without_playing_games(self):
        import benchmark_agents as benchmark
        settings = {"fixed_rounds": 100, "mean_rounds": 100, "std_rounds": 10,
                    "min_rounds": 80, "max_rounds": 200}
        setups = []
        def fake_match(agent, focal_spec, opponents, rounds, noise, seed, seat):
            setups.append((opponents, rounds, noise, seed, seat))
            value = 4 + int(opponents[0]["family"] == "defect") + 2 * int(opponents[1]["family"] == "defect")
            return value, 3, 3, .5
        with patch.object(benchmark, "FAMILIES", ("cooperate", "defect")), patch.object(benchmark, "play_match", fake_match):
            a = benchmark.evaluate_candidate({"name": "script:defect", "family": "defect"}, settings, 3, 123, [.02], 1)
            b = benchmark.evaluate_candidate({"name": "script:random", "family": "random"}, settings, 3, 123, [.02], 1)
            self.assertEqual(setups[:12], setups[12:])
            self.assertEqual(len(a["pairs"]), 4)
            self.assertEqual(a["summaries"][0]["matches"], 12)
            self.assertEqual(a["summaries"][0]["mean"], 5.5)
            with tempfile.TemporaryDirectory() as temporary:
                benchmark.write_reports([a, b], Path(temporary))
                text = (Path(temporary) / "leaderboard.csv").read_text()
                self.assertIn("delta_vs_always_defect", text)
                self.assertIn("script:random", text)
                self.assertTrue((Path(temporary) / "comparison.md").exists())

    def test_partial_row_resume_and_match_traces(self):
        import benchmark_agents as benchmark
        settings = {"fixed_rounds": 100}
        calls = []
        def fake_match(*args, trace=None):
            calls.append(args[5])
            if trace is not None:
                trace.update(intended_actions=[[0, 0, 0]], executed_actions=[[0, 0, 0]])
            return 6, 6, 6, 1.
        with tempfile.TemporaryDirectory() as temporary, patch.object(benchmark, "FAMILIES", ("cooperate", "defect")), patch.object(benchmark, "play_match", fake_match):
            kwargs = dict(candidate={"name": "baseline", "family": "cooperate"}, config=settings,
                          repeats=2, seed=4, noises=[.02], timing_repeats=1, log_directory=temporary)
            a = benchmark.evaluate_candidate(**kwargs)
            self.assertEqual(len(calls), 8)
            b = benchmark.evaluate_candidate(**kwargs)
            self.assertEqual(len(calls), 8)  # Saved rows are reused without any matches.
            self.assertEqual(a["summaries"], b["summaries"])
            traces = list(Path(temporary).glob("*.matches.jsonl.gz"))
            self.assertEqual(len(traces), 2)
            with gzip.open(traces[0], "rt") as stream:
                record = json.loads(next(stream))
                self.assertIn("intended_actions", record)


class OrchestrationTests(unittest.TestCase):
    def test_detailed_reports_join_metrics_and_seed_summary(self):
        from detailed_reports import finish_reports
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / 'models' / 'mlp32_seed1234'
            model.mkdir(parents=True)
            metrics = {"q_rmse_payoff_units": .1, "teacher_regret_payoff_units": .01, "joint_nll": .4, "joint_accuracy": .8}
            stats = {"architecture": "mlp32", "seed": 1234, "parameters": 6950, "validation": metrics,
                     "test": metrics, "export_parity": {"passed": True}, "training_examples": 100}
            (model / 'training.json').write_text(json.dumps(stats))
            report = root / 'reports'
            report.mkdir()
            result = {"name": "mlp32_seed1234", "deployment": {"max_play_ms": .5, "source_bytes": 2000,
                      "observed_under_one_second": True}, "summaries": [{"noise": .02, "mean": 5.,
                      "ci95_low": 4.9, "ci95_high": 5.1, "matches": 100}]}
            finish_reports([result], root, report)
            self.assertIn('seeds_completed', (report / 'architecture_summary.csv').read_text())
            self.assertIn('mlp32_seed1234', (report / 'training_metrics.csv').read_text())
            page = (report / 'comparison.html').read_text(encoding='utf-8')
            self.assertIn('mlp32_seed1234', page)
            self.assertNotIn('PAYLOAD', page)
            self.assertEqual(json.loads((report / 'performance_summary.json').read_text())[0]['mean'], 5.)

    def test_large_preset_counts_and_dry_run_has_no_side_effects(self):
        import run_model_experiments as runner
        with tempfile.TemporaryDirectory() as temporary:
            with patch('sys.argv', ['runner', '--suite', 'large', '--output', temporary, '--dry-run']):
                args = runner.arguments()
            self.assertEqual(len(args.models), 21)
            self.assertEqual(len(args.train_seeds), 3)
            with patch.object(runner, 'prepare_cache') as prepare, patch.object(runner, 'train_models') as train, patch.object(runner, 'evaluate_models') as evaluate:
                runner.execute(args)
                prepare.assert_not_called(); train.assert_not_called(); evaluate.assert_not_called()
            plan = json.loads((Path(temporary) / 'plan.json').read_text())
            self.assertEqual(plan['total_matches'], 1213440)

