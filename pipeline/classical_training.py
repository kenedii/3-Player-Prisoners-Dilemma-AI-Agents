"""Server-only scikit-learn fitting with audited array-based Python exports."""

import math
import time

import numpy as np

from agent_export import swap_index
from experiment_catalog import CLASSICAL
from experiment_logging import event, heartbeat, resources


def make_estimator(name, seed, jobs):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, MultiTaskElasticNet
    from sklearn.multioutput import MultiOutputRegressor
    spec = CLASSICAL[name]
    family = spec["family"]
    if family == "ridge":
        return Ridge(alpha=spec["alpha"], solver="lsqr", tol=1e-5, max_iter=1000)
    if family == "elasticnet":
        return MultiTaskElasticNet(alpha=spec["alpha"], l1_ratio=spec["l1_ratio"], max_iter=1000,
                                   tol=1e-4, random_state=seed)
    if family == "tree":
        return DecisionTreeRegressor(max_depth=spec["max_depth"], min_samples_leaf=32, random_state=seed)
    if family in ("forest", "extra"):
        factory = RandomForestRegressor if family == "forest" else ExtraTreesRegressor
        return factory(n_estimators=spec["n_estimators"], max_depth=spec["max_depth"],
                       min_samples_leaf=32, max_features=.7, n_jobs=jobs, random_state=seed)
    return MultiOutputRegressor(GradientBoostingRegressor(n_estimators=spec["n_estimators"],
                                max_depth=spec["max_depth"], min_samples_leaf=32,
                                learning_rate=.05, random_state=seed), n_jobs=jobs)


def encode_targets(targets, gap_scale, aux_scale):
    q = targets[:, :2] / 8
    return np.c_[(q[:, 0] + q[:, 1]) / 2, (q[:, 1] - q[:, 0]) * gap_scale,
                 targets[:, 2:] * aux_scale]


def decode(raw, gap_scale, aux_scale):
    probabilities = np.maximum(raw[:, 2:] / aux_scale, 1e-9)
    probabilities /= probabilities.sum(1, keepdims=True)
    return np.c_[raw[:, 0] - raw[:, 1] / gap_scale / 2,
                 raw[:, 0] + raw[:, 1] / gap_scale / 2, np.log(probabilities)]


def predict(estimator, x, window, gap_scale, aux_scale):
    dtype = np.float64 if hasattr(estimator, "coef_") else np.float32
    a = decode(estimator.predict(np.asarray(x, dtype=np.float32).astype(dtype)), gap_scale, aux_scale)
    b = decode(estimator.predict(np.asarray(x[:, swap_index(window)], dtype=np.float32).astype(dtype)), gap_scale, aux_scale)
    return (a + b[:, [0, 1, 2, 4, 3, 5]]) / 2


def tree_nodes(estimator):
    tree = estimator.tree_
    nodes = []
    for i in range(tree.node_count):
        leaf = tree.children_left[i] == tree.children_right[i]
        nodes.append([int(tree.feature[i]) if not leaf else -1, float(tree.threshold[i]),
                      int(tree.children_left[i]), int(tree.children_right[i]),
                      tree.value[i, :, 0].astype(float).tolist() if leaf else None])
    return nodes


def export_parameters(estimator, name, gap_scale, aux_scale):
    family = CLASSICAL[name]["family"]
    p = {"postprocess": "mean_gap_prob", "gap_scale": gap_scale, "aux_scale": aux_scale,
         "float32_inputs": True}
    if family in ("ridge", "elasticnet"):
        p.update(kind="dense", layers=[{"weight": estimator.coef_.astype(float).tolist(),
                                        "bias": estimator.intercept_.astype(float).tolist()}])
    elif family in ("tree", "forest", "extra"):
        trees = [estimator] if family == "tree" else estimator.estimators_
        p.update(kind="forest", trees=[tree_nodes(t) for t in trees])
    else:
        base, trees = [], []
        for output, fitted in enumerate(estimator.estimators_):
            base.append(float(fitted.init_.constant_[0, 0]))
            for tree in fitted.estimators_[:, 0]:
                trees.append({"output": output, "nodes": tree_nodes(tree)})
        p.update(kind="boosting", base=base, learning_rate=.05, trees=trees)
    return p


def fit_classical(args, name, seed, array, meta, directory):
    dimension = meta["dimension"]
    cap = min(args.classical_max_train or len(array), len(array))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(array), cap, replace=False)
    chosen.sort()  # Sequential disk reads.
    np.save(directory / "selected_training_indices.npy", chosen)
    records = np.array(array[chosen], copy=True)
    x = records[:, :dimension]
    targets = records[:, dimension:dimension + 6]
    swapped_targets = targets[:, [0, 1, 2, 4, 3, 5]]
    x = np.concatenate((x, x[:, swap_index(meta["window"])]))
    targets = np.concatenate((targets, swapped_targets))
    gap_scale, aux_scale = math.sqrt(max(args.gap_weight, 1e-6)), math.sqrt(max(args.aux_weight, 1e-6))
    encoded = encode_targets(targets, gap_scale, aux_scale)
    if CLASSICAL[name]["family"] in ("ridge", "elasticnet"):
        x, encoded = x.astype(np.float64), encoded.astype(np.float64)
    estimator = make_estimator(name, seed, args.classical_jobs)
    events = directory / "events.jsonl"
    event(events, "classical_fit_start", model=name, training_rows=cap, augmented_rows=len(x),
          estimator=estimator.get_params(deep=False), resources=resources())
    begin = time.perf_counter()
    with heartbeat(events, model=name, stage="fit"):
        estimator.fit(x, encoded)
    elapsed = time.perf_counter() - begin
    event(events, "classical_fit_end", seconds=elapsed, resources=resources())
    fitted_info = {"training_rows": cap, "augmented_rows": len(x), "fit_seconds": elapsed,
                   "settings": CLASSICAL[name], "gap_scale": gap_scale, "aux_scale": aux_scale}
    if hasattr(estimator, "feature_importances_"):
        fitted_info["feature_importances"] = estimator.feature_importances_.tolist()
    if hasattr(estimator, "n_iter_"):
        fitted_info["iterations"] = np.asarray(estimator.n_iter_).tolist()
    return estimator, export_parameters(estimator, name, gap_scale, aux_scale), fitted_info


def offline_metrics(estimator, array, meta, args, parameters):
    from experiment_data import batches
    d = meta["dimension"]
    totals = np.zeros(10, dtype=np.float64)
    for block in batches(array, args.batch_size):
        out = predict(estimator, block[:, :d], meta["window"], parameters["gap_scale"], parameters["aux_scale"])
        q = block[:, d:d + 2] / 8
        joint = block[:, d + 2:d + 6].argmax(1)
        gap = 8 * (q[:, 1] - q[:, 0])
        pgap = 8 * (out[:, 1] - out[:, 0])
        action = (pgap > 0).astype(int)
        correct = action == (gap > 0)
        logits = out[:, 2:]
        logits = logits - logits.max(1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
        nll = -logp[np.arange(len(block)), joint]
        confident = np.abs(gap) > 1.96 * block[:, d + 6]
        value_sq = ((8 * (out[:, :2] - q)) ** 2).sum() / 2
        gap_sq = ((pgap - gap) ** 2).sum()
        loss_sum = value_sq / 64 + args.gap_weight * gap_sq / 64 + args.aux_weight * nll.sum()
        totals += [len(block), loss_sum, value_sq, gap_sq, correct.sum(),
                   (q.max(1) - q[np.arange(len(block)), action]).sum() * 8,
                   nll.sum(), (logits.argmax(1) == joint).sum(), confident.sum(), (correct & confident).sum()]
    n, loss, vsq, gsq, agree, regret, nll, acc, conf, confcorrect = totals
    return {"examples": int(n), "loss": loss / n, "q_rmse_payoff_units": math.sqrt(vsq / n),
            "gap_rmse_payoff_units": math.sqrt(gsq / n), "teacher_action_agreement": agree / n,
            "teacher_regret_payoff_units": regret / n, "joint_nll": nll / n,
            "joint_accuracy": acc / n, "confident_examples": int(conf),
            "confident_action_agreement": confcorrect / conf if conf else None}
