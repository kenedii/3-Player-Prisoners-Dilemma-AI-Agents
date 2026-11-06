"""Auditable model inventory; every model has a plain-Python exporter."""

NEURAL = {
    "linear": [], "mlp16": [16], "mlp32": [32], "mlp64_32": [64, 32],
    "mlp128_64": [128, 64], "mlp64_32_16": [64, 32, 16],
    "gru8": None, "gru16": None, "gru32": None, "rnn16": None,
}
CLASSICAL = {
    "ridge01": {"family": "ridge", "alpha": .1},
    "ridge10": {"family": "ridge", "alpha": 10.0},
    "elasticnet": {"family": "elasticnet", "alpha": .0001, "l1_ratio": .5},
    "tree4": {"family": "tree", "max_depth": 4},
    "tree8": {"family": "tree", "max_depth": 8},
    "tree12": {"family": "tree", "max_depth": 12},
    "tree16": {"family": "tree", "max_depth": 16},
    "forest16": {"family": "forest", "n_estimators": 16, "max_depth": 8},
    "forest32": {"family": "forest", "n_estimators": 32, "max_depth": 10},
    "extratrees32": {"family": "extra", "n_estimators": 32, "max_depth": 10},
    "boosted32": {"family": "boosting", "n_estimators": 32, "max_depth": 3},
}
ARCHITECTURES = tuple(NEURAL) + tuple(CLASSICAL)
STANDARD = ("linear", "mlp32", "mlp64_32", "gru16")


def describe_suite(models, seeds, repeats, noises, baseline_count=16):
    return {"architectures": list(models), "training_seeds": list(seeds),
            "learned_candidates": len(models) * len(seeds), "scripted_baselines": baseline_count,
            "opponent_pairs": 256, "repeats": repeats, "noise_levels": list(noises),
            "total_matches": (len(models) * len(seeds) + baseline_count) * 256 * repeats * len(noises)}
