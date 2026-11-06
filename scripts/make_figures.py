"""Render every README figure (light + dark variants) from the CSVs in results/.

    python scripts/make_figures.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"],
                  ramp=["#f3f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
                  neg="#e34948"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"],
                 ramp=["#1e2633", "#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
                 neg="#e66767"),
}

# Colour follows the entity in every chart.
MODEL_SLOT = {
    "PrisonerAI500K": 0,          # control GRU16 + never-defect-first
    "Diverse GRU16": 1,
    "Policy-improvement GRU16": 2,
    "Original GRU16": 3,
    "Control GRU16": 4,
    "Self-play GRU16": 5,
}
ARM_SLOT = {"control": 4, "diverse": 1, "policy_improvement": 2, "original": 3, "selfplay": 5}
ARM_LABEL = {"control": "Control (original data)", "diverse": "Diverse opponents", "selfplay": "Self-play league",
             "policy_improvement": "Policy improvement", "original": "Original sweep", "script": "Scripted baseline"}


def style(t):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "xtick.color": t["ink2"], "ytick.color": t["ink2"],
        "axes.edgecolor": t["axis"], "axes.titlecolor": t["ink"], "axes.titleweight": "bold",
        "axes.titlesize": 14, "axes.titlelocation": "left", "axes.titlepad": 30,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "legend.frameon": False, "legend.labelcolor": t["ink2"],
    })


def finish(fig, ax, t, subtitle=None):
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, color=t["ink2"], fontsize=10.5, va="bottom")
    fig.tight_layout()


def render(name, draw):
    for mode, t in THEMES.items():
        style(t)
        fig = draw(t)
        fig.savefig(OUT / f"{name}-{mode}.png", dpi=170)
        plt.close(fig)
    print("wrote", name)


def read_csv(name):
    with open(RES / name) as f:
        return list(csv.DictReader(f))


def hbar_values(ax, ys, xs, t, at=None, fmt="{:.2f}", pad=0.02):
    """Label each bar with its value, placed just past `at` (e.g. the end of an error bar)."""
    for y, x, a in zip(ys, xs, at or xs):
        ax.text(a + pad, y, fmt.format(x), va="center", ha="left", fontsize=9.5, color=t["ink"])


# ---------------------------------------------------------------- figure 1
def fig_architectures(t):
    rows = [r for r in read_csv("architecture_sweep.csv") if r["noise"] == "0.02"]
    rows.sort(key=lambda r: float(r["mean_score_across_seeds"]))
    neural = {"gru8", "gru16", "gru32", "rnn16", "mlp16", "mlp32", "mlp64_32", "mlp64_32_16", "mlp128_64"}
    fig, ax = plt.subplots(figsize=(9, 7.4))
    ys = range(len(rows))
    xs = [float(r["mean_score_across_seeds"]) for r in rows]
    err = [float(r["stddev_across_seeds"]) for r in rows]
    cols = [t["series"][0] if r["architecture"] in neural else t["muted"] for r in rows]
    ax.barh(list(ys), xs, color=cols, height=0.72, edgecolor=t["surface"], linewidth=1.5)
    ax.errorbar(xs, list(ys), xerr=err, fmt="none", ecolor=t["ink2"], elinewidth=1, capsize=2)
    ax.set_yticks(list(ys), [r["architecture"] for r in rows])
    hbar_values(ax, ys, xs, t, at=[x + e for x, e in zip(xs, err)])
    ax.set_xlim(3.7, 4.75)
    ax.set_xlabel("Mean payoff per round at 2% noise (3 training seeds, ±1 sd)")
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Architecture sweep: recurrent nets win")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=t["series"][0], label="Neural network"),
                       Patch(color=t["muted"], label="Linear / tree model")], loc="lower right")
    finish(fig, ax, t, "20 architectures, each trained on 2M labelled rounds, scored against all 16 scripted families")
    return fig


# ---------------------------------------------------------------- figure 2
def fig_noise(t):
    rows = read_csv("architecture_sweep.csv")
    pick = [("gru16", "GRU 16", 0), ("mlp64_32", "MLP 64-32", 2), ("tree16", "Decision tree (depth 16)", 3),
            ("linear", "Linear", 5)]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for arch, label, slot in pick:
        pts = sorted((float(r["noise"]) * 100, float(r["mean_score_across_seeds"])) for r in rows if r["architecture"] == arch)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=t["series"][slot], linewidth=2.2, marker="o", markersize=7,
                markeredgecolor=t["surface"], markeredgewidth=2)
        ax.text(xs[-1] + 0.12, ys[-1], f"{label}  {ys[-1]:.2f}", va="center", color=t["ink"], fontsize=10)
    ax.set_xticks([0, 2, 5], ["0%", "2% (target)", "5%"])
    ax.set_xlim(-0.3, 7.2)
    ax.set_ylabel("Mean payoff per round")
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Every model loses ground as noise rises")
    finish(fig, ax, t, "Action-flip probability vs score; the GRU's lead holds at every noise level")
    return fig


# ---------------------------------------------------------------- figure 3
def fig_campaign(t):
    rows = [r for r in read_csv("campaign_final_evaluation.csv") if r["noise"] == "0.02"]
    rows.sort(key=lambda r: float(r["mean"]))
    fig, ax = plt.subplots(figsize=(9, 6.6))
    ys = list(range(len(rows)))
    xs = [float(r["mean"]) for r in rows]
    lo = [x - float(r["ci95_low"]) for x, r in zip(xs, rows)]
    hi = [float(r["ci95_high"]) - x for x, r in zip(xs, rows)]
    names, cols = [], []
    for r in rows:
        a = r["agent"]
        if a.startswith("script:"):
            names.append({"two_strikes": "Two Strikes", "grim": "Grim trigger", "defect": "Always Defect"}[a[7:]])
            cols.append(t["muted"])
        else:
            arm, model = a.split("__")
            names.append(f"{ARM_LABEL[arm]} · {model.split('_')[0].upper()}")
            cols.append(t["series"][ARM_SLOT[arm]])
    ax.barh(ys, xs, color=cols, height=0.72, edgecolor=t["surface"], linewidth=1.5)
    ax.errorbar(xs, ys, xerr=[lo, hi], fmt="none", ecolor=t["ink2"], elinewidth=1, capsize=2)
    ax.set_yticks(ys, names)
    hbar_values(ax, ys, xs, t, at=[x + h for x, h in zip(xs, hi)])
    ax.set_xlim(3.6, 4.7)
    ax.set_xlabel("Mean payoff per round at 2% noise (95% CI, 5,832 matches each)")
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Training-data campaign: every model beats the classics")
    finish(fig, ax, t, "Four data recipes x GRU16/GRU32, scored on fresh seeds against all scripted families")
    return fig


# ---------------------------------------------------------------- figure 4
def fig_generalisation(t):
    rows = [r for r in read_csv("campaign_generalization.csv") if r.get("noise") == "0.02"]
    models = [("original__gru16_seed1234", "Original GRU16", 3), ("control__gru16_seed2345", "Control GRU16", 4),
              ("diverse__gru16_seed2345", "Diverse GRU16", 1), ("selfplay__gru16_seed2345", "Self-play GRU16", 5),
              ("policy_improvement__gru16_seed1234", "Policy-improvement GRU16", 2)]
    subsets = [("all_opponent_pairs", "Known scripted\nopponents"), ("unseen_family_pairs", "Unseen scripted\nfamilies"),
               ("learned_opponent_pairs", "Other learned\nagents")]
    val = {(r["candidate"], r["subset"]): float(r["delta_vs_two_strikes"]) for r in rows}
    fig, ax = plt.subplots(figsize=(10, 5.6))
    w = 0.16
    for k, (key, label, slot) in enumerate(models):
        xs = [i + (k - 2) * w for i in range(len(subsets))]
        vs = [val[(key, s)] for s, _ in subsets]
        ax.bar(xs, vs, width=w, color=t["series"][slot], edgecolor=t["surface"], linewidth=1.5, label=label)
        for x, v in zip(xs, vs):
            ax.text(x, v + (0.012 if v >= 0 else -0.012), f"{v:+.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8.2, color=t["ink2"])
    ax.axhline(0, color=t["axis"], linewidth=1.2)
    ax.set_xticks(range(len(subsets)), [s for _, s in subsets])
    ax.set_ylabel("Payoff per round vs Two Strikes")
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.25, 0.5)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=9.5)
    ax.set_title("Who generalises? Each data recipe wins a different test")
    finish(fig, ax, t, "Paired difference against the Two Strikes baseline (0 = same as Two Strikes)")
    return fig


# ---------------------------------------------------------------- heatmap helper
def heatmap(ax, t, matrix, rows, cols, vmin, vmax, fmt="{:.2f}"):
    cmap = LinearSegmentedColormap.from_list("seq", t["ramp"])
    ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = matrix[i][j]
            frac = (v - vmin) / (vmax - vmin)
            light_text = frac > 0.55 if t is THEMES["light"] else frac < 0.45
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8.8,
                    color="#ffffff" if light_text else ("#0b0b0b" if t is THEMES["light"] else "#f0efec"))
    ax.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color=t["surface"], linewidth=2)
    ax.tick_params(which="minor", length=0)


# ---------------------------------------------------------------- figure 5
def fig_robustness(t):
    rows = read_csv("robustness_populations.csv")
    pops = ["balanced_known", "cooperative", "retaliatory", "exploitative", "new_behaviours", "learned_mixed", "previous_holdouts"]
    labels = ["Balanced\nknown", "Cooperative", "Retaliatory", "Exploitative", "New\nbehaviours", "Learned\nagents", "Earlier\nholdouts"]
    agents = [("diverse_gru16", "Diverse GRU16"), ("policy_improvement_gru16", "Policy-improvement GRU16"),
              ("two_strikes", "Two Strikes (script)")]
    val = {}
    for r in rows:
        val[(r["candidate"], r["population"])] = float(r["mean"])
    present = [a for a in agents if any((a[0], p) in val for p in pops)]
    if len(present) < len(agents):
        keys = sorted({r["candidate"] for r in rows})
        present = [(k, k) for k in keys]
    m = [[val.get((a, p), float("nan")) for p in pops] for a, _ in present]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    heatmap(ax, t, m, [n for _, n in present], [l.replace("\n", " ") for l in labels], 4.0, 5.4)
    ax.set_title("Robustness check: no model wins every population")
    finish(fig, ax, t, "Mean payoff per round, 4,032 fresh paired matches at 2% noise (darker = higher)")
    return fig


# ---------------------------------------------------------------- tournament data
def tournament(field):
    return [r for r in read_csv("stress_tournaments.csv") if r["field"] == field]


def fig_cooperative_field(t):
    rows = tournament("cooperative")
    show = [r for r in rows if r["kind"] == "script" or r["guard"] == "0" or r["model"] == "PrisonerAI500K"]
    show.sort(key=lambda r: float(r["score"]))
    fig, ax = plt.subplots(figsize=(10, 7.8))
    ys = list(range(len(show)))
    xs = [float(r["score"]) for r in show]
    cols = [t["series"][MODEL_SLOT[r["model"]]] if r["kind"] == "model" else t["muted"] for r in show]
    ax.barh(ys, xs, color=cols, height=0.72, edgecolor=t["surface"], linewidth=1.5)
    ax.set_yticks(ys, [r["label"] for r in show])
    for lbl, r in zip(ax.get_yticklabels(), show):
        if r["kind"] == "model":
            lbl.set_fontweight("bold")
            lbl.set_color(t["ink"])
    hbar_values(ax, ys, xs, t)
    ax.set_xlim(2.0, 5.1)
    ax.set_xlabel("Mean payoff per round vs every pair of the field (2% noise, 2 seeds)")
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    rank = len(show) - [r["model"] for r in show].index("PrisonerAI500K")
    ax.set_title(f"Cooperative field: PrisonerAI500K ranks #{rank} of {len(show)}")
    finish(fig, ax, t, "15 forgiving, retaliating and grim strategies; models in bold")
    return fig


def fig_guard(t):
    rows = tournament("cooperative") + tournament("hostile")
    models = ["Control GRU16", "Diverse GRU16", "Original GRU16", "Policy-improvement GRU16", "Self-play GRU16"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, field, title in zip(axes, ("cooperative", "hostile"), ("Cooperative field", "Hostile field")):
        ys = []
        for i, m in enumerate(models):
            raw = [r for r in rows if r["field"] == field and r["base"] == m and r["guard"] == "0"]
            grd = [r for r in rows if r["field"] == field and r["base"] == m and r["guard"] == "1"]
            if not raw or not grd:
                continue
            a, b = float(raw[0]["score"]), float(grd[0]["score"])
            c = t["series"][MODEL_SLOT[m]]
            ax.plot([a, b], [i, i], color=t["axis"], linewidth=2.2, zorder=1)
            ax.scatter([a], [i], s=70, color=t["surface"], edgecolor=c, linewidth=2, zorder=2)
            ax.scatter([b], [i], s=95, color=c, edgecolor=t["surface"], linewidth=1.5, zorder=3)
            ax.text(max(a, b) + 0.06, i, f"{b - a:+.2f}", va="center", fontsize=9.5, color=t["ink"])
            ys.append(i)
        ax.set_yticks(range(len(models)), models)
        ax.set_title(title, fontsize=12)
        ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Mean payoff per round")
    lo = min(float(r["score"]) for r in rows if r["kind"] == "model") - 0.15
    hi = max(float(r["score"]) for r in rows if r["kind"] == "model") + 0.25
    for ax in axes:
        ax.set_xlim(lo, hi)
    fig.suptitle("The never-defect-first rule: a big gain with cooperators, a smaller cost with exploiters",
                 x=0.01, ha="left", fontweight="bold", fontsize=14, color=t["ink"])
    fig.text(0.01, 0.885, "Hollow dot = model alone, filled dot = model + rule", color=t["ink2"], fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def fig_families(t):
    data = [json.loads(line) for line in open(RES / "family_benchmark.jsonl")]
    order = [("gru:control_gru16:guard", "PrisonerAI500K"), ("gru:control_gru16", "Control GRU16"),
             ("gru:diverse_gru16", "Diverse GRU16"), ("gru:policy_improvement_gru16", "Policy-improvement GRU16"),
             ("gru:original_gru16", "Original GRU16"), ("script:two_strikes", "Two Strikes"),
             ("script:tft_any", "Tit for Tat"), ("script:grim", "Grim trigger"), ("script:generous_tft", "Generous TFT"),
             ("script:defect", "Always Defect"), ("script:cooperate", "Always Cooperate")]
    by = {d["agent"]: d["scores"] for d in data}
    order = [o for o in order if o[0] in by]
    fams = list(by[order[0][0]].keys())
    pretty = {"tft_any": "TFT (any)", "tft_both": "TFT (both)", "tft_one": "TFT (one)", "generous_tft": "Generous TFT",
              "contrite_tft": "Contrite TFT", "two_strikes": "Two Strikes", "wsls": "Win-stay lose-shift",
              "memory_one": "Memory-one", "cooperate": "Always C", "defect": "Always D"}
    m = [[by[a][f] for f in fams] for a, _ in order]
    avg = [sum(r) / len(r) for r in m]
    fig, ax = plt.subplots(figsize=(13, 5.8))
    heatmap(ax, t, [r + [v] for r, v in zip(m, avg)], [n for _, n in order],
            [pretty.get(f, f.replace("_", " ").title()) for f in fams] + ["AVERAGE"], 0.0, 7.5, fmt="{:.1f}")
    ax.axvline(len(fams) - 0.5, color=t["surface"], linewidth=5)
    for lbl in ax.get_yticklabels()[:5]:
        lbl.set_fontweight("bold")
        lbl.set_color(t["ink"])
    ax.set_title("Head-to-head: score against two copies of each scripted family")
    finish(fig, ax, t, "Mean payoff per round, 20 matches per cell at 2% noise; learned models in bold")
    return fig


if __name__ == "__main__":
    render("architecture_sweep", fig_architectures)
    render("noise_sensitivity", fig_noise)
    render("campaign_leaderboard", fig_campaign)
    render("generalisation", fig_generalisation)
    render("robustness_populations", fig_robustness)
    if (RES / "stress_tournaments.csv").exists():
        render("cooperative_field", fig_cooperative_field)
        render("guard_effect", fig_guard)
    if (RES / "family_benchmark.jsonl").exists():
        render("family_heatmap", fig_families)
