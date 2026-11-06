"""Stress-test agents against opponent fields at 2% noise.

Usage (from the repo root):
    python evaluation/tournaments.py cooperative --agents gru:control_gru16:guard gru:diverse_gru16
    python evaluation/tournaments.py hostile     --agents gru:diverse_gru16 script:"Always Defect"
    python evaluation/tournaments.py families    --agents gru:control_gru16:guard script:two_strikes

Agent specs:
    gru:<weights>[:guard]   a trained model from models/weights (guard = never defect first)
    script:<name>           a strategy from evaluation/opponents.py (cooperative/hostile fields)
                            or a training family name such as two_strikes (families mode)

Modes:
    cooperative / hostile  The focal agent plays one match against every unordered pair of the
                     field (two seeds). Score = mean payoff per round for the focal agent.
    families         The focal agent plays N matches against two copies of each of the 16
                     scripted training families (fresh random parameters per match).

Each run appends JSON lines to the --out file so long runs can be split up.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "evaluation"))

from gru_agent import GRUAgent  # noqa: E402
from opponents import COOPERATIVE_FIELD, HOSTILE_FIELD, Scripted  # noqa: E402
from prison import PAYOFF  # noqa: E402

NOISE = 0.02


def match_length():
    """Rounded normal(100, 10), rejection-truncated to 80-200 rounds."""
    while True:
        n = round(random.gauss(100, 10))
        if 80 <= n <= 200:
            return n


def play_match(players):
    """Return the focal (seat 0) player's mean payoff per round."""
    h = [[], [], []]
    score = 0
    n = match_length()
    for _ in range(n):
        acts = [players[0].play(h[0], h[1], h[2]),
                players[1].play(h[1], h[2], h[0]),
                players[2].play(h[2], h[0], h[1])]
        acts = [x ^ int(random.random() < NOISE) for x in acts]
        for i in range(3):
            h[i].append(acts[i])
        score += PAYOFF[acts[0]][acts[1]][acts[2]]
    return score / n


def field_score(agent, field, seeds=(11, 12)):
    scores = []
    for seed in seeds:
        random.seed(seed)
        total = count = 0
        for i in range(len(field)):
            for j in range(i, len(field)):
                total += play_match([agent, field[i], field[j]])
                count += 1
        scores.append(total / count)
    return scores


class FamilyPlayer:
    """Wrap a scripted training family from pipeline/generate_training_data.py."""

    def __init__(self, family, rng):
        import generate_training_data as g
        self.g = g
        self.spec = g.sample_spec(family, rng)
        self.rng = rng
        self.name = family

    def play(self, mine, opp1, opp2):
        return self.g.choose(self.spec, mine, opp1, opp2, self.rng)


def family_scores(make_agent, matches=20, seed=4242):
    import generate_training_data as g
    out = {}
    for k, family in enumerate(g.FAMILIES):
        rng = random.Random(seed + k)
        random.seed(seed + 1000 + k)
        total = 0
        for _ in range(matches):
            total += play_match([make_agent(rng), FamilyPlayer(family, rng), FamilyPlayer(family, rng)])
        out[family] = total / matches
    return out


def build(spec, field):
    kind, _, rest = spec.partition(":")
    if kind == "gru":
        name, _, flag = rest.partition(":")
        return lambda rng=None: GRUAgent(name, never_defect_first=(flag == "guard"))
    if kind == "script":
        pool = {s.name: s for s in COOPERATIVE_FIELD + HOSTILE_FIELD}
        if rest in pool:
            return lambda rng=None: pool[rest]
        return lambda rng=None: FamilyPlayer(rest, rng or random.Random(0))
    raise SystemExit(f"Unknown agent spec: {spec}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("cooperative", "hostile", "families"))
    ap.add_argument("--agents", nargs="+", required=True)
    ap.add_argument("--matches", type=int, default=20, help="families mode: matches per family")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tournament_runs.jsonl")
    args = ap.parse_args()
    field = COOPERATIVE_FIELD if args.mode == "cooperative" else HOSTILE_FIELD
    with args.out.open("a") as fh:
        for spec in args.agents:
            make = build(spec, field)
            if args.mode == "families":
                row = {"mode": "families", "agent": spec, "scores": family_scores(make, args.matches)}
            else:
                seeds = field_score(make(), field)
                row = {"mode": args.mode, "agent": spec, "score": sum(seeds) / len(seeds), "seeds": seeds}
            print(json.dumps(row), flush=True)
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
