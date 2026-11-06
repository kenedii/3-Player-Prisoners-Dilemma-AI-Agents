"""Pure-Python inference for the trained GRU agents.

No ML libraries are needed: the forward pass uses only the standard library.
Weights live in models/weights/<name>.json.

    from gru_agent import GRUAgent
    agent = GRUAgent("control_gru16", never_defect_first=True)   # PrisonerAI500K
    action = agent.play(my_history, opp1_history, opp2_history)  # 0 = C, 1 = D
"""
import json
import math
from pathlib import Path

PAYOFF = [[[6, 3], [3, 0]], [[8, 5], [5, 2]]]
WINDOW = 50
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def encode_input(past, window=WINDOW):
    """Turn the match history into the model's 210-number input vector.

    past is a list of rounds, each (my action, opp1 action, opp2 action).
    Layout: last `window` rounds (left-padded), a real-round mask, then
    10 match summary features.
    """
    rounds_played = len(past)
    recent = past[-window:]
    missing = window - len(recent)

    vector = [0] * (3 * missing)
    for row in recent:
        for action in row:
            vector.append(action)

    for _ in range(missing):
        vector.append(0)
    for _ in range(len(recent)):
        vector.append(1)

    rates = []
    streaks = []
    for seat in range(3):
        cooperations = 0
        for row in past:
            if row[seat] == 0:
                cooperations += 1
        rates.append(cooperations / rounds_played if rounds_played else 0.0)

        streak = 0
        for row in reversed(past):
            if row[seat] != 1:
                break
            streak += 1
        streaks.append(streak / 100)

    both_cooperate = 0
    both_defect = 0
    total_payoff = 0
    for row in past:
        if row[1] == 0 and row[2] == 0:
            both_cooperate += 1
        if row[1] == 1 and row[2] == 1:
            both_defect += 1
        total_payoff += PAYOFF[row[0]][row[1]][row[2]]

    if rounds_played:
        summary = [both_cooperate / rounds_played, both_defect / rounds_played,
                   total_payoff / (8 * rounds_played)]
    else:
        summary = [0.0, 0.0, 0.0]

    return vector + [rounds_played / 100] + rates + streaks + summary


def _swap_index(window=WINDOW):
    """Input positions with opponent 1 and opponent 2 exchanged."""
    index = []
    for t in range(window):
        index += [3 * t, 3 * t + 2, 3 * t + 1]
    index += list(range(3 * window, 4 * window))
    s = 4 * window
    index += [s + i for i in (0, 1, 3, 2, 4, 6, 5, 7, 8, 9)]
    return index


SWAP_INDEX = _swap_index()


class GRUAgent:
    """GRU value model: predicts the payoff of cooperating vs defecting now.

    never_defect_first: if True, the agent cooperates whenever both opponents
    cooperated last round, even if the model prefers defecting. This is the
    rule that turns control_gru16 into the PrisonerAI500K agent.
    """

    def __init__(self, weights="control_gru16", never_defect_first=False, name=None):
        path = Path(weights)
        if not path.suffix:
            path = WEIGHTS_DIR / f"{weights}.json"
        self.p = json.loads(path.read_text())
        self.never_defect_first = never_defect_first
        self.name = name or path.stem + ("+guard" if never_defect_first else "")
        self.last_outputs = None

    def agentName(self):
        return self.name

    @staticmethod
    def _dense(x, weights, bias):
        outputs = []
        for i in range(len(weights)):
            total = 0.0
            for j in range(len(x)):
                total += weights[i][j] * x[j]
            outputs.append(bias[i] + total)
        return outputs

    @staticmethod
    def _sigmoid(x):
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        e = math.exp(x)
        return e / (1.0 + e)

    def _forward(self, x):
        p = self.p
        size = p["hidden_size"]
        hidden = [0.0] * size
        for t in range(WINDOW):
            mask = x[3 * WINDOW + t]
            if not mask:
                continue
            step = [x[3 * t], x[3 * t + 1], x[3 * t + 2], mask]
            gi = self._dense(step, p["weight_ih"], p["bias_ih"])
            gh = self._dense(hidden, p["weight_hh"], p["bias_hh"])
            updated = []
            for j in range(size):
                reset = self._sigmoid(gi[j] + gh[j])
                update = self._sigmoid(gi[size + j] + gh[size + j])
                new = math.tanh(gi[2 * size + j] + reset * gh[2 * size + j])
                updated.append((1 - update) * new + update * hidden[j])
            hidden = updated

        x = hidden + x[-10:]
        layers = p["layers"]
        for i in range(len(layers)):
            x = self._dense(x, layers[i]["weight"], layers[i]["bias"])
            if i < len(layers) - 1:
                x = [max(0.0, v) for v in x]
        return x

    def predict(self, x):
        """[Q_cooperate, Q_defect, P_CC, P_CD, P_DC, P_DD], averaged over both opponent orders."""
        swapped = [x[i] for i in SWAP_INDEX]
        normal_out = self._forward(x)
        swapped_out = self._forward(swapped)
        swapped_position = [0, 1, 2, 4, 3, 5]
        return [(normal_out[i] + swapped_out[swapped_position[i]]) / 2 for i in range(6)]

    def play(self, myHistory, oppHistory1, oppHistory2):
        past = list(zip(myHistory, oppHistory1, oppHistory2))
        outputs = self.predict(encode_input(past))
        self.last_outputs = outputs
        if outputs[1] <= outputs[0]:
            return 0
        if self.never_defect_first and past and oppHistory1[-1] == 0 and oppHistory2[-1] == 0:
            return 0
        return 1
