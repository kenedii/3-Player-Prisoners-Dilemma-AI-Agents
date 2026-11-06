"""Offline export and dependency-free inference. No ML libraries required here."""

import inspect
from pathlib import Path

from generate_training_data import encode_input


RUNTIME = '''
class CompactAgent(Player):
    def agentName(self):
        return MODEL_NAME

    @staticmethod
    def _dense(x, weights, bias):
        return [b + sum(a * v for a, v in zip(row, x))
                for row, b in zip(weights, bias)]

    @staticmethod
    def _sigmoid(x):
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        e = math.exp(x)
        return e / (1.0 + e)

    @staticmethod
    def _tree(x, nodes):
        index = 0
        while nodes[index][0] >= 0:
            feature, threshold, left, right, _ = nodes[index]
            index = left if x[feature] <= threshold else right
        return nodes[index][4]

    def _forward(self, x):
        p = PARAMETERS
        if p.get('float32_inputs'):
            x = [struct.unpack('f', struct.pack('f', v))[0] for v in x]
        if p['kind'] in ('forest', 'boosting'):
            if p['kind'] == 'forest':
                raw = [0.0] * 6
                for tree in p['trees']:
                    leaf = self._tree(x, tree)
                    raw = [a + b / len(p['trees']) for a, b in zip(raw, leaf)]
            else:
                raw = p['base'][:]
                for tree in p['trees']:
                    raw[tree['output']] += p['learning_rate'] * self._tree(x, tree['nodes'])[0]
            return self._decode(raw)
        if p['kind'] in ('gru', 'rnn'):
            size = p['hidden_size']
            hidden = [0.0] * size
            for t in range(WINDOW):
                mask = x[3 * WINDOW + t]
                if not mask:
                    continue
                step = x[3 * t:3 * t + 3] + [mask]
                gi = self._dense(step, p['weight_ih'], p['bias_ih'])
                gh = self._dense(hidden, p['weight_hh'], p['bias_hh'])
                if p['kind'] == 'rnn':
                    hidden = [math.tanh(a + b) for a, b in zip(gi, gh)]
                    continue
                updated = []
                for j in range(size):
                    reset = self._sigmoid(gi[j] + gh[j])
                    update = self._sigmoid(gi[size + j] + gh[size + j])
                    new = math.tanh(gi[2 * size + j] + reset * gh[2 * size + j])
                    updated.append((1 - update) * new + update * hidden[j])
                hidden = updated
            x = hidden + x[-10:]
        for i, layer in enumerate(p['layers']):
            x = self._dense(x, layer['weight'], layer['bias'])
            if i < len(p['layers']) - 1:
                x = [max(0.0, v) for v in x]
        return self._decode(x)

    @staticmethod
    def _decode(raw):
        p = PARAMETERS
        if p.get('postprocess') != 'mean_gap_prob':
            return raw
        mean, gap = raw[0], raw[1] / p['gap_scale']
        probabilities = [max(1e-9, v / p['aux_scale']) for v in raw[2:]]
        total = sum(probabilities)
        return [mean - gap / 2, mean + gap / 2] + [math.log(v / total) for v in probabilities]

    def predict(self, x):
        # Exact symmetry: Q values invariant to exchanging opponents.
        swapped = [x[i] for i in SWAP_INDEX]
        a, b = self._forward(x), self._forward(swapped)
        return [(a[i] + b[j]) / 2 for i, j in enumerate([0, 1, 2, 4, 3, 5])]

    def play(self, myHistory, oppHistory1, oppHistory2):
        # Stateless reconstruction also handles reused objects and repeated seats.
        past = list(zip(myHistory, oppHistory1, oppHistory2))
        outputs = self.predict(encode_input(past, WINDOW))
        self.last_outputs = outputs  # Optional offline trace; no effect on decisions.
        return int(outputs[1] > outputs[0])
'''


def swap_index(window):
    indices = list(range(4 * window + 10))
    for t in range(window):
        indices[3 * t + 1], indices[3 * t + 2] = indices[3 * t + 2], indices[3 * t + 1]
    offset = 4 * window
    for a, b in ((2, 3), (5, 6)):
        indices[offset + a], indices[offset + b] = indices[offset + b], indices[offset + a]
    return indices


def export_source(parameters, window, name):
    # Python repr preserves enough float precision for PyTorch/Python parity.
    return ("# Generated standalone agent. No external model files or ML libraries.\n"
            "# Outputs: two Q values scaled by 1/8, then four opponent logits.\n"
            "from prison import Player\nimport math\n\n"
            + ("import struct\n" if parameters.get("float32_inputs") else "")
            + "PAYOFF = [[[6,3],[3,0]],[[8,5],[5,2]]]\n"
            f"WINDOW = {window!r}\nMODEL_NAME = {name!r}\n"
            f"SWAP_INDEX = {swap_index(window)!r}\nPARAMETERS = {parameters!r}\n\n"
            + inspect.getsource(encode_input) + "\n" + RUNTIME)


def load_agent(path):
    """Load only locally generated trusted agent source during offline evaluation."""
    namespace = {"__name__": "exported_agent"}
    exec(compile(Path(path).read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace["CompactAgent"]


def parameter_count(parameters):
    def count(value):
        if isinstance(value, list):
            return sum(count(v) for v in value)
        return int(isinstance(value, (float, int)))
    if parameters['kind'] in ('forest', 'boosting'):
        # Numeric stored model fields, not a neural trainable-parameter count.
        trees = parameters['trees']
        nodes = [n for t in trees for n in (t['nodes'] if isinstance(t, dict) else t)]
        return sum(4 + len(n[4] or []) for n in nodes)
    total = sum(count(layer[k]) for layer in parameters["layers"] for k in ("weight", "bias"))
    if parameters["kind"] in ("gru", "rnn"):
        total += sum(count(parameters[k]) for k in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"))
    return total
