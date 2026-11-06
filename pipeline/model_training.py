"""Server-only PyTorch models. Exported inference does not import this file."""

import torch
from torch import nn

from agent_export import swap_index
from experiment_catalog import NEURAL


MODEL_NAMES = tuple(NEURAL)


class SmallModel(nn.Module):
    def __init__(self, name, window):
        super().__init__()
        self.name, self.window = name, window
        dimension = 4 * window + 10
        self.register_buffer("swap", torch.tensor(swap_index(window)))
        self.register_buffer("output_swap", torch.tensor([0, 1, 2, 4, 3, 5]))
        if name.startswith(("gru", "rnn")):
            self.recurrent_kind = "gru" if name.startswith("gru") else "rnn"
            self.hidden_size = int(name[3:])
            factory = nn.GRU if self.recurrent_kind == "gru" else nn.RNN
            self.gru = factory(4, self.hidden_size, batch_first=True)
            dimensions = [self.hidden_size + 10, 32, 6]
        else:
            hidden = NEURAL[name]
            dimensions = [dimension] + hidden + [6]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dimensions, dimensions[1:]))

    def raw(self, x):
        if self.name.startswith(("gru", "rnn")):
            batch = x.shape[0]
            actions = x[:, :3 * self.window].reshape(batch, self.window, 3)
            masks = x[:, 3 * self.window:4 * self.window]
            lengths = masks.sum(1).long()
            sequence = torch.cat((actions, masks.unsqueeze(-1)), dim=-1)
            # Convert left padding to right padding; select the last valid step.
            positions = (torch.arange(self.window, device=x.device)[None, :]
                         + (self.window - lengths)[:, None]) % self.window
            sequence = sequence.gather(1, positions[:, :, None].expand(-1, -1, 4))
            outputs, _ = self.gru(sequence)
            hidden = outputs[torch.arange(batch, device=x.device), (lengths - 1).clamp(min=0)]
            hidden = hidden * (lengths > 0).unsqueeze(1)
            x = torch.cat((hidden, x[:, -10:]), dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x

    def forward(self, x):
        joined = torch.cat((x, x[:, self.swap]), dim=0)
        a, b = self.raw(joined).chunk(2, dim=0)
        return (a + b[:, self.output_swap]) / 2

    def export_parameters(self):
        p = {"kind": self.recurrent_kind if self.name.startswith(("gru", "rnn")) else "dense",
             "layers": [{"weight": layer.weight.detach().cpu().tolist(),
                         "bias": layer.bias.detach().cpu().tolist()} for layer in self.layers]}
        if self.name.startswith(("gru", "rnn")):
            p["hidden_size"] = self.hidden_size
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                p[name] = getattr(self.gru, name + "_l0").detach().cpu().tolist()
        return p
