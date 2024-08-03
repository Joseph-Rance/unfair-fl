"""Implementation of simple, densely connected network."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import Cfg


class FullyConnected(nn.Module):

    def __init__(self, config: Cfg) -> None:
        super(FullyConnected, self).__init__()

        input_size = config.input_size
        hidden = config.hidden

        self.layers = nn.ModuleList()
        for s in hidden:
            self.layers.append(nn.Linear(input_size, s))
            input_size = s
        self.drop = nn.Dropout(0.1)
        self.output = nn.Linear(input_size, 1)

    def forward(self, x: torch.float) -> torch.float:
        for l in self.layers:
            x = F.relu(l(x))
        x = self.drop(x)
        out = F.sigmoid(self.output(x))
        return out
