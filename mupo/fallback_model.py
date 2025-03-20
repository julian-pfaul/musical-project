from .model import *
from .utils import *

import torch.nn as nn

class FallbackModel(Model):
    def __init__(self):
        super().__init__("fallback-model")

        in_size = 3
        hidden_size = 48
        num_layers = 6
        out_size = 3
        dropout_p = 1e-8

        self.rnn = nn.RNN(in_size, hidden_size, num_layers, dropout=dropout_p)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        _, hidden = self.rnn(x)

        if is_batched(x):
            hidden = hidden[-1:, :, :]
        else:
            hidden = hidden[-1:, :]

        out = self.fc(hidden)
        return out
