from .model import *
from .utils import *

import torch
import torch.nn as nn

import math

class SequenceModel(Model):
    def __init__(self, seq_len):
        super().__init__("sequence-model")

        in_size = 3
        hidden_size = int(64 * (1 + math.log(seq_len) / 2))
        n_layers = int(2 * (1 + math.log(seq_len) / 2))
        out_size = 3
        
        dropout_p = 1e-8

        self.rnn = nn.RNN(in_size, hidden_size, n_layers, dropout=dropout_p)
        self.h2o = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        _, hidden = self.rnn(x)

        batched = is_batched(x)

        if batched:
            hidden = hidden.permute((1, 0 ,2))
            out = self.h2o(hidden[:, -1, :])
            return out
        else:
            out = self.h2o(hidden[-1, :])
            return out
