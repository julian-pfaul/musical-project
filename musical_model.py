import torch
import torch.nn as nn

import math

class MusicalModel(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        
        self.input_size = 4
        self.hidden_size = int(64 * (1 + math.log(n_inputs)))
        self.num_layers = int(2 * (1 + math.log(n_inputs)))
        self.output_size = 4

        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True)

        self.h2o = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        out, hidden = self.rnn(x)

        hidden = hidden if self.num_layers == 1 else hidden[-1:, :, :]

        out = self.h2o(hidden.permute((1, 0, 2)))

        return torch.hstack((x, out))
    

