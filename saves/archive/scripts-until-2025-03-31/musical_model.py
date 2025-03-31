import torch
import torch.nn as nn

import math

class MusicalModel(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        
        self.input_size = 4
        self.hidden_size = int(64 * (1 + math.log(n_inputs) / 2))
        self.num_layers = int(2 * (1 + math.log(n_inputs) / 2))
        self.output_size = 4
        self.dropout_probability = 1e-8

        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout_probability, batch_first=True, nonlinearity='tanh')

        self.h2o = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        out, hidden = self.rnn(x)

        batched = len(x.shape) == 3

        if batched:
            hidden = hidden if self.num_layers == 1 else hidden[-1:, :, :]
            out = self.h2o(hidden.permute((1, 0, 2)))
            return torch.hstack((x, out))
        else:
            hidden = hidden if self.num_layers == 1 else hidden[-1:, :]
            out = self.h2o(hidden)
            return torch.vstack((x, out))
    
class MusicalHyperModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.m_sub_modules = dict()

    def forward(self, x):
        seq_len = None

        if len(x.shape) == 2: # unbatched
            seq_len = x.shape[0]
        elif len(x.shape) == 3: # batched
            seq_len = x.shape[1]
        else:
            raise RuntimeError(f'dimensionality error, expected [L, 4] or [N, L, 4] for batched inputs, but got {x.shape}')

        seq_len_key = str(seq_len)

        if seq_len_key not in self.m_sub_modules: # submodule for sequences of length seq_len doesn't exist
            model = MusicalModel(seq_len).to("cuda" if torch.cuda.is_available() else "cpu")

            self.m_sub_modules[seq_len_key] = {
                'model': model
            }

        model = self.m_sub_modules[seq_len_key]['model']

        if sum(1 for _ in self.children()) == 0:
            self.add_module('musical_model', model)
        else:
            self.set_submodule('musical_model', model)

        if self.training:
            model.train()
        else:
            model.eval()

        return model(x)
