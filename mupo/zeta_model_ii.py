import torch
import torch.nn as nn
import mamba_ssm as ms

import math

class ZetaModelII(nn.Module):
    def __init__(self):
        super().__init__()
        
        hidden = 24

        self.sequence0 = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Dropout(1e-4),
            ms.Mamba(
                8, 64, 64, 6
            ),
            nn.Linear(8, hidden),
            nn.Tanh(),
            nn.Dropout(1e-4),
            ms.Mamba(
                hidden, 64, 1028, 6
            ),
            #nn.ReLU(),
            #nn.Tanh(),
            #ms.Mamba(
            #    hidden, 64, 4, 2
            #),
            #nn.ReLU(),
            #nn.Tanh(),
            #ms.Mamba(
            #    hidden, 64, 4, 2
            #),
            #nn.ReLU(),
            #nn.Tanh(),
            #ms.Mamba(
            #    hidden, 64, 4, 2
            #),
            #nn.ReLU()

        )

        self.sequence1 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

        self.relu = nn.LeakyReLU()

        self.register_parameter(name='start_max_correction', param=torch.nn.Parameter(torch.rand(1) ** 2))
        self.register_parameter(name='seq_len_correction', param=torch.nn.Parameter(torch.rand(1) ** 2)) 

    def forward(self, x):
        # x : (B, L, 4)
        # y : (B, 1)

        start_max = torch.max(x[:, :, 2]) * (1.0 / self.start_max_correction.item())
        seq_len = x.shape[1] * (1.0 / self.seq_len_correction.item())

        print(self.start_max_correction.item(), self.seq_len_correction.item())

        x[:, :, 0] = x[:, :, 0] / 255.0
        x[:, :, 1] = x[:, :, 1] / 255.0
        x[:, :, 2] = x[:, :, 2] / start_max

        out = self.sequence0(x)
        out = self.sequence1(out[:, -1, :])

        out[:, 0] = out[:, 0] * (start_max / seq_len)

        return self.relu(out)




