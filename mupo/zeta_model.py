import torch
import torch.nn as nn
import mamba_ssm as ms

import math

class ZetaModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc = nn.Linear(4, 320)

        self.seq0 = nn.Sequential()

        dropout_p = 1e-6

        for _ in range(0, 4):
            self.seq0.append(nn.Linear(320, 320))
            self.seq0.append(nn.Tanh())
            self.seq0.append(ms.Mamba(320, 16, 8, 4))
            self.seq0.append(nn.ELU())
            self.seq0.append(nn.Dropout(p=dropout_p))


        self.seq1 = nn.Sequential(
            nn.Linear(320, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
       

        self.relu = nn.ReLU()

    def forward(self, x):
        # x : (B, L, 4)
        # y : (B, 1)

        start_max = torch.max(x[:, :, 2])
        seq_len = x.shape[1]

        x[:, :, 0] = x[:, :, 0] / 255.0
        x[:, :, 1] = x[:, :, 1] / 255.0
        x[:, :, 2] = x[:, :, 2] / start_max

        out = self.fc(x)
        out = self.seq0(out)
        out = self.seq1(out[:, -1, :]) # -> out : (B, 1)

        out[:, 0] = out[:, 0] * (start_max / (1 + math.log(seq_len)))

        return self.relu(out)


