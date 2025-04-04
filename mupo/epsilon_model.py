import torch
import torch.nn as nn
import mamba_ssm as ms

class EpsilonModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc = nn.Linear(4, 128)

        self.seq0 = nn.Sequential()

        dropout_p = 1e-4

        for _ in range(0, 3):
            self.seq0.append(nn.Dropout(p=dropout_p))
            self.seq0.append(nn.Linear(128, 128))
            self.seq0.append(nn.Tanh())
            self.seq0.append(ms.Mamba(128, 16, 8, 4))
            self.seq0.append(nn.ReLU())

        self.seq1 = nn.Sequential(
            nn.Linear(128, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 2)
        )
       

        self.relu = nn.ReLU()

    def forward(self, x):
        # x : (B, L, 4)
        # y : (B, 2)

        start_max = torch.max(x[:, :, 2])

        x[:, :, 0] = x[:, :, 0] / 255.0
        x[:, :, 1] = x[:, :, 1] / 255.0
        x[:, :, 2] = x[:, :, 2] / start_max

        out = self.fc(x)
        out = self.seq0(out)
        out = self.seq1(out[:, -1, :]) # (B, L, 4) -> (B, 1, 2)

        out[:, 0] = out[:, 0] * start_max
        
        return self.relu(out)


