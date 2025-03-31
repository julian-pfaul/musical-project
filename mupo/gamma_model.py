import torch
import torch.nn as nn
import mamba_ssm as ms

class GammaModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc0 = nn.Linear(4, 32)

        self.seq = nn.Sequential()

        for _ in range(0, 3):
            self.seq.append(nn.Linear(32, 32))
            self.seq.append(nn.Tanh())
            self.seq.append(ms.Mamba(32, 12, 8, 4))
            self.seq.append(nn.ReLU())

        self.fc1 = nn.Linear(32, 2)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x : (B, L, 4)
        # y : (B, 2)

        start_max = torch.max(x[:, :, 2])

        x[:, :, 0] = x[:, :, 0] / 255.0
        x[:, :, 1] = x[:, :, 1] / 255.0
        x[:, :, 2] = x[:, :, 2] / start_max

        out = self.fc0(x)
        out = self.seq(out)
        out = self.fc1(out[:, -1, :]) # (B, L, 4) -> (B, 1, 2)

        out[:, 0] = out[:, 0]
        
        return self.relu(out)


