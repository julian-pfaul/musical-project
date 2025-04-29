import torch
import torch.nn as nn
import mamba_ssm as ms

class MuModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.outputs = 144

        self.proj0 = nn.Linear(3, 512)

        self.dropout = nn.Dropout(1e-2)
        self.act = nn.Tanh()
        self.mamba = ms.Mamba(512, 16, 4, 2)

        self.proj1 = nn.Linear(512, self.outputs)

        self.softmax = nn.Softmax()

    def forward(self, x):
        # x : (B, L, 3)

        out = self.proj0(x)
        out = self.dropout(out)
        out = self.act(out)
        out = self.mamba(out)
        out = self.proj1(out)

        out = self.softmax(out)

        return out

