import torch
import torch.nn as nn
import mamba_ssm as ms

class KappaModel(nn.Module):
    def __init__(self):
        super().__init__()

        inputs = 3
        dropout_p = 1e-3
        hidden0 = 128
        hidden1 = 64
        outputs = 3
        n_mamba_layers = 3
        n_linear_layers = 12

        self.proj0 = nn.Linear(inputs, hidden0)

        self.mamba_seq = nn.Sequential()

        for _ in range(0, n_mamba_layers):
            self.mamba_seq.append(nn.Dropout(dropout_p))
            self.mamba_seq.append(nn.Tanh())
            self.mamba_seq.append(ms.Mamba(hidden0, 32, 8, 4))
            self.mamba_seq.append(nn.ELU())
            self.mamba_seq.append(nn.Linear(hidden0, hidden0))

        self.proj1 = nn.Linear(hidden0, hidden1)

        self.linear_seq = nn.Sequential()

        for _ in range(0, n_linear_layers):
            self.linear_seq.append(nn.ELU())
            self.linear_seq.append(nn.Linear(hidden1, hidden1))

        self.proj2 = nn.Linear(inputs + hidden0 + hidden1, outputs)

    def forward(self, x):
        # x : (B, L, 3)
        # y : (B, L, 3)

        out = self.proj0(x)
        mamba = self.mamba_seq(out)
        out = self.proj1(mamba)
        out = self.linear_seq(out)

        combined = torch.dstack((x, mamba, out))

        out = self.proj2(combined)

        return out
