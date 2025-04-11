import torch
import torch.nn as nn
import mamba_ssm as ms

class KappaModel(nn.Module):
    def __init__(self):
        super().__init__()

        inputs = 3
        dropout_p = 1e-4
        hidden0 = 256
        hidden1 = 64
        outputs = 3
        n_mamba_layers = 3
        n_linear_layers = 2

        self.proj0 = nn.Linear(inputs, hidden0)

        self.mamba_seq = nn.Sequential()

        for _ in range(0, n_mamba_layers):
            self.mamba_seq.append(nn.Dropout(dropout_p))
            self.mamba_seq.append(nn.Tanh())
            self.mamba_seq.append(ms.Mamba(hidden0, 32, 8, 4))

        self.proj1 = nn.Linear(hidden0, hidden1)

        self.linear_seq = nn.Sequential()

        for _ in range(0, n_linear_layers):
            self.linear_seq.append(nn.LeakyReLU())
            self.linear_seq.append(nn.Linear(hidden1, hidden1))

        self.proj2 = nn.Linear(inputs + hidden0 + hidden1, outputs)

    def forward(self, x):
        # x : (B, L, 3)
        # y : (B, L, 3)

        projected0 = self.proj0(x)
        through_mamba = self.mamba_seq(projected0)
        projected1 = self.proj1(through_mamba)
        through_linear = self.linear_seq(projected1)

        #print(through_mamba.shape)
        #print(through_linear.shape)
        combined_inputs_mamba_and_linear = torch.dstack((x, through_mamba, through_linear))
        #print(combined_mamba_and_linear.shape)

        out = self.proj2(combined_inputs_mamba_and_linear)

        return out
