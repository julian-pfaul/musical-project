import torch
import torch.nn as nn
import mamba_ssm as ms

import numpy as np

class KappaModel(nn.Module):
    def __init__(self):
        super().__init__()

        inputs = 3
        dropout_p = 1e-4
        hidden0 = 128
        hidden1 = 32
        outputs = 5
        n_mamba_layers = 1
        n_linear_layers = 1

        self.proj0 = nn.Linear(inputs, hidden0)

        self.mamba_seq = nn.Sequential()

        for _ in range(0, n_mamba_layers):
            self.mamba_seq.append(nn.Dropout(dropout_p))
            self.mamba_seq.append(nn.Tanh())
            self.mamba_seq.append(ms.Mamba(hidden0, 8, 4, 2))
            self.mamba_seq.append(nn.ELU())
            self.mamba_seq.append(nn.Linear(hidden0, hidden0))

        self.proj1 = nn.Linear(hidden0, hidden1)

        self.linear_seq = nn.Sequential()

        for _ in range(0, n_linear_layers):
            self.linear_seq.append(nn.ELU())
            self.linear_seq.append(nn.Linear(hidden1, hidden1))

        self.proj2 = nn.Linear(inputs + hidden0 + hidden1, outputs)

        self.relu = nn.ReLU()

    def special_rounding_operation(self, ipt):
        ipt[:, :, 0:] = ipt[:, :, 0:] + torch.sin(ipt[:, :, 0:] * np.pi) / 2
    
        return ipt
    
    #def sig_rounding(self, ipt):


    def forward(self, x):
        # x : (B, L, 3)
        # y : (B, L, 3)

        unbatched = len(x.shape) == 2

        if unbatched:
            x = x.unsqueeze(dim=0)

        out = self.proj0(x)
        mamba = self.mamba_seq(out)
        out = self.proj1(mamba)
        out = self.linear_seq(out)

        combined = torch.dstack((x, mamba, out))

        out = self.proj2(combined)

        rounding_iterations = 2

        for _ in range(0, rounding_iterations):
            out = self.special_rounding_operation(out)
    
        #correction_scaler = 1 * (10 ** (-rounding_iterations / 3))
    
        out = (out + 1) / 2 #+ correction_scaler
        out[:, :, 2:] = out[:, :, 2:] + 1

        print(' '.join([f"{elem.item():.2f}" for elem in out[0, -1, :]]))

        real_out = torch.zeros_like(x)

        #epsilon = 1e-8
        
        #for batch in range(0, x.shape[2]):
        #    for length in range(0, x.shape[1]):
        #        real_out[batch, length, 0] = x[batch, length, 0] + out[batch, length, 0]
        #        real_out[batch, length, 1] = x[batch, length, 1] + x[batch, length, 2] * (out[batch, length, 1] / max(epsilon, out[batch, length, 2])) + out[batch, length, 5]
        #        real_out[batch, length, 2] = x[batch, length, 2] * (out[batch, length, 3] / max(epsilon, out[batch, length, 4])) + out[batch, length, 6]

        real_out[:, :, 0] = x[:, :, 0] + out[:, :, 0]
        real_out[:, :, 1] = x[:, :, 1] + x[:, :, 2] * (out[:, :, 1] / out[:, :, 2])
        real_out[:, :, 2] = x[:, :, 2] * (out[:, :, 3] / out[:, :, 4])

        #print(f"sm: {out[0, -1, :]}")

        if unbatched:
            real_out = real_out.squeeze() 

        return real_out
