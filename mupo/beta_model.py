import torch
import torch.nn as nn

import mamba_ssm as ms

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def stable_softmax(x):
    e_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True).values)
    return e_x / (e_x.sum(dim=-1, keepdim=True) + 1e-8)


def rescale_to_minus_one_to_one(tensor):
    tensor_min = tensor.min()
    tensor_max = tensor.max()

    if tensor_max - tensor_min == 0:
        return torch.zeros_like(tensor)

    if tensor_max - tensor_min == 0:
        print("HELP HELP HELP"*20)

    return 2 * (tensor - tensor_min) / (tensor_max - tensor_min) - 1


class BetaModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(3, 256)

        self.fc.apply(init_weights)
        #self.layer_norm = nn.LayerNorm(256)

        self.mamba = nn.Sequential()

        for _ in range(0, 2):
            self.mamba.append(ms.Mamba(256, 32, 4, 2))

        self.seq = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
        )

        self.seq.apply(init_weights)

        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x : (B, L, 3)
        # y : (B, 1)

        unbatched = len(x.shape) == 2

        if unbatched:
            x = x.unsqueeze(dim=0)
        
        start_max = torch.max(x[:, :, 1])
        duration_max = torch.max(x[:, :, 2])

        epsilon = 1e-8

        #print(f"x before normalizaion: {x}")

        x[:, :, 0] = x[:, :, 0] / 255.0
        x[:, :, 1] = x[:, :, 1] / (start_max + epsilon)
        x[:, :, 2] = x[:, :, 2] / (duration_max + epsilon)

        #print(f"x after normalizaion: {x}")

        # Check for NaNs after normalization
        if torch.isnan(x).any():
            print("NaN detected after normalization")
        
        out = self.fc(x)
        #out = self.layer_norm(out)
        
        # Check for NaNs after the first linear layer
        if torch.isnan(out).any():
            print("NaN detected after first linear layer")
        
        out = self.mamba(out)
        
        # Check for NaNs after Mamba layers
        if torch.isnan(out).any():
            print("NaN detected after Mamba layers")
        
        out = self.seq(out)
        
        # Check for NaNs after the sequential layers
        if torch.isnan(out).any():
            print("NaN detected after sequential layers")
        
        out = rescale_to_minus_one_to_one(out)
        #print(out)
        out = stable_softmax(out)
        
        # Check for NaNs after softmax
        if torch.isnan(out).any():
            print("NaN detected after softmax")
        
        #out = self.fc(x)
        #print(f"out after fc: {out}")

        #out = self.mamba(out)
        #print(f"out after mamba: {out}")

        #out = self.seq(out[:, :, :])
        #print(f"out after seq: {out}")

        #out = self.softmax(out)
        #print(f"out after softmax: {out}")

        #out = torch.clamp(out, min=epsilon, max=1.0)  # Clamp to avoid log(0) in loss calculations
        #print(f"out after clamp: {out}")

        if unbatched:
            return out.squeeze()
        else:
            return out
