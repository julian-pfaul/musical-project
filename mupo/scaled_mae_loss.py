import torch
import torch.nn as nn

class ScaledMAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_labels, y_preds):
        epsilon = 1e-8

        abs_loss = torch.abs(y_labels - y_preds) ** 2

        return torch.mean(abs_loss)
