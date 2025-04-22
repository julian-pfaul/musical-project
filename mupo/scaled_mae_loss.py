import torch
import torch.nn as nn

class ScaledMAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_labels, y_preds):
        epsilon = 1e-8

        abs_loss = torch.abs(y_labels - y_preds)

        abs_loss[:, :, 0] = abs_loss[:, :, 0] * 1.0
        abs_loss[:, :, 1] = abs_loss[:, :, 1] * 10000.0
        abs_loss[:, :, 2] = abs_loss[:, :, 2] * 100.0

        return torch.sum(abs_loss)
