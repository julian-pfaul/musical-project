import torch
import torch.utils as utils

import math

class SequenceDataset(utils.data.Dataset):
    def __init__(self, pieces, seq_len, device):
        super().__init__()

        self.data = []

        for name, data in pieces:
            if data.shape[0] <= seq_len:
                continue

            inputs = data[0:seq_len]
            labels = data[seq_len]
            one_before_labels = data[seq_len-1]

            distance_to_one_before = labels[1] - one_before_labels[1]

            labels[1] = distance_to_one_before

            self.data.append((inputs.to(device), labels.to(device)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



