import torch
import torch.utils as utils

import math

class SequenceDataset(utils.data.Dataset):
    def __init__(self, pieces, seq_len):
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

            self.data.append((inputs, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, labels = self.data[idx]

        labels_p = labels[0]
        labels_r = labels[1:]

        p_arr = torch.zeros(256)
        p_arr[int(labels_p.item())] = 1.0

        labels_p = p_arr
        
        return inputs, labels_p, labels_r



