import torch
import torch.utils as utils

class BetaDataset(utils.data.Dataset):
    def __init__(self, data):
        super().__init__()

        self.seq_len = 1
        self.data = [tensor for _, tensor in data]

    def __len__(self):
        return len([d for d in self.data if self.seq_len-1 <= d.shape[0]])

    def __getitem__(self, idx):
        data = [d for d in self.data if self.seq_len-1 <= d.shape[0]]

        tensor = data[idx]

        inputs = tensor[0:self.seq_len, :]

        pitches = tensor[1:self.seq_len+1, 0]

        one_hot_encoded_pitches = []

        for pitch in pitches:
            epsilon = 1e-8

            one_hot_vector = torch.zeros(256)
            one_hot_vector[int(pitch.item())] = 1.0 - epsilon
            one_hot_vector = one_hot_vector + epsilon
            one_hot_encoded_pitches.append(one_hot_vector)
        
        labels = torch.stack(one_hot_encoded_pitches)

        return (inputs, labels)
