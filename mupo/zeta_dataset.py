import torch
import random

class ZetaDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()

        self.seq_len = 1
        self.data = []

        for title, tensor in data:
            self.data.append(tensor)

    def set_sequence_length(self, l):
        if l < 1:
            raise RuntimeError(f"invalid sequence length l={l}")

        self.seq_len = l

    def __len__(self):
        data = [d for d in self.data if d.shape[0] >= self.seq_len+1]
        return len(data)

    def __getitem__(self, idx):
        data = [d for d in self.data if d.shape[0] >= self.seq_len+1]

        tensor = data[idx]

        #print(tensor)

        #sl = tensor.shape[0] - self.seq_len

        #s = random.randint(0, sl-1)
        #print(s, sl)

        seq = tensor[:self.seq_len+1]

        second_to_last = seq[-2]
        last = seq[-1]

        inputs = seq[:-1] # sequence with the last entry removed
        inputs_len = inputs.shape[0]
    
        last_pitch = last[0].item()

        last_pitch_arr = torch.zeros(inputs_len) + last_pitch
        last_pitch_arr = last_pitch_arr.unsqueeze(dim=1)

        inputs = torch.hstack((last_pitch_arr, inputs))

        second_to_last_start = second_to_last[1]
        last_start = last[1]

        labels = last_start - second_to_last_start

        return inputs, labels.unsqueeze(dim=0)

