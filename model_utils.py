import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils as utils

class MusicalModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv1d(4, 8, 2)
        
        self.f_rnn = nn.RNN(8, 8)
        self.b_rnn = nn.RNN(8, 8)

        self.flatten = nn.Flatten(0, 1)

        self.fc0 = nn.Linear(64, 1)
        self.fc1 = nn.Linear(64, 1)
        self.fc2 = nn.Linear(64, 1)
        self.fc3 = nn.Linear(64, 1)
        
        

    def forward(self, piece, index):
        convoluted = self.conv(piece.permute((1, 0)))

        f_part = convoluted[:,0:index]
        b_part = convoluted[:,index+1:].flip(dims=[0])

        if index == 0:
            f_part = torch.zeros(size=(8, 1)).to(torch.get_device(piece))

        if index+1 >= convoluted.shape[1]:
            b_part = torch.zeros(size=(8, 1)).to(torch.get_device(piece))

        f_rnn_out, f_rnn_hidden = self.f_rnn(f_part.permute((1, 0)))
        b_rnn_out, b_rnn_hidden = self.b_rnn(b_part.permute((1, 0)))

        f_rnn_last = f_rnn_out[-1:,:]
        b_rnn_last = b_rnn_out[-1:,:]

        MATRIX = torch.matmul(f_rnn_last.permute((1, 0)), b_rnn_last)

        flat = self.flatten(MATRIX)

        pitch    = self.fc0(flat)
        velocity = self.fc1(flat)
        start    = self.fc2(flat)
        duration = self.fc3(flat)

        return torch.hstack((pitch, velocity, start, duration))

class MusicalDataset(utils.data.Dataset):
    def __init__(self, data_file, device):
        tensor_list = torch.load(data_file)

        self.items = []

        for name, tensor in tensor_list:
            tensor = tensor.to(device)

            for index, note in enumerate(tensor):
                self.items.append((tensor, index, note))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        tensor, index, note = self.items[idx]

        return tensor, index, note
