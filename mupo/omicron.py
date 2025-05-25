import torch
import torch.nn as nn
import torch.utils as utils

class OmicronMetaData:
    def __init__(self, mw_size=512):
        self.mw_size = mw_size

class OmicronDataset(utils.data.Dataset):
    def __init__(self, data, meta_data=OmicronMetaData()):
        super().__init__()
        self.meta_data = meta_data

        self.pieces = []
        self.ids = []

        mw_size = self.meta_data.mw_size+1

        for name, piece in data:
            self.pieces.append(piece)

            piece_len = piece.shape[0]
            windows = piece_len - mw_size + 1

            if windows >= 1:
                for offset in range(0, windows):
                    self.ids.append((len(self.pieces)-1, offset))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        index, offset = self.ids[idx]

        mw_size = self.meta_data.mw_size+1

        window = self.pieces[index][offset:offset+mw_size, :]

        inputs = window[:-1, :]
        labels = window[-1, :]

        return inputs, labels




class OmicronModel(nn.Module):
    def __init__(self, meta_data=OmicronMetaData()):
        super().__init__()
        self.meta_data = meta_data

        in_channels = 3
        hidden_channels = 64
        out_channels = 3
        conv_kernel_size = 4
        pool_kernel_size = 8

        self.conv0 = nn.Conv1d(in_channels, hidden_channels, conv_kernel_size, padding="same") # (B, C, L) -> (B, C, L)     # 64
        self.norm0 = nn.BatchNorm1d(hidden_channels) # (B, C, L) -> (B, C, L)
        self.pool0 = nn.MaxPool1d(pool_kernel_size) # (B, C, L) -> (B, C, L/4)                                              # 16
        self.conv1 = nn.Conv1d(hidden_channels, hidden_channels, conv_kernel_size, padding="same") # (B, C, L) -> (B, C, L) # 16
        self.norm1 = nn.BatchNorm1d(hidden_channels)
        self.pool1 = nn.MaxPool1d(pool_kernel_size) # (B, C, L) -> (B, C, L/4)                                              # 4

        self.proj0 = nn.Linear(hidden_channels, out_channels)
        self.proj1 = nn.LazyLinear(1)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, ipt):
        # ipt : (B, L, C)

        last = ipt[:, -1, :]

        x = ipt.permute((0, 2, 1)) # (B, L, C) -> (B, C, L)

        out = self.conv0(x)
        out = self.norm0(out)
        out = self.relu(out)
        out = self.pool0(out)

        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.pool1(out)

        # out : (B, C, L)

        out = out.permute((0, 2, 1)) # (B, C, L) -> (B, L, C)
        out = self.proj0(out)
        out = out.permute((0, 2, 1)) # (B, L, C) -> (B, C, L)
        out = self.proj1(out)
        out = out.permute((0, 2, 1)) # (B, C, L) -> (B, L, C)

        return last + out.squeeze(dim=1) # (B, C)
