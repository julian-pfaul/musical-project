import torch
import torch.utils as utils

class MusicalDataset(utils.data.Dataset):
    def __init__(self, path, device):
        file_data = torch.load(path)

        self.data = []

        for (name, piece_tensor) in file_data:
            for index in range(4 + 1, piece_tensor.shape[0] - 1):
                note_tensor = piece_tensor[index]

                self.data.append((piece_tensor[0:index].to(device), note_tensor.to(device)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
