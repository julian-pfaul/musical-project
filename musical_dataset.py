import torch
import torch.utils as utils

class MusicalDataset(utils.data.Dataset):
    def __init__(self, path, n_inputs, device):
        file_data = torch.load(path)

        self.data = []

        for (name, piece_tensor) in file_data:
            index = n_inputs

            if n_inputs >= piece_tensor.shape[0]:
                continue

            piece_input = piece_tensor[0:index].to(device)
            piece_label = piece_tensor[0:index+1].to(device)

            self.data.append((piece_input, piece_label))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
