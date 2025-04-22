import torch

class LambdaDataset(torch.utils.data.Dataset):
    def __init__(self, data, initial_sequence_length=4):
        super().__init__()

        self.data = []
        self.sequence_length = initial_sequence_length

        for _, piece_tensor in data:
            self.data.append(piece_tensor)

    def __len__(self):
        return len([tensor for tensor in self.data if tensor.shape[0] > self.sequence_length])

    def __getitem__(self, idx):
        pieces = [piece for piece in self.data if piece.shape[0] > self.sequence_length]

        piece = pieces[idx]
        inputs = piece[0:self.sequence_length]
        labels = piece[self.sequence_length+1]

        return inputs, labels


