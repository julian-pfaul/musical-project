import torch
import torch.nn as nn

class MusicalModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.RNN(4, 20, 4)

        self.relu = nn.ReLU()

        self.flatten = nn.Flatten(0)
        self.fc0 = nn.LazyLinear(20)
        self.fc1 = nn.Linear(20, 4)

        self.fc2 = nn.Linear(20, 4)
        
    def forward(self, x):
        out, hidden = self.rnn(x)
        out = self.relu(out)
        out = self.fc2(out)

        flattened = self.flatten(hidden)

        gen = self.fc0(flattened)
        gen = self.relu(gen)
        gen = self.fc1(gen)

        return torch.vstack((out, gen))
