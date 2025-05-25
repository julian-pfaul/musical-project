import torch
import torch.nn as nn

from tqdm import tqdm

class CustomGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomGRUCell, self).__init__()
        self.gru = nn.GRUCell(input_size, hidden_size)

        # Proper initialization
        nn.init.kaiming_uniform_(self.gru.weight_ih, nonlinearity='tanh')  # Input weights
        nn.init.kaiming_uniform_(self.gru.weight_hh, nonlinearity='tanh')  # Hidden weights
        nn.init.zeros_(self.gru.bias_ih)  # Input biases
        nn.init.zeros_(self.gru.bias_hh)  # Hidden biases

    def forward(self, x, hidden):
        return self.gru(x, hidden)

class ResidualGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.1, pool_size=4):
        super(ResidualGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.gru = CustomGRUCell(input_size, hidden_size)
        self.linear = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.pool = nn.MaxPool1d(pool_size)

        assert hidden_size % pool_size == 0

        self.expand = nn.LazyLinear(hidden_size)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, hidden):
        hidden = self.gru(x, hidden)
        residual = self.linear(x)

        hidden = hidden + residual
        hidden = self.batch_norm(hidden)
        hidden = torch.relu(hidden)
        hidden = self.dropout(hidden)

        hidden = self.pool(hidden)
        hidden = self.expand(hidden)

        return torch.tanh(hidden)

class DeepResidualGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate=0.1):
        super(DeepResidualGRU, self).__init__()

        self.output_size = output_size
        self.num_layers = num_layers

        self.gru_cells = nn.ModuleList([ResidualGRUCell(input_size if i == 0 else hidden_size, hidden_size, dropout_rate) for i in range(num_layers)])
        self.proj = nn.Linear(hidden_size, output_size)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(1)
        hidden = torch.zeros(batch_size, self.gru_cells[0].hidden_size).to(x.device)
        output_t = torch.ones(batch_size, sequence_length, self.output_size).to(x.device)

        for t in range(sequence_length):
            input_t = x[:, t, :]
            for i in range(self.num_layers):
                hidden = self.gru_cells[i](input_t, hidden)
                if torch.isnan(hidden).any():
                    print(f"NaN detected in hidden state at time {t}, layer {i}")
                input_t = hidden
                output_t[:, t, :] = self.proj(hidden)

        return output_t

class PiDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()

        self.seq_len = 512
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

        inputs = tensor[0:self.seq_len]
        labels = tensor[1:self.seq_len+1]

        return inputs, labels

