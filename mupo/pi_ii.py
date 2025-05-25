import torch
import torch.nn as nn

from tqdm import tqdm

class SelectiveCell(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len=2048, dropout=0.1, pool_size=4):
        super().__init__()

        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        self.weight_h = nn.Parameter(torch.ones(input_size, hidden_size))
        self.bias_h = nn.Parameter(torch.ones(hidden_size))

        self.weight_wt = nn.Parameter(torch.ones(input_size, hidden_size, hidden_size))
        self.bias_wt = nn.Parameter(torch.ones(hidden_size, hidden_size))

        self.weight_bt = nn.Parameter(torch.ones(input_size, hidden_size))
        self.bias_bt = nn.Parameter(torch.ones(hidden_size))

        self.pool = nn.MaxPool1d(pool_size)

        assert hidden_size % pool_size == 0

        self.expand = nn.Linear(hidden_size // pool_size, hidden_size)

    def forward(self, x):
        # x : (B, S, C)
        # y : (B, S, H)
        batch_size = x.size(0)
        channels = x.size(2)

        assert self.seq_len == x.size(1)

        output_y = torch.zeros(batch_size, self.seq_len, self.hidden_size)

        prev_f = torch.zeros(batch_size, self.hidden_size)

        for t in range(0, self.seq_len):
            input_t = x[:, t, :]

            h = input_t @ self.weight_h + self.bias_h # (B, H)
            h = h + prev_f
            h = torch.relu(h)
            h = self.dropout(h)
            h = self.pool(h)
            h = self.expand(h)

            reshaped_input_t = input_t.view(-1, channels) # (B, C)

            expanded_weight_wt = self.weight_wt.unsqueeze(0) # (1, C, H, H)
            expanded_reshaped_input_t = reshaped_input_t.unsqueeze(1) # (B, 1, C)

            # (B, H, H)
            weight_f = (expanded_reshaped_input_t @ expanded_weight_wt + self.bias_wt).view(batch_size, seq_len, self.hidden_size, self.hidden_size)
            bias_f = x @ self.weight_bt + self.bias_bt # (B, H)

            f = h @ weight_f + bias_f # (B, H)
            f = torch.relu(f)
            f = self.dropout(f)
            f = self.pool(f)
            f = self.expand(f)

            output_y[:, t, :] = h

            prev_f = f

        return output_y



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

