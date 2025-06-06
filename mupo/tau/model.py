from .encoder import *
from .decoder import *

NUM_PITCHES = 128

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, size):
        super().__init__()

        self.proj = nn.Linear(d_model, size)

        for layer in [self.proj]:
            nn.init.xavier_uniform_(layer.weight)

            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)

class Model(nn.Module):
    def __init__(self, num_features=4, d_model=128, num_layers=3, d_ff=1024, dropout=0.1):
        super().__init__()

        self.num_features = num_features

        self.encoder = Encoder(num_features, d_model, dropout)
        self.decoder = Decoder(d_model, d_ff, num_layers, dropout)

        self.proj_s = nn.Linear(d_model, 1)
        self.proj_e = nn.Linear(d_model, 1)
        self.proj_p = ProjectionLayer(d_model, NUM_PITCHES)
        self.proj_v = nn.Linear(d_model, 1)

        for layer in [self.proj_s, self.proj_e, self.proj_v]:
            nn.init.xavier_uniform_(layer.weight)

            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        self.float()

    def forward(self, x, max_length=800):
        x = self.encoder(x)
        x = x[:, -1, :].unsqueeze(dim=1).expand(-1, max_length, -1)

        #print(x.shape)

        s = torch.zeros(x.size(0), max_length, 1).to(x.device)
        e = torch.zeros(x.size(0), max_length, 1).to(x.device)
        p = torch.zeros(x.size(0), max_length, NUM_PITCHES).to(x.device)
        v = torch.zeros(x.size(0), max_length, 1).to(x.device)

        x = self.decoder(x, max_length)

        s[:, :, :] = self.proj_s(x)
        e[:, :, :] = self.proj_e(x)
        p[:, :, :] = self.proj_p(x)
        v[:, :, :] = self.proj_v(x)

        return s, e, p, v
