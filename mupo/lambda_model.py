import torch
import torch.nn as nn
import mamba_ssm as ms
import numpy as np

def step_function(x, lowest_step, steps, step_width):
    accumulator = torch.zeros_like(x)

    for n in range(1, steps+1):
        accumulator += torch.sigmoid(x + steps*step_width/2 - step_width * n)

    return lowest_step + accumulator


class LambdaModel(nn.Module):
    def __init__(self, length):
        super().__init__()

        #self.p_mamba = ms.Mamba(length, 16, 4, 2)
        #self.s_mamba = ms.Mamba(length, 16, 4, 2)
        #self.d_mamba = ms.Mamba(length, 16, 4, 2)

        #self.softmax = nn.Softmax(dim=0)

        #self.fc = nn.Linear(3 * length, 3)
        self.length = length

        hidden = 256

        self.proj0 = nn.Linear(3, hidden)

        self.seq0 = nn.Sequential()

        for _ in range(0, 2):
            self.seq0.append(nn.Dropout(1e-4))
            self.seq0.append(nn.Sigmoid())
            self.seq0.append(ms.Mamba(hidden, 16, 4, 2))
            self.seq0.append(nn.LeakyReLU())
            self.seq0.append(nn.Linear(hidden, hidden))

        #self.seq1 = nn.Sequential()
#
#        for _ in range(0, 1):
#            self.seq1.append(nn.Linear(64, 32))
#            self.seq1.append(nn.LeakyReLU())
#            self.seq1.append(nn.Linear(32, 64))
#            self.seq1.append(nn.LeakyReLU())

        self.proj1 = nn.Linear(hidden, 5)

    def forward(self, inp, gen_mode=False):
        # x : (B, L, 3)
        # y : (B, 3)

        x = torch.zeros_like(inp)

        x[:, :, 0] = inp[:, :, 0] / 255.0
        x[:, :, 1] = inp[:, :, 1] / self.length
        x[:, :, 2] = inp[:, :, 2]

        #print(f"before proj0: {x}")
        out = self.proj0(x)
        #print(f"after proj0: {out}")
        out = self.seq0(out)
        #print(f"after mamba: {out}")
        #out = self.seq1(out[:, -1, :])
        out = self.proj1(out[:, -1, :])

        out[:, 0] = step_function(out[:, 0], -71, 144, 24)
        #out[:, 1] = step_function(out[:, 1], 0, 64, 24)
        out[:, 1:4+1] = step_function(out[:, 1:4+1], -63, 128, 24)

        #print(' '.join([f"{val.item():.4f}" for val in out[-1, :]]))
        
        #print(out)

        if gen_mode:
            out[:, 0] = torch.round(out[:, 0])
            out[:, 1:] = torch.round(out[:, 1:], decimals=3)
            print(out)

        ret = torch.zeros_like(x[:, -1, :])

        ret[:, 0] = inp[:, -1, 0] + out[:, 0]
        print(inp[:, 0, 2] + 0.0010)
        print(torch.nan_to_num(torch.div(out[:, 1], out[:, 2]), nan=0.0, posinf=0.0, neginf=0.0))
        ret[:, 1] = inp[:, -1, 1] + ((inp[:, 0, 2] + 0.0010) * torch.nan_to_num(torch.div(out[:, 1], out[:, 2]), nan=0.0, posinf=0.0, neginf=0.0))
        ret[:, 2] = inp[:, 0, 2] * (torch.nan_to_num(torch.div(out[:, 3],out[:, 4]), nan=0.0, posinf=0.0, neginf=0.0)) 

        if gen_mode:
            print(ret)
            ret[:, 1:] = torch.round(ret[:, 1:], decimals=3)
            ret[:, 0] = torch.round(ret[:, 0])

        #print(ret[-1, :])

        return ret

        #pitch = x[:, :, 0].unsqueeze(dim=0) # -> (B, L, 1)
        #start = x[:, :, 1].unsqueeze(dim=0)      
        #duration = x[:, :, 2].unsqueeze(dim=0)   

        #p_out = self.p_mamba(pitch)
        #s_out = self.s_mamba(start)
        #d_out = self.d_mamba(duration)

        #p_out = p_out.squeeze() # -> (B, L)
        #s_out = s_out.squeeze()
        #d_out = d_out.squeeze()

        #p_out = self.softmax(p_out)
        #s_out = self.softmax(s_out)
        #d_out = self.softmax(d_out)

        #out = torch.cat((p_out, s_out, d_out), 1) # -> (B, 3L)
        
        #out = self.fc(out) # -> (B, 3)

        #return out


class LambdaHyperModel(nn.Module):
    def __init__(self, lengths):
        super().__init__()

        self.module_list = nn.ModuleList()
        for l in range(0, lengths):
            self.module_list.extend([LambdaModel(l+1).cuda()])

    def prepare(self, sequence_length):
        idx = sequence_length

        added = False

        while idx >= len(self.module_list):
            self.module_list.extend([LambdaModel(len(self.module_list) + 1).cuda()])

            added = True

        return added



    def forward(self, x, gen_mode=False):
        #
        # x : (B, L, 3)
        #
        #   pitch : (B, L, 1)
        #   start : (B, L, 1)
        #   duration : (B, L, 1)
        #
        
        unbatched = len(x.shape) == 2

        if unbatched:
            x = x.unsqueeze(dim=0)

        idx = int(x.shape[1]) - 1

        out = self.module_list[idx](x, gen_mode)

        if unbatched:
            out = out.squeeze()

        return out
        

