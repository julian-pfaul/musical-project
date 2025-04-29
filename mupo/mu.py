class MuMetaData:
    def __init__(self):
        self.intervals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.interval_directions = [1,-1]
        self.interval_octaves = [0, 1, 2, 3, 4, 5]

        self.time_ratios_i = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]
        self.time_ratios_ii = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]


def probability_distribution(options, given):
    length = len(options)

    out = torch.zeros(length)

    min_diff_index = int(0)
    min_difference = abs(options[0] - given)

    for index in range(0, length):
        difference = abs(options[index] - given)

        if difference < min_difference:
            min_diff_index = int(index)
            min_difference = difference

    out[min_diff_index] = 1.0

    return out


import torch
import torch.nn as nn
import mamba_ssm as ms
import math
import numpy as np

class MuSelectorModel(nn.Module):
    def __init__(self, list_data, sequence_length):
        super().__init__()

        self.list_data = list_data # of type MuMetaData
        self.length = sequence_length

        inputs = 3
        hidden = int(max(8 * (1.0 + math.log(self.length)), 64.0))
        dropout_p = 1e-2
        outputs = len(self.list_data)
        #self.h_size = 8

        self.proj0 = nn.Linear(inputs, hidden)
        #self.proj_h0 = nn.Linear(self.h_size, self.h_size)
        #self.relu = nn.ReLU()

        #self.unif = nn.Linear(hidden + self.h_size, hidden)

        self.seq0 = nn.Sequential()

        for _ in range(0, 2):
            self.seq0.append(nn.Dropout(dropout_p))
            self.seq0.append(nn.Sigmoid())
            self.seq0.append(ms.Mamba(hidden, 16, 4, 2))
            self.seq0.append(nn.Sigmoid())
            self.seq0.append(nn.Linear(hidden, hidden))

        self.proj1 = nn.Linear(hidden, outputs)
        self.softmax = nn.Softmax(dim=1)

        #self.proj_h1 = nn.Linear(hidden, self.h_size)

    def forward(self, ipt):
        # my guess is that 12 notes take around 1 second on average
        time_normalisation_factor = 1.0 / (self.length / 12.0)

        x = torch.zeros_like(ipt)
        x[:, :, 0] = ipt[:, :, 0] / 127.0
        x[:, :, 1] = ipt[:, :, 1] / time_normalisation_factor

        #if h == None:
        #    h = torch.ones(x.shape[0], x.shape[1], self.h_size).cuda()

        out = self.proj0(x)
        #h_out = self.proj_h0(h)
        #h_out = self.relu(h_out)
        #out = self.unif(torch.dstack((x_out, h_out)))
        #out = self.relu(out)

        pre_cut = self.seq0(out)
        out = self.proj1(pre_cut[:, -1, :])
        #out = self.relu(out)
        out = self.softmax(out)

        #h = self.proj_h1(pre_cut)

        return out#, h

    def format(self, x):
        indices = torch.argmax(x, dim=1)
        selected = torch.tensor([self.list_data[index] for index in indices]).unsqueeze(dim=1)

        return selected

class MuModel(nn.Module):
    def __init__(self, meta_data, sequence_length):
        super().__init__()

        self.meta_data = meta_data
        self.length = sequence_length

        self.interval_selector = MuSelectorModel(self.meta_data.intervals, self.length)
        self.interval_direction_selector = MuSelectorModel(self.meta_data.interval_directions, self.length)
        self.interval_octave_selector = MuSelectorModel(self.meta_data.interval_octaves, self.length)
        self.time_ratio_selector_i = MuSelectorModel(self.meta_data.time_ratios_i, self.length)
        self.time_ratio_selector_ii = MuSelectorModel(self.meta_data.time_ratios_ii, self.length)

    def forward(self, x):
        interval_out = self.interval_selector(x)
        interval_direction_out = self.interval_direction_selector(x)
        interval_octave_out = self.interval_octave_selector(x)
        time_ratio_out_i = self.time_ratio_selector_i(x)
        time_ratio_out_ii = self.time_ratio_selector_ii(x)

        out = torch.hstack((interval_out, interval_direction_out, interval_octave_out, time_ratio_out_i, time_ratio_out_ii))

        return out

    def format(self, x):
        interval_selector_list_length = len(self.interval_selector.list_data)
        interval_direction_selector_list_length = len(self.interval_direction_selector.list_data)
        interval_octave_selector_list_length = len(self.interval_octave_selector.list_data)
        time_ratio_selector_i_list_length = len(self.time_ratio_selector_i.list_data)
        time_ratio_selector_ii_list_length = len(self.time_ratio_selector_ii.list_data)

        index_accumulator = 0

        interval_section = x[:, index_accumulator:index_accumulator+interval_selector_list_length]
        index_accumulator += interval_selector_list_length

        interval_direction_section = x[:, index_accumulator:index_accumulator+interval_direction_selector_list_length]
        index_accumulator += interval_direction_selector_list_length

        interval_octave_section = x[:, index_accumulator:index_accumulator+interval_octave_selector_list_length]
        index_accumulator += interval_octave_selector_list_length

        time_ratio_i_section = x[:, index_accumulator:index_accumulator+time_ratio_selector_i_list_length]
        index_accumulator += time_ratio_selector_i_list_length

        time_ratio_ii_section = x[:, index_accumulator:index_accumulator+time_ratio_selector_ii_list_length]

        intervals = self.interval_selector.format(interval_section)
        interval_directions = self.interval_direction_selector.format(interval_direction_section)
        interval_octaves = self.interval_octave_selector.format(interval_octave_section)
        time_ratios_i = self.time_ratio_selector_i.format(time_ratio_i_section)
        time_ratios_ii = self.time_ratio_selector_ii.format(time_ratio_ii_section)

        pitch_deltas = intervals + 12.0 * interval_octaves * interval_directions
        start_delta_ratios = time_ratios_i
        duration_ratios = time_ratios_ii

        return torch.hstack((pitch_deltas, start_delta_ratios, duration_ratios)).cuda()

    def apply_transformation(self, transformation, x, musescore=True):
        first = x[:, 0, :]
        last = x[:, -1, :].clone()

        m = 0 if not musescore else 0.0010 # weird time correction only needed for MIDIs generated by musescore

        print(transformation)
        print(last)

        last[:, 0] += transformation[:, 0]
        last[:, 1] += (first[:, 2] + m) * transformation[:, 1]
        last[:, 2] = first[:, 2] * transformation[:, 2]

        print(last)

        return last.cuda()


class MuHyperModel(nn.Module):
    def __init__(self, meta_data, lengths):
        super().__init__()

        self.module_list = nn.ModuleList()
        for l in range(0, lengths):
            self.module_list.extend([MuModel(meta_data, l+1).cuda()])

    def prepare(self, sequence_length):
        idx = sequence_length

        added = False

        while idx >= len(self.module_list):
            self.module_list.extend([MuModel(len(self.module_list) + 1).cuda()])

            added = True

        return added

    def format(self, x):
        return self.module_list[0].format(x)

    def apply_transformation(self, t, x, musescore=True):
        return self.module_list[0].apply_transformation(t, x, musescore)

    def forward(self, x):
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

        out = self.module_list[idx](x)

        if unbatched:
            out = out.squeeze()

        return out


class MuDataset(torch.utils.data.Dataset):
    def __init__(self, data, meta_data, initial_sequence_length=1):
        super().__init__()

        self.data = []
        self.meta_data = meta_data
        self.sequence_length = initial_sequence_length

        for _, piece_tensor in data:
            self.data.append(piece_tensor)

    def __len__(self):
        return len([tensor for tensor in self.data if tensor.shape[0] > self.sequence_length])

    def __getitem__(self, idx):
        pieces = [piece for piece in self.data if piece.shape[0] > self.sequence_length]

        piece = pieces[idx]
        inputs = piece[0:self.sequence_length]

        first = piece[0]
        last = piece[self.sequence_length-1]
        unknown = piece[self.sequence_length]

        pitch_delta = int((unknown[0] - last[0]).item())
        time_ratio_i = ((unknown[1] - last[1]) / first[2]).item()
        time_ratio_ii = (unknown[2] / first[2]).item()

        direction = np.sign(pitch_delta)
        octave = 12.0
        octaves = 0

        while abs(pitch_delta) > octave or pitch_delta < 0:
            pitch_delta -= octave * direction
            octaves += 1

        interval = abs(pitch_delta)

        interval_pd = probability_distribution(self.meta_data.intervals, interval)
        direction_pd = probability_distribution(self.meta_data.interval_directions, direction)
        octave_pd = probability_distribution(self.meta_data.interval_octaves, octaves)
        time_ratio_i_pd = probability_distribution(self.meta_data.time_ratios_i, time_ratio_i)
        time_ratio_ii_pd = probability_distribution(self.meta_data.time_ratios_ii, time_ratio_ii)

        #print(interval_pd, direction_pd, octave_pd, time_ratio_i_pd, time_ratio_ii_pd)

        labels = torch.hstack((interval_pd, direction_pd, octave_pd, time_ratio_i_pd, time_ratio_ii_pd))

        return inputs, labels


