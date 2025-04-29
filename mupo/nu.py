class NuMetaData:
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

import random
import torch
import torch.nn as nn
import mamba_ssm as ms
import math
import numpy as np

class NuSelectorModel(nn.Module):
    def __init__(self, list_data, sequence_length):
        super().__init__()

        self.list_data = list_data # of type NuMetaData
        self.length = sequence_length
        
        kernel_size = 3
        stride = 1
        padding = (kernel_size - 1) // 2
        
        in_channels = 3
        hidden_channels = 32
        after_flatten = sequence_length * hidden_channels

        dropout_p = 1e-4
        outputs = len(self.list_data)
        

        #self.proj0 = nn.Linear(in_channels, hidden_channels)
        #self.conv0 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        #self.dropout = nn.Dropout(dropout_p)
        #self.flatten = nn.Flatten(1, 2)
        #self.proj1 = nn.Sequential(
        #        nn.Linear(after_flatten, 512),
        #        nn.ReLU(),
        #        nn.Linear(512, 512),
        #        nn.ReLU(),
        #        nn.Linear(512, outputs)
        #)
        self.seq = nn.Sequential(
                nn.Linear(3, 128),
                nn.Dropout(dropout_p),
                nn.Sigmoid(),
                ms.Mamba(128, 16, 4, 2),
                nn.Linear(128, outputs)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, ipt, temperature=1.0):
        # my guess is that 12 notes take around 1 second on average
        #time_normalisation_factor = 1.0 / (self.length / 12.0)

        x = torch.zeros_like(ipt)
        x[:, :, 0] = ipt[:, :, 0] / 127.0
        x[:, :, 1] = ipt[:, :, 1] / 1.0

        #print(f"x.shape={x.shape}")

        #out = self.proj0(x)
        #print(f"out.shape={out.shape}")
        #out = out.permute(0, 2, 1)
        #out = self.conv0(out)
        #print(f"out.shape={out.shape}")

        #out = self.flatten(out)
        #print(f"out.shape={out.shape}")

        #out = self.dropout(out)
        #out = self.proj1(out)
        out = self.seq(x)

        out = out / temperature
        out = self.softmax(out[:, -1, :])

        #print(f"out.shape={out.shape}")

        return out

    def format(self, x):
        indices = torch.argmax(x, dim=1)
        selected = torch.tensor([self.list_data[index] for index in indices]).unsqueeze(dim=1)

        return selected

class NuModel(nn.Module):
    def __init__(self, meta_data, sequence_length):
        super().__init__()

        self.meta_data = meta_data
        self.length = sequence_length

        self.interval_selector = NuSelectorModel(self.meta_data.intervals, self.length)
        self.interval_direction_selector = NuSelectorModel(self.meta_data.interval_directions, self.length)
        self.interval_octave_selector = NuSelectorModel(self.meta_data.interval_octaves, self.length)
        self.time_ratio_selector_i = NuSelectorModel(self.meta_data.time_ratios_i, self.length)
        self.time_ratio_selector_ii = NuSelectorModel(self.meta_data.time_ratios_ii, self.length)

    def forward(self, x, temperature=1.0):
        interval_out = self.interval_selector(x, temperature)
        interval_direction_out = self.interval_direction_selector(x, temperature)
        interval_octave_out = self.interval_octave_selector(x, temperature)
        time_ratio_out_i = self.time_ratio_selector_i(x, temperature)
        time_ratio_out_ii = self.time_ratio_selector_ii(x, temperature)

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


class NuHyperModel(nn.Module):
    def __init__(self, meta_data, lengths):
        super().__init__()

        self.module_list = nn.ModuleList()
        for l in range(0, lengths):
            self.module_list.extend([NuModel(meta_data, l+1).cuda()])

    def prepare(self, sequence_length):
        idx = sequence_length

        added = False

        while idx >= len(self.module_list):
            self.module_list.extend([NuModel(len(self.module_list) + 1).cuda()])

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



class NuDataset(torch.utils.data.Dataset):
    def __init__(self, data, meta_data, sequence_length):
        super().__init__()

        self.data = []
        self.meta_data = meta_data
        self.sequence_length = sequence_length

        for _, piece_tensor in data:
            self.data.append(piece_tensor)

    def __len__(self):
        return len([tensor for tensor in self.data if tensor.shape[0] > self.sequence_length])

    def __getitem__(self, idx):
        pieces = [piece for piece in self.data if piece.shape[0] > self.sequence_length]

        piece = pieces[idx]

        offset = random.randint(0, piece.shape[0] - self.sequence_length - 1)
        inputs = piece[offset:self.sequence_length+offset]

        first = piece[offset]
        last = piece[self.sequence_length+offset-1]
        unknown = piece[self.sequence_length+offset]

        pitch_delta = int((unknown[0] - last[0]).item())
        time_ratio_i = ((unknown[1] - last[1]) / first[2]).item()
        time_ratio_ii = (unknown[2] / first[2]).item()

        direction = np.sign(pitch_delta)
        octave = 12.0
        octaves = 0

        tmp = pitch_delta

        while abs(pitch_delta) >= octave or pitch_delta < 0:
            pitch_delta -= octave * direction
            octaves += 1

        #print(tmp, octaves, pitch_delta, direction)

        interval = abs(pitch_delta)

        interval_pd = probability_distribution(self.meta_data.intervals, interval)
        direction_pd = probability_distribution(self.meta_data.interval_directions, direction)
        octave_pd = probability_distribution(self.meta_data.interval_octaves, octaves)
        time_ratio_i_pd = probability_distribution(self.meta_data.time_ratios_i, time_ratio_i)
        time_ratio_ii_pd = probability_distribution(self.meta_data.time_ratios_ii, time_ratio_ii)

        #print(interval_pd, direction_pd, octave_pd, time_ratio_i_pd, time_ratio_ii_pd)

        labels = torch.hstack((interval_pd, direction_pd, octave_pd, time_ratio_i_pd, time_ratio_ii_pd))

        return inputs, labels

#

class NuDatasetII(torch.utils.data.Dataset):
    def __init__(self, data, meta_data, sequence_length):
        super().__init__()

        self.data = []
        self.meta_data = meta_data
        self.sequence_length = sequence_length

        for _, piece_tensor in data:
            self.data.append(piece_tensor)

    def __len__(self):
        return len([tensor for tensor in self.data if tensor.shape[0] > self.sequence_length])

    def __getitem__(self, idx):
        pieces = [piece for piece in self.data if piece.shape[0] > self.sequence_length]

        piece = pieces[idx]

        offset = random.randint(0, piece.shape[0] - self.sequence_length - 1)
        inputs = piece[offset:self.sequence_length+offset]

        second_to_last = piece[self.sequence_length+offset-2]
        last = piece[self.sequence_length+offset-1]
        unknown = piece[self.sequence_length+offset]

        pitch_delta = int((unknown[0] - last[0]).item())
        time_ratio_i = ((unknown[1] - last[1]) / second_to_last[2]).item()
        time_ratio_ii = (unknown[2] / second_to_last[2]).item()

        direction = np.sign(pitch_delta)
        octave = 12.0
        octaves = 0

        tmp = pitch_delta

        while abs(pitch_delta) >= octave or pitch_delta < 0:
            pitch_delta -= octave * direction
            octaves += 1

        if octaves == 0:
            direction = 0

        #print(tmp, octaves, pitch_delta, direction)

        interval = abs(pitch_delta)

        interval_pd = probability_distribution(self.meta_data.intervals, interval)
        direction_pd = probability_distribution(self.meta_data.interval_directions, direction)
        octave_pd = probability_distribution(self.meta_data.interval_octaves, octaves)
        time_ratio_i_pd = probability_distribution(self.meta_data.time_ratios_i, time_ratio_i)
        time_ratio_ii_pd = probability_distribution(self.meta_data.time_ratios_ii, time_ratio_ii)

        #print(interval_pd, direction_pd, octave_pd, time_ratio_i_pd, time_ratio_ii_pd)

        labels = torch.hstack((interval_pd, direction_pd, octave_pd, time_ratio_i_pd, time_ratio_ii_pd))

        return inputs, labels

class NuMetaDataII:
    def __init__(self):
        self.intervals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.interval_directions = [1, 0, -1]
        self.interval_octaves = [1, 2]

        self.time_ratios_i = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]
        self.time_ratios_ii = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]


class NuSelectorModelII(nn.Module):
    def __init__(self, list_data, sequence_length):
        super().__init__()

        self.list_data = list_data # of type NuMetaData
        self.length = sequence_length

        internal_features = 256

        dropout_p = 1e-2
        outputs = len(self.list_data)

        self.seq = nn.Sequential(
                nn.LazyLinear(internal_features),
                nn.Dropout(dropout_p),
                nn.Sigmoid(),
                ms.Mamba(internal_features, 16, 8, 4),
                nn.Linear(internal_features, outputs),
                #nn.Sigmoid()
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, temperature=1.0):
        out = self.seq(x)

        out = out / temperature
        out = self.softmax(out[:, -1, :])

        return out

    def format(self, x):
        selected_list = []

        p = False
        for batch in x:
            top_values, top_indices = torch.topk(batch, k=int(len(self.list_data) / 2))
        
            chosen_index = random.choices(
                [i for i in top_indices],
                weights=[c for c in top_values],
                k=1
            )[0]

            if not p:
                #print(batch)
                #print(top_values)
                #print(top_indices)
                #print("selected: ", self.list_data[chosen_index], chosen_index.item(), batch[chosen_index].item())
                p = True

            selected = torch.tensor(self.list_data[chosen_index])
            selected_list.append(selected)

        #print(selected_list)

        out = torch.stack(selected_list).unsqueeze(dim=1)

        #print(out)
        return out

class NuModelII(nn.Module):
    def __init__(self, meta_data, sequence_length):
        super().__init__()

        self.meta_data = meta_data
        self.length = sequence_length

        self.interval_selector = NuSelectorModelII(self.meta_data.intervals, self.length)
        self.interval_direction_selector = NuSelectorModelII(self.meta_data.interval_directions, self.length)
        self.interval_octave_selector = NuSelectorModelII(self.meta_data.interval_octaves, self.length)
        self.time_ratio_selector_i = NuSelectorModelII(self.meta_data.time_ratios_i, self.length)
        self.time_ratio_selector_ii = NuSelectorModelII(self.meta_data.time_ratios_ii, self.length)

    def forward(self, x, temperature=1.0):
        next_x = x

        interval_out = self.interval_selector(next_x, temperature)

        #print(x.shape)
        #print(interval_out.shape)
        #print(interval_out.unsqueeze(dim=1).repeat(1, self.length, 1).shape)

        next_x = torch.dstack((x, interval_out.unsqueeze(dim=1).repeat(1, self.length, 1)))

        interval_direction_out = self.interval_direction_selector(next_x, temperature)
        next_x = torch.dstack((
            x,
            interval_out.unsqueeze(dim=1).repeat(1, self.length, 1),
            interval_direction_out.unsqueeze(dim=1).repeat(1, self.length, 1)
        ))

        interval_octave_out = self.interval_octave_selector(next_x, temperature)
        next_x = torch.dstack((
            x,
            interval_out.unsqueeze(dim=1).repeat(1, self.length, 1),
            interval_direction_out.unsqueeze(dim=1).repeat(1, self.length, 1),
            interval_octave_out.unsqueeze(dim=1).repeat(1, self.length, 1)
        ))

        time_ratio_out_i = self.time_ratio_selector_i(next_x, temperature)
        next_x = torch.dstack((
            x,
            interval_out.unsqueeze(dim=1).repeat(1, self.length, 1),
            interval_direction_out.unsqueeze(dim=1).repeat(1, self.length, 1),
            interval_octave_out.unsqueeze(dim=1).repeat(1, self.length, 1),
            time_ratio_out_i.unsqueeze(dim=1).repeat(1, self.length, 1)
        ))

        time_ratio_out_ii = self.time_ratio_selector_ii(next_x, temperature)

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
        second_to_last = x[:, -2, :]
        last = x[:, -1, :].clone()

        m = 0 if not musescore else 0.0010 # weird time correction only needed for MIDIs generated by musescore

        print(transformation)
        print(last)

        last[:, 0] += transformation[:, 0]
        last[:, 1] += (second_to_last[:, 2] + m) * transformation[:, 1]
        last[:, 2] = second_to_last[:, 2] * transformation[:, 2]

        print(last)

        return last.cuda()

