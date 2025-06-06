import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

import argparse 

import alive_progress as ap

from tqdm import tqdm

import mupo

import numpy as np

import os

import matplotlib
import matplotlib.pyplot as plt

import random

import pypianoroll as pr

def average_effective_lr(optimizer, original_lr):
    total_lr = 0
    count = 0
    
    for state in optimizer.state.values():
        if 'step' in state and 'exp_avg' in state and 'exp_avg_sq' in state:
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            effective_lr = original_lr / (torch.sqrt(exp_avg_sq) + 1e-8)
            total_lr += effective_lr.mean().item()
            count += 1

    average_effective_lr = total_lr / count if count > 0 else 0
    
    return average_effective_lr

print = lambda x: tqdm.write(f"{x}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--data-path", type=str, required=True)
    parser.add_argument("-mp", "--model-path", type=str, required=True)

    parser.add_argument("-mt", "--model-type", type=str, nargs="?", choices=["kappa", "lambda", "lambda-ii", "mu", "nu", "omicron", "pi", "rho", "sigma", "tau", "ypsilon"], required=True)
    parser.add_argument("-sv", "--sub-version", type=str, nargs="?", choices=["i", "ii", "iii"], default="i")
    parser.add_argument("--mode", type=str, nargs="?")

    parser.add_argument("-l", "--length", type=int, required=True)

    parser.add_argument("-ai", "--animation-interval", type=int, nargs="?", default=10)

    parser.add_argument("-t", "--temperature", type=float, nargs="?", default=1.0)

    parser.add_argument("-e", "--epochs", type=int, nargs="?", default=8000)
    parser.add_argument("-lr", "--learning-rate", type=float, nargs="?", default=0.01)
    parser.add_argument("-bs", "--batch-size", type=int, nargs="?", default=64)

    parser.add_argument("-v", "--view-size", type=int, nargs="?", default=200)

    parser.add_argument("--no-gui", action="store_true")

    args = parser.parse_args()

    data_path = args.data_path
    model_path = args.model_path
    model_type = args.model_type
    sub_version = args.sub_version
    length = args.length
    temperature = args.temperature
    epochs = args.epochs
    learning_rate = args.learning_rate
    view_size = args.view_size
    batch_size = args.batch_size
    mode = args.mode
    no_gui = args.no_gui

    hidden_size = 512
    num_layers = 6

    animation_interval = args.animation_interval

    torch.set_default_dtype(torch.float)

    data = torch.load(data_path, weights_only=False)

    dataset = None

    meta_data = None

    if model_type == "mu":
        meta_data = mupo.MuMetaData()
    elif model_type == "nu":
        match sub_version:
            case "i":
                meta_data = mupo.NuMetaData()
            case "ii":
                meta_data = mupo.NuMetaDataII()
            case "iii":
                meta_data = mupo.NuMetaDataIII()

    match model_type:
        case "kappa":
            dataset = mupo.KappaDataset(data)
        case "lambda":
            dataset = mupo.LambdaDataset(data)
        case "lambda-ii":
            dataset = mupo.LambdaDataset(data)
        case "mu":
            dataset = mupo.MuDataset(data, meta_data)
        case "nu":
            match sub_version:
                case "i":
                    dataset = mupo.NuDataset(data, meta_data, length)
                case "ii":
                    dataset = mupo.NuDatasetII(data, meta_data, length)
                case "iii":
                    dataset = mupo.NuDatasetIII(data, meta_data, length)
        case "omicron":
            dataset = mupo.OmicronDataset(data)
        case "pi":
            dataset = mupo.PiDataset(data)
            dataset.seq_len = length
        case "rho":
            dataset = mupo.PiDataset(data)
            dataset.seq_len = length
        case "sigma":
            dataset = mupo.SigmaDataset(data, length)

            if sub_version == "iii":
                dataset = mupo.SigmaDatasetIII(data, length)
        case "tau":
            dataset = mupo.tau.TrackDataset(data, length)
        case "ypsilon":
            dataset = mupo.ypsilon.TrackDataset(data, length)


    dataloader = utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = None

    if os.path.exists(model_path):
        model = torch.load(model_path, weights_only=False).cuda()
    else:
        match model_type:
            case "kappa":
                model = mupo.KappaModel().cuda()
            case "lambda":
                model = mupo.LambdaHyperModel(128).cuda()
            case "lambda-ii":
                model = mupo.LambdaHyperModelII(128).cuda()
            case "mu":
                model = mupo.MuHyperModel(meta_data, 128).cuda()
            case "nu":
                match sub_version:
                    case "i":
                        model = mupo.NuModel(meta_data, length).cuda()
                    case "ii":
                          model = mupo.NuModelII(meta_data, length).cuda()
                    case "iii":
                          model = mupo.NuModelIII(meta_data, length).cuda()
            case "omicron":
                model = mupo.OmicronModel().cuda()
            case "pi":
                model = mupo.DeepResidualGRU(3, hidden_size, 3, num_layers).cuda()
            case "rho":
                model = mupo.Rho().cuda()
            case "sigma":
                match sub_version:
                    case "i":
                        model = mupo.SigmaModel().cuda()
                    case "ii":
                        model = mupo.SigmaModelII().cuda()
                    case "iii":
                        model = mupo.SigmaModelIII().cuda()
            case "tau":
                model = mupo.tau.Model().cuda()
            case "ypsilon":
                model = mupo.ypsilon.Model().cuda()


    criterion = nn.L1Loss(reduction="mean")
    l1loss = nn.MSELoss()

    if model_type == "sigma":
        criterion = nn.CrossEntropyLoss(reduction="mean")

    l1_loss_fn = nn.L1Loss(reduction="mean")
    mse_loss_fn = nn.MSELoss(reduction="mean")
    cross_entropy_loss_fn = nn.CrossEntropyLoss(reduction="mean")

    slice_criterion = nn.L1Loss(reduction="mean")
    original_lr = learning_rate
    optimizer = optim.Adam(model.parameters(), lr=original_lr)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5)

    n_epochs = epochs
    
    plt.ion()
    
    fig, axes = plt.subplots(2, 4 if model_type == "tau" or model_type == "ypsilon" else 3)
    ax0 = axes[0][0]
    ax0l0, = ax0.plot([], [])
    ax0l1, = ax0.plot([], [])
    ax0.set_xlabel("Previous N-Epoch")
    ax0.set_ylabel("Loss")
    ax0.set_title("Training Loss")

    ax1 = axes[1][1]
    ax1l0, = ax1.plot([], [])
    ax1l1, = ax1.plot([], [])
    ax1.set_xlabel("Previous N-Epochs")
    ax1.set_ylabel("Average Loss")

    ax2 = axes[1][0]
    ax2l0, = ax2.plot([], [])
    ax2.set_xlabel("Previous N-Epochs")
    ax2.set_ylabel("Learning Rate")

    ax3 = axes[0][1]

    ax4 = axes[0][2]
    ax4l0, = ax4.plot([], [])
    ax4.set_xlabel("Previous N-Epoch")
    ax4.set_ylabel("Loss")

    ax5 = axes[1][2]
    ax5l0, = ax5.plot([], [])
    ax5.set_xlabel("Previous N-Epoch")
    ax5.set_ylabel("Loss")

    if model_type == "tau" or model_type == "ypsilon":
        ax6 = axes[0][3]
        ax6l0, = ax6.plot([], [])
        ax6.set_xlabel("Previous N-Epoch")
        ax6.set_ylabel("Loss")

        ax7 = axes[1][3]
        ax7l0, = ax7.plot([], [])
        ax7.set_xlabel("Previous N-Epoch")
        ax7.set_ylabel("Loss")

    if model_type == "tau" or model_type == "ypsilon":
        ax4.set_ylabel("Start Loss")
        ax5.set_ylabel("End Loss")
        ax6.set_ylabel("Pitch Loss")
        ax7.set_ylabel("Velocity Loss")

    plt.show()

    losses = []
    l1l_losses = []

    s_losses = []
    e_losses = []
    p_losses = []
    v_losses = []

    averages = []
    double_averages = []
    means = []
    lrs = []
    for_lengths = []
    
    ax3pr = None

    reference = None

    max_gen_len = 200

    for epoch in tqdm(range(0, n_epochs)):
        model.train()

        if epoch % 400 == 0:
            for param in model.parameters():
                if True in torch.isnan(param.data):
                    model.module_list[dataset.sequence_length-1] = mupo.LambdaModel(dataset.sequence_length).cuda()
                    print("RESET")

        #dataset.sequence_length = random.randint(2, 24)
        #dataset.sequence_length %= 1024
        #dataset.sequence_length += 1
        dataset.sequence_length = length

        #if dataset.sequence_length <= 1:
        #    dataset.sequence_length = 4092
        #else:
        #    dataset.sequence_length -= 1

        batch_iterator = tqdm(dataloader, leave=False)

        for index, batch in enumerate(batch_iterator):
            model.train()

            inputs = None
            labels = None
            extra_labels = None
            s_labels = None
            e_labels = None
            p_labels = None
            v_labels = None


            if model_type == "sigma" and sub_version == "iii":
                inputs, labels, extra_labels = batch
                extra_labels = extra_labels.cuda()
                inputs = inputs.cuda()
                labels = labels.cuda()
            elif model_type == "tau":
                inputs, s_labels, e_labels, p_labels, v_labels = batch
                inputs = inputs.cuda()
                s_labels = s_labels[:, :max_gen_len, :].cuda()
                e_labels = e_labels[:, :max_gen_len, :].cuda()
                p_labels = p_labels[:, :max_gen_len, :].cuda()
                v_labels = v_labels[:, :max_gen_len, :].cuda()
            elif model_type == "ypsilon":
                inputs, s_labels, e_labels, p_labels, v_labels = batch
                inputs = inputs.cuda()
                s_labels = s_labels[:, :max_gen_len, :].cuda()
                e_labels = e_labels[:, :max_gen_len, :].cuda()
                p_labels = p_labels[:, :max_gen_len, :].cuda()
                v_labels = v_labels[:, :max_gen_len, :].cuda()
            else:
                inputs, labels = batch
                inputs = inputs.cuda()
                labels = labels.cuda()

            if epoch == 0 and index == 0:
                reference = inputs[0].unsqueeze(dim=0)

                if model_type == "ypsilon":
                    reference = batch

            #print(f"{inputs}, {labels}")

            #added = model.prepare(dataset.sequence_length)
            #if added:
            #    optimizer = optim.Adam(model.parameters(), lr=lr)

            outputs = None
            extra = None

            s_outputs = None
            e_outputs = None
            p_outputs = None
            v_outputs = None

            if model_type == "nu" and sub_version == "ii":
                outputs = model(inputs, temperature)
            elif model_type == "sigma" and sub_version == "iii":
                outputs, extra = model(inputs)
            elif model_type == "tau":
                s_outputs, e_outputs, p_outputs, v_outputs = model(inputs, max_gen_len)
                #print(s_outputs.shape)
                #print(s_labels.shape)
            elif model_type == "ypsilon":
                p_outputs = model(inputs, "pitch", pitches=None, starts=None, ends=None, gen_length=max_gen_len)
                s_outputs = model(inputs, "start", pitches=p_labels, starts=None, ends=None, gen_length=max_gen_len)
                e_outputs = model(inputs, "end", pitches=p_labels, starts=s_labels, ends=None, gen_length=max_gen_len)
                v_outputs = model(inputs, "velocity", pitches=p_labels, starts=s_labels, ends=e_labels, gen_length=max_gen_len)
                # match mode:
                #     case "pitch":
                #         p_outputs = model(inputs, "pitch", max_gen_len)
                #     case "start":
                #         s_outputs = model(inputs, "start", pitches=p_labels, max_gen_len)
                #     case "end":
                #         e_outputs = model(inputs, "end", pitches=p_labels, starts=s_labels, max_gen_len)
                #     case "velocity":
                #         v_outputs = model(inputs, "velocity", pitches=p_labels, starts=s_labels, ends=e_labels, max_gen_len)
            else:
                outputs = model(inputs)

            #print(f"outputs: {outputs}")
            #print(f"labels: {labels}")
            # if model_type != "tau":
            #     if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            #         print("NaN detected in outputs")
            #     if torch.isnan(labels).any() or torch.isinf(labels).any():
            #         print("NaN detected in labels")

            l1l = None

            s_loss = None
            e_loss = None
            p_loss = None
            v_loss = None

            if model_type == "sigma":
                if sub_version == "i" or sub_version == "ii":
                    loss = criterion(outputs.view(-1, 128), mupo.sigma_to_values(labels).view(-1))
                elif sub_version == "iii":
                    loss = criterion(outputs.view(-1, 128), mupo.sigma_to_values(labels).view(-1))
                    l1l = l1loss(extra, extra_labels)
                    loss += l1l
            elif model_type == "tau":
                s_loss = l1_loss_fn(s_outputs, s_labels)
                e_loss = l1_loss_fn(e_outputs, e_labels)
                p_loss = cross_entropy_loss_fn(p_outputs.float().view(-1, 128), p_labels.long().view(-1))
                v_loss = l1_loss_fn(v_outputs, v_labels)

                loss = s_loss + e_loss + p_loss + v_loss
            elif model_type == "ypsilon":
                s_loss = l1_loss_fn(s_outputs, s_labels)
                e_loss = l1_loss_fn(e_outputs, e_labels)
                p_loss = cross_entropy_loss_fn(p_outputs.float().view(-1, 128), p_labels.long().view(-1))
                v_loss = l1_loss_fn(v_outputs, v_labels)

                loss = s_loss + e_loss + p_loss + v_loss
            else:
                loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if no_gui:
                if model_type == "ypsilon" and index % 5 == 0:
                    print(f"{loss.item()} {s_loss.item()} {e_loss.item()} {p_loss.item()} {v_loss.item()}")
                continue

            first_avg = view_size

            if model_type == "sigma" and sub_version == "iii":
                losses.append(loss.item() - l1l.item())
                l1l_losses.append(l1l.item())
            elif model_type == "tau":
                s_losses.append(s_loss.item())
                e_losses.append(e_loss.item())
                p_losses.append(p_loss.item())
                v_losses.append(v_loss.item())
                losses.append(loss.item())
            elif model_type == "ypsilon":
                s_losses.append(s_loss.item())
                e_losses.append(e_loss.item())
                p_losses.append(p_loss.item())
                v_losses.append(v_loss.item())
                losses.append(loss.item())
            else:
                losses.append(loss.item())

            averages.append(np.median(np.array(losses[-first_avg*3:])))
            double_averages.append(np.median(np.array(averages[-first_avg*5:])))
            means.append(np.mean(np.array(losses[-first_avg*3:])))

            lr = average_effective_lr(optimizer, original_lr)
            #lr = scheduler.get_last_lr()[0]
            lrs.append(lr)

            while len(for_lengths) < dataset.sequence_length:
                for_lengths.append(0)

            #scheduler.step(loss)
            #if index % 2 == 0:
                #outputs = outputs.cpu().detach()
                #labels = labels.cpu().detach()

                #for i in range(1, dataset.sequence_length):
                #    sliced_outputs = outputs[:, i, :]
                #    sliced_labels = labels[:, i, :]
                #
                #    sliced_loss = slice_criterion(sliced_outputs, sliced_labels)
                #    for_lengths[i] = sliced_loss.item()

            for_lengths[dataset.sequence_length - 1] = loss.item()

            if index % animation_interval == 0:
                ax0l0.remove()
                ax0l0, = ax0.plot(np.arange(len(losses[-first_avg:])), losses[-first_avg:], "k-")
                ax0.relim()
                ax0.autoscale_view()

                ax1l0.remove()
                ax1l1.remove()
                ax1l0, = ax1.plot(np.arange(len(averages[-first_avg:])), averages[-first_avg:], "k-")
                ax1l1, = ax1.plot(np.arange(len(double_averages[-first_avg:])), double_averages[-first_avg:], "k-")
                ax1.relim()
                ax1.autoscale_view()

                ax2l0.remove()
                ax2l0, = ax2.plot(np.arange(len(lrs[-first_avg*8:])), lrs[-first_avg*8:], "k-")
                ax2.relim()
                ax2.autoscale_view()

                if model_type == "sigma" and sub_version == "iii":
                    ax5l0.remove()
                    ax5l0, = ax5.plot(np.arange(len(l1l_losses[-first_avg:])), l1l_losses[-first_avg:], "k-")
                    ax5.relim()
                    ax5.autoscale_view()

                if model_type == "sigma":
                    r_outputs = None
                    r_extra = None

                    with torch.no_grad():
                        r_outputs, r_extra = model(reference.unsqueeze(dim=0))

                    piece = mupo.sigma_to_values(r_outputs[0]).cpu().detach()
                    p_len = piece.size(0)
                    starts = torch.arange(0, p_len / 5, 0.2)
                    durations = torch.tensor([0.2]).expand(p_len)

                    if sub_version == "iii":
                        r_extra = r_extra[0].cpu().detach()
                        starts = torch.relu(r_extra[:, 0])
                        durations = torch.relu(r_extra[:, 1])

                    piece = torch.stack((piece, starts, durations)).permute((1, 0))

                    #print(piece.shape)

                    #print(piece.shape)
                    try:
                        midi_data = mupo.convert_tensor_to_midi(piece[:100])

                        piano_roll = pr.from_pretty_midi(midi_data)

                        ax3.cla()
                        pr.plot_track(piano_roll.tracks[0], ax3)
                        ax3.relim()
                        ax3.autoscale_view()
                    except:
                        ...

                    #

                    if False:
                        piece_data = reference[:32, :].unsqueeze(dim=0)

                        while piece_data.size(1) < 100:
                            model_output = None
                            r_extra = None

                            with torch.no_grad():
                                if sub_version == "iii":
                                    model_output, r_extra = model(piece_data)
                                else:
                                    model_output = model(piece_data)

                            model_output = model_output[0].cpu().detach()
                            values = mupo.sigma_to_values(model_output)

                            pitch = values[-1]
                            start = piece_data[0, -1, 1] + 0.2
                            start = start.cpu().detach()
                            duration = torch.tensor(0.2)

                            if sub_version == "iii":
                                r_extra = r_extra[0]
                                start = r_extra[-1, 0]
                                start = torch.relu(start.cpu().detach())
                                duration = torch.relu(r_extra[-1, 1].cpu().detach())

                            #print(pitch)
                            #print(start)
                            #print(duration)

                            model_output = torch.stack((pitch, start, duration)).cuda()

                            piece_data = piece_data.squeeze()
                            piece_data = torch.vstack((piece_data, model_output))
                            #print(piece_data.shape)

                            piece_data = piece_data.unsqueeze(dim=0)


                        piece_data = piece_data.squeeze()

                        midi_data = mupo.convert_tensor_to_midi(piece_data[:100])

                        if epoch % 10 == 0:
                            try:
                                midi_data.write(f"workspace/gen-epoch-{epoch}.mid")
                            except:
                                ...

                        try:
                            piano_roll = pr.from_pretty_midi(midi_data)

                            ax4.cla()
                            pr.plot_track(piano_roll.tracks[0], ax4)
                            ax4.relim()
                            ax4.autoscale_view()
                        except:
                            ...

                if model_type == "tau" or model_type == "ypsilon":
                    ax4l0.remove()
                    ax4l0, = ax4.plot(np.arange(len(s_losses[-first_avg:])), s_losses[-first_avg:], "k-")
                    ax4.relim()
                    ax4.autoscale_view()

                    ax5l0.remove()
                    ax5l0, = ax5.plot(np.arange(len(e_losses[-first_avg:])), e_losses[-first_avg:], "k-")
                    ax5.relim()
                    ax5.autoscale_view()

                    ax6l0.remove()
                    ax6l0, = ax6.plot(np.arange(len(p_losses[-first_avg:])), p_losses[-first_avg:], "k-")
                    ax6.relim()
                    ax6.autoscale_view()

                    ax7l0.remove()
                    ax7l0, = ax7.plot(np.arange(len(v_losses[-first_avg:])), v_losses[-first_avg:], "k-")
                    ax7.relim()
                    ax7.autoscale_view()

                if model_type == "tau" or model_type == "ypsilon":
                    outputs = None

                    if model_type == "tau":
                        with torch.no_grad():
                            s_outputs, e_outputs, p_outputs, v_outputs = model(reference, max_gen_len)

                        inputs = reference.squeeze()

                        start = s_outputs
                        end = e_outputs
                        pitch = p_outputs
                        velocity = v_outputs

                        _, pitch = torch.max(pitch, dim=-1)

                        pitch = pitch.unsqueeze(dim=-1)

                        outputs = torch.dstack([start, end, pitch, velocity]).squeeze()
                    elif model_type == "ypsilon":
                        inputs, s_labels, e_labels, p_labels, v_labels = reference
                        inputs = inputs[0].cuda().unsqueeze(dim=0)
                        s_labels = s_labels[0, :max_gen_len, :].cuda().unsqueeze(dim=0)
                        e_labels = e_labels[0, :max_gen_len, :].cuda().unsqueeze(dim=0)
                        p_labels = p_labels[0, :max_gen_len, :].cuda().unsqueeze(dim=0)
                        v_labels = v_labels[0, :max_gen_len, :].cuda().unsqueeze(dim=0)

                        with torch.no_grad():
                            p_outputs = model(inputs, "pitch", pitches=None, starts=None, ends=None, gen_length=max_gen_len)

                        pitch = p_outputs
                        _, pitch = torch.max(pitch, dim=-1)
                        pitch = pitch.unsqueeze(dim=-1)

                        with torch.no_grad():
                            s_outputs = model(inputs, "start", pitches=p_labels, starts=None, ends=None, gen_length=max_gen_len)

                        start = s_outputs

                        with torch.no_grad():
                            e_outputs = model(inputs, "end", pitches=p_labels, starts=s_labels, ends=None, gen_length=max_gen_len)

                        end = e_outputs

                        with torch.no_grad():
                            v_outputs = model(inputs, "velocity", pitches=p_labels, starts=s_labels, ends=e_labels, gen_length=max_gen_len)

                        velocity = v_outputs

                        inputs = inputs.squeeze()
                        outputs = torch.dstack([start, end, pitch, velocity]).squeeze()

                    assert outputs is not None

                    #print(inputs.shape)
                    #print(outputs.shape)

                    piece = torch.vstack([inputs, outputs])

                    #print(piece.shape)

                    midi_data = mupo.decode_midi(piece)

                    try:
                        piano_roll = pr.from_pretty_midi(midi_data)

                        ax3.cla()
                        pr.plot_track(piano_roll.tracks[0], ax3)
                        ax3.relim()
                        ax3.autoscale_view()
                    except Exception as e:
                        print(e)
                        ...


                plt.pause(0.01)

                if model_type == "mu" or model_type == "nu":
                    outputs = model.format(outputs.detach())
                    labels = model.format(labels.detach())

                if model_type == "sigma":
                    outputs = mupo.sigma_to_values(outputs[0])
                    labels = mupo.sigma_to_values(labels[0])

                #print(f"{outputs[-1]}"

                #print(f"[{epoch}/{n_epochs}/{index}] loss: {loss.item():.4f}, lr: {lr}, output: {outputs[0][-1].cpu().detach()}, label: {labels[0][-1].cpu().detach()}, seq_len: {dataset.sequence_length:}")
                if model_type == "tau" or model_type == "ypsilon":
                    continue

                print(f"[{epoch}/{n_epochs}/{index}] loss: {loss.item():.4f}, lr: {lr}, output: {outputs[-5:].cpu().detach()}, label: {labels[-5:].cpu().detach()}, seq_len: {dataset.sequence_length:}")

        model.eval()
        torch.save(model, f"{model_path[:-4]}-epoch-{epoch}.dat")
        model.train()

    model.eval()
    torch.save(model, model_path)


if __name__ == "__main__":
    main()
