import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

import argparse 

import alive_progress as ap

import mupo

import numpy as np

import os

import matplotlib
import matplotlib.pyplot as plt

import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--data-path", type=str, required=True)
    parser.add_argument("-mp", "--model-path", type=str, required=True)

    parser.add_argument("-mt", "--model-type", type=str, nargs="?", choices=["kappa", "lambda"], required=True)

    parser.add_argument("-l", "--length", type=int, required=True)

    args = parser.parse_args()

    data_path = args.data_path
    model_path = args.model_path
    model_type = args.model_type
    length = args.length

    data = torch.load(data_path)

    dataset = None

    match model_type:
        case "kappa":
            dataset = mupo.KappaDataset(data)
        case "lambda":
            dataset = mupo.LambdaDataset(data)

    dataloader = utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    model = None

    if os.path.exists(model_path):
        model = torch.load(model_path, weights_only=False).cuda()
    else:
        match model_type:
            case "kappa":
                model = mupo.KappaModel().cuda()
            case "lambda":
                model = mupo.LambdaHyperModel(128).cuda()

    criterion = nn.L1Loss(reduction="mean")
    slice_criterion = nn.L1Loss(reduction="mean")
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1600, factor=0.8)

    n_epochs = 800
    
    plt.ion()
    
    fig, axes = plt.subplots(2, 2)
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
    ax3l0, = ax3.plot([], [])
    ax3.set_xlabel("Previous N-Epochs")
    ax3.set_ylabel("Mean Loss")

    plt.show()

    losses = [1.0]
    averages = []
    double_averages = []
    means = []
    lrs = []
    for_lengths = []
    
    with ap.alive_bar(n_epochs, title="training", spinner=None) as bar:
        model.train()
        for epoch in range(0, n_epochs):
            if epoch % 50 == 0:
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


            inputs, labels = next(iter(dataloader))
            inputs = inputs.cuda()
            labels = labels.cuda()
    
            #added = model.prepare(dataset.sequence_length)
            #if added:
            #    optimizer = optim.Adam(model.parameters(), lr=lr)
    
            optimizer.zero_grad()

            outputs = model(inputs)
    
            #print(f"outputs: {outputs}")
            #print(f"labels: {labels}")
    
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("NaN detected in outputs")
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                print("NaN detected in labels")
    
            loss = criterion(outputs, labels)
            loss.backward()
    
            optimizer.step()

            first_avg = 2000

            losses.append(loss.item())
            averages.append(np.median(np.array(losses[-first_avg*3:])))
            double_averages.append(np.median(np.array(averages[-first_avg*5:])))
            means.append(np.mean(np.array(losses[-first_avg*3:])))
            lrs.append(scheduler.get_last_lr()[0])

            while len(for_lengths) < dataset.sequence_length:
                for_lengths.append(0)

            #scheduler.step(loss)

            if epoch % 500 == 0:
                model.eval()
                torch.save(model, f"{model_path[:-4]}-epoch-{epoch}.dat")
                model.train()
            #if epoch % 2 == 0:
                #outputs = outputs.cpu().detach()
                #labels = labels.cpu().detach()

                #for i in range(1, dataset.sequence_length):
                #    sliced_outputs = outputs[:, i, :]
                #    sliced_labels = labels[:, i, :]
                # 
                #    sliced_loss = slice_criterion(sliced_outputs, sliced_labels)
                #    for_lengths[i] = sliced_loss.item()

            for_lengths[dataset.sequence_length - 1] = loss.item()
            if epoch % 10 == 0:
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
                ax2l0, = ax2.plot(np.arange(len(lrs)), lrs, "k-")
                ax2.relim()
                ax2.autoscale_view()

                ax3l0.remove()
                ax3l0, = ax3.plot(np.arange(len(for_lengths)), for_lengths, "o")
                ax3.relim()
                ax3.autoscale_view()

                plt.pause(0.01)

                print(f"[{epoch}/{n_epochs}] loss: {loss.item():.4f}, lr: {scheduler.get_last_lr()}, output: {outputs[0].cpu().detach()}, label: {labels[0].cpu().detach()}, seq_len: {dataset.sequence_length:}")

            bar()
    model.eval()
    torch.save(model, model_path)


if __name__ == "__main__":
    main()
