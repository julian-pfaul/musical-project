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
    parser.add_argument("-dp", "--data-path", type=str, nargs=1, required=True)

    args = parser.parse_args()

    data_path = args.data_path[0]

    data = torch.load(data_path)
    dataset = mupo.ZetaDataset(data)
    dataloader = utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    model = mupo.ZetaModelII().to("cuda")

    if os.path.exists("workspace/model/zeta-model-ii.dat"):
        model = torch.load("workspace/model/zeta-model-ii.dat", weights_only=False)

    criterion0 = nn.L1Loss()
    #criterion1 = nn.L1Loss()
   # criterion2 = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000000025)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1600, factor=0.8)
    
    n_epochs = 20000
    
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

    losses = [0]
    averages = []
    double_averages = []
    means = []
    lrs = []
    for_lengths = []
    
    with ap.alive_bar(n_epochs, title="training", spinner=None) as bar:
        for epoch in range(0, n_epochs):
            inputs, labels = next(iter(dataloader))
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
    
            model.train()
    
            optimizer.zero_grad()
    
            outputs = model(inputs)
    
            #print(f"outputs: {outputs}")
            #print(f"labels: {labels}")
    
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("NaN detected in outputs")
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                print("NaN detected in labels")
    
            loss0 = criterion0(outputs, labels)
            #loss1 = criterion1(outputs, labels)
            #loss2 = criterion2(outputs, labels)
            loss = loss0 #+ loss1 #+ loss2
            loss.backward()
    
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
            optimizer.step()

            first_avg = 800

            losses.append(loss.item() if loss.item() < first_avg else losses[-1])
            averages.append(np.median(np.array(losses[-first_avg*3:])))
            double_averages.append(np.median(np.array(averages[-first_avg*5:])))
            means.append(np.mean(np.array(losses[-first_avg*3:])))
            lrs.append(scheduler.get_last_lr()[0])

            while len(for_lengths) <= dataset.seq_len:
                for_lengths.append(0)

            for_lengths[dataset.seq_len] = loss.item()

            #scheduler.step(loss)

            if epoch % 500 == 0:
                torch.save(model, f"workspace/model/zeta-model-ii-epoch-{epoch}.dat")
            if epoch % 2 == 0:
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

                print(f"[{epoch}/{n_epochs}] loss: {loss.item():.4f}, lr: {scheduler.get_last_lr()}, output: {outputs[0].cpu().detach().item():.4f}, label: {labels[0].cpu().detach().item():.4f}, seq_len: {dataset.seq_len:}")


            dataset.seq_len = random.randint(1, 2048)

            bar()

    torch.save(model, "workspace/model/zeta-model-ii.dat")


if __name__ == "__main__":
    main()
