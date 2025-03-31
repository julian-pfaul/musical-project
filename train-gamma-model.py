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
    dataset = mupo.GammaDataset(data)
    dataloader = utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    model = mupo.GammaModel().to("cuda")

    if os.path.exists("workspace/model/gamma-model.dat"):
        model = torch.load("workspace/model/gamma-model.dat", weights_only=False)

    criterion0 = nn.L1Loss()
    #criterion1 = nn.L1Loss()
   # criterion2 = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2000, factor=0.75)
    
    n_epochs = 6000
    
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

    losses = []
    averages = []
    double_averages = []
    means = []
    lrs = []
    
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

            losses.append(loss.item() if loss.item() < 10 else losses[-1])
            averages.append(np.average(np.array(losses[-3000:])))
            double_averages.append(np.average(np.array(averages[-5000:])))
            means.append(np.mean(np.array(losses[-3000:])))

            lrs.append(scheduler.get_last_lr()[0])

            #scheduler.step(double_averages[-1])

            if epoch % 20 == 0:
                ax0l0.remove()
                ax0l0, = ax0.plot(np.arange(len(losses[-1000:])), losses[-1000:], "k-")
                ax0.relim()
                ax0.autoscale_view()

                ax1l0.remove()
                ax1l1.remove()
                ax1l0, = ax1.plot(np.arange(len(averages[-1000:])), averages[-1000:], "k-")
                ax1l1, = ax1.plot(np.arange(len(double_averages[-1000:])), double_averages[-1000:], "k-")
                ax1.relim()
                ax1.autoscale_view()    

                ax2l0.remove()
                ax2l0, = ax2.plot(np.arange(len(lrs[-1000:])), lrs[-1000:], "k-")
                ax2.relim()
                ax2.autoscale_view()

                ax3l0.remove()
                ax3l0, = ax3.plot(np.arange(len(means[-1000:])), means[-1000:], "k-")
                ax3.relim()
                ax3.autoscale_view()

                plt.pause(0.01)

                print(f"[{epoch}/{n_epochs}] loss: {loss.item()}, lr: {scheduler.get_last_lr()}, output: {outputs[0].cpu().detach()}, label: {labels[0].cpu().detach()}")


            dataset.set_sequence_length(random.randint(1, 100))

            bar()

    torch.save(model, "workspace/model/gamma-model.dat")


if __name__ == "__main__":
    main()
