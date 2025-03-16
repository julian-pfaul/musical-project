import model_utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation

import random

import os
import shutil
import sys

data_path = sys.argv[1]

model_path = None
lr_path = None
iteration_path = None

try:
    model_path = sys.argv[2]
    lr_path = model_path[:-4] + "-lr.dat"
    iteration_path = model_path[:-4] + "-iteration.dat"
except:
    ...

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model_utils.MusicalModel().to(device)
lr = 0.001

if model_path is not None:
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        try:
            lr = torch.load(lr_path)
            print("lr loaded")
        except:
            ...
        print("model loaded")
    except:
        ...

print("loading dataset")
dataset = model_utils.MusicalDataset(data_path, device)
print("dataset loaded")

num_epochs = 2000
iterations_per_epoch = 100

try:
    num_epochs = int(sys.argv[3])
except:
    ...

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.8)

overall_losses = 0
iteration_losses = 0

ani_titles = []
ani_overall_losses = []
ani_iteration_losses = []

for epoch in range(1, num_epochs):
    model.train()

    iteration_losses = 0
    for ipe in range(0, iterations_per_epoch):
        ri = random.randint(0, len(dataset) - 1)
        piece, index, note = dataset[ri]

        outputs = model(piece, index)

        optimizer.zero_grad()

        loss = criterion(outputs, note)
        loss.backward()
        optimizer.step()

        #scheduler.step(loss)

        iteration_losses += loss.item()
        overall_losses += loss.item()

    scheduler.step(iteration_losses)

    ani_titles.append(f"[{epoch}/{num_epochs}] loss: {loss.item():.4f}, overall: {overall_losses/(epoch*iterations_per_epoch):.4f}, iteration: {iteration_losses/iterations_per_epoch:.4f}, lr: {scheduler.get_last_lr()[0]:.8f}")
    ani_overall_losses.append(overall_losses / (epoch * iterations_per_epoch))
    ani_iteration_losses.append(iteration_losses / iterations_per_epoch)

    if epoch % 10 == 0:
        print(f"[{epoch}/{num_epochs}] loss: {loss.item():.4f}, overall: {overall_losses/(epoch*iterations_per_epoch):.4f}, iteration: {iteration_losses/iterations_per_epoch:.4f}, lr: {scheduler.get_last_lr()[0]:.8f}")

if model_path is not None:
    torch.save(model.state_dict(), model_path)
    torch.save(scheduler.get_last_lr()[0], lr_path)

    try:
        iteration = 0

        try:
            iteration = torch.load(iteration_path)
        except:
            ...

        torch.save(iteration+1, iteration_path)

        if not os.path.exists(".anim"):
            os.mkdir(".anim")

        for index, title in enumerate(ani_titles):
            plt.close("all")
            plt.title(title)
            plt.plot(ani_iteration_losses[0:index + 1])
            plt.plot(ani_overall_losses[0:index + 1])
            plt.savefig(f".anim/{index}.png")

        
        frames = []
        fig = plt.figure()  


        for index, _ in enumerate(ani_titles):
            image = mpimg.imread(f".anim/{index}.png")

            plt.axes().set_axis_off()

            frames.append([plt.imshow(image)])

        ani = animation.ArtistAnimation(fig, frames, interval=30, blit=True)
        ani.save(f"animation{iteration}.mp4")

        shutil.rmtree(".anim")

    except Exception as e:
        raise e

    print("model saved")
