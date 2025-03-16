import musical_model
import musical_dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation

import random
import io
import PIL

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

model = musical_model.MusicalModel().to(device)
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
dataset = musical_dataset.MusicalDataset(data_path, device)
print("dataset loaded")

num_epochs = 2000
iterations_per_epoch = 100

try:
    num_epochs = int(sys.argv[3])
except:
    ...

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=800, factor=0.9)

overall_losses = 0

ani_titles = []
ani_overall_losses = []
ani_iteration_losses = []

for epoch in range(1, num_epochs+1):
    model.train()

    ri = random.randint(0, len(dataset) - 1)
    
    for index in range(0, 12):
        piece, note = dataset[ri]

        output = model(piece)

        optimizer.zero_grad()

        loss = criterion(output, torch.vstack((piece, note.unsqueeze(dim=0))))
        loss.backward()
        optimizer.step()

    #scheduler.step(loss)

        overall_losses += loss.item()

    scheduler.step(overall_losses/epoch/12)

    model.eval()

    ani_titles.append(f"[{epoch}/{num_epochs}] loss: {loss.item():.4f}, overall: {overall_losses/epoch:.4f}, lr: {scheduler.get_last_lr()[0]:.8f}")
    ani_overall_losses.append(overall_losses / epoch)
    ani_iteration_losses.append(loss.item())

    if epoch % 10 == 0:
        print(f"[{epoch}/{num_epochs}] loss: {loss.item():.4f}, overall: {overall_losses/epoch:.4f}, lr: {scheduler.get_last_lr()[0]:.8f}")

if model_path is not None:
    torch.save(model.state_dict(), model_path)
    torch.save(scheduler.get_last_lr()[0], lr_path)

    print("model saved")

    try:
        iteration = 0

        try:
            iteration = torch.load(iteration_path)
        except:
            ...

        torch.save(iteration+1, iteration_path)
    except:
        ...

