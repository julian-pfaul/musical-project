import musical_model
import musical_dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils as utils

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
n_inputs = int(sys.argv[4])

model_path = None
lr_path = None
iteration_path = None

try:
    model_path = sys.argv[2]
    lr_path = model_path[:-4] + f"-lr-{n_inputs}.dat"
    iteration_path = model_path[:-4] + f"-iteration-{n_inputs}.dat"
except:
    ...

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = musical_model.MusicalHyperModel().to(device)
lr = 0.001

if model_path is not None:
    try:
        model = torch.load(model_path, weights_only=False).to(device)
    except:
        ...

    if str(n_inputs) not in model.m_sub_modules:
        print("trying reload hack")
        model.eval()
        out = model(torch.rand(1, n_inputs, 4).to(device))

        torch.save(model, model_path)

        model = torch.load(model_path, weights_only=False).to(device)
        print("reload hack")

    try:
        lr = torch.load(lr_path)
        print("lr loaded")
    except:
        ...

    print("model loaded")

print("loading dataset")
dataset = musical_dataset.MusicalDataset(data_path, n_inputs, device)
dataloader = utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
print("dataset loaded")

num_epochs = 2000
iterations_per_epoch = 100

try:
    num_epochs = int(sys.argv[3])
except:
    ...

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.8)

for epoch in range(1, num_epochs+1):
    model.train()

    train_features, train_labels = next(iter(dataloader))

    outputs = model(train_features)

    optimizer.zero_grad()

    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()

    scheduler.step(loss)

    model.eval()

    if epoch % 200 == 0 or epoch <= 200:
        print(f"[{epoch}/{num_epochs}] loss: {loss.item():.6f}, lr: {scheduler.get_last_lr()[0]:.8f}")

if model_path is not None:
    torch.save(model, model_path)
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

