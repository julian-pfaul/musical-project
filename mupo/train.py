import io

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

import alive_progress as ap

from .utils import *

def train(model, dataloader, n_epochs, criterion, optimizer, device = "cpu", scheduler = None, out_stream = None, alive_bar = None, output_interval=10):
    model = model.to(device)

    for epoch in range(0, n_epochs):
        model.train()

        inputs, labels_p, labels_r = next(iter(dataloader))
        inputs = inputs.to(device)
        labels_p = labels_p.to(device)
        labels_r = labels_r.to(device)

        optimizer.zero_grad()

        outputs_p, outputs_r = model(inputs)

        loss0 = criterion(outputs_p, labels_p)
        loss1 = criterion(outputs_r, labels_r)
        loss = loss0 * 1.0 + loss1 * 12.0

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step(loss)

        model.eval()

        if out_stream is not None:
            if epoch % output_interval == 0:
                out_stream.write(f"[{epoch}/{n_epochs}] loss: {loss.item():.8f}, learning_rate: {learning_rate(optimizer):.8f}\n")

        if alive_bar is not None:
            alive_bar()
