import io

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

import alive_progress as ap

from .utils import *

def train(model, dataloader, n_epochs, criterion, optimizer, scheduler = None, out_stream = None, alive_bar = None):
    for epoch in range(0, n_epochs):
        model.train()

        inputs, labels = next(iter(dataloader))

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step(loss)

        model.eval()

        if out_stream is not None:
            if epoch % 50 == 0:
                out_stream.write(f"[{epoch}/{n_epochs}] loss: {loss.item():.8f}, learning_rate: {learning_rate(optimizer):.8f}\n")

        alive_bar()
