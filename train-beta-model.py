import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

import argparse 

import mupo

import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--data-path", type=str, nargs=1, required=True)
    parser.add_argument("-pw", "--pyglet-width", type=int, nargs="?", default=800)
    parser.add_argument("-ph", "--pyglet-height", type=int, nargs="?", default=600)

    args = parser.parse_args()

    data_path = args.data_path[0]
    pg_width = args.pyglet_width
    pg_height = args.pyglet_height

    def launch_animation():


    x = np.linspace(0, 2 * np.pi, 100)
    frequency = 1
    amplitude = 100
    global offset 
    offset = 0

    def update(dt):
        global offset
        offset += 0.1

    @pg_window.event
    def on_draw():
        pg_window.clear()
        pyglet.gl.glBegin(pyglet.gl.GL_LINE_STRIP)
        for i in range(len(x)):
            y = (amplitude * np.sin(frequency * (x[i] + offset))) + (pg_height // 2)
            pyglet.gl.glVertex2f(x[i] * (pg_width / (2 * np.pi)), y)
        pyglet.gl.glEnd()

    pyglet.clock.schedule_interval(update, 1.0/60.0)
    pyglet.app.run()

    data = torch.load(data_path)
    dataset = mupo.BetaDataset(data)
    dataset.seq_len = 10
    dataloader = utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    model = mupo.BetaModel().to("cuda")
    
    criterion0 = nn.L1Loss()
    #criterion1 = nn.L1Loss()
    #criterion2 = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    n_epochs = 200000
    
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if epoch % 20 == 0:
            print(f"[{epoch}/{n_epochs}] loss: {loss.item()}")
            print(f"output-sample: {torch.argmax(outputs[0], dim=1).cpu().detach()}")
            print(f"output-certainty: {torch.max(outputs[0], dim=1)[0].cpu().detach()}")
            print(f"outputs: {outputs[0, 0].cpu().detach()}")
            print(f"label-sample: {torch.argmax(labels[0], dim=1).cpu().detach()}")

            plt.close("all")
            plt.plot(outputs[0, 0].cpu().detach())
            plt.savefig("tmp.png")

    torch.save(model, "workspace/model/beta-model.dat")


if __name__ == "__main__":
    main()
