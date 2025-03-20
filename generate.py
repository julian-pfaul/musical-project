import musical_model
import midi_utils
import torch

import os
import sys

model_path = sys.argv[1]
starting_point = sys.argv[2]
length = int(sys.argv[3])

device = torch.device("cuda")

model = musical_model.MusicalHyperModel().to(device)

try:
    model = torch.load(model_path, weights_only=False)
    print("model loaded")
except:
    ...

piece_name, piece_data = torch.load(starting_point)[0]

model.eval()

for step in range(0, length):
    output = model(piece_data.to(device))

    last = piece_data[-1]
    
    print(last)
    print(output)
    output[1] = last[1] + output[1]

    print(output)
    piece_data = torch.vstack((piece_data.cpu(), output.cpu()))
    print(piece_data)

torch.save([(piece_name[:-4] + "-generated.mid", piece_data)], starting_point[:-4] + "-" + piece_name[:-4] + "-generated.dat")


