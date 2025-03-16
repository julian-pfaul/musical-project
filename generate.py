import musical_model
import midi_utils
import torch

import os
import sys

model_path = sys.argv[1]
starting_point = sys.argv[2]
length = int(sys.argv[3])

model = musical_model.MusicalModel()

try:
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print("model loaded")
except:
    ...

for piece_name, piece_data in torch.load(starting_point):

    model.eval()

    for step in range(0, length):
        output = model(piece_data)
        piece_data = output

    torch.save([(piece_name[:-4] + "-generated.dat", piece_data)], starting_point[:-4] + piece_name[:-4] + "-generated.dat")


