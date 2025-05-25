import torch
import pretty_midi
import mupo

import os
import sys

data_file_path = sys.argv[1]
midi_directory_path = sys.argv[2]

data_list = torch.load(data_file_path)

for index, (file_name, tensor) in enumerate(data_list):
    print(f"[{index}/{len(data_list)}]", end=" ")

    try:
        midi_file_path = os.path.join(midi_directory_path, file_name)

        midi = mupo.data_to_midi(tensor)
        midi.write(midi_file_path)
    except Exception as exception:
        print(f"file_name: {file_name}, exception: {exception}, {tensor}", end=" ")

    print()

print("done")
