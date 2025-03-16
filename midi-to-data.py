import pretty_midi
import midi_utils

import torch

import os
import sys

midi_directory_path = sys.argv[1]
data_file_path = sys.argv[2]

midi_file_paths_and_names = [
        (os.path.join(midi_directory_path, file_name), file_name) 
        for file_name in os.listdir(midi_directory_path) 
        if file_name.endswith(".mid")
]

data_tensors = []

for index, (midi_file_path, midi_file_name) in enumerate(midi_file_paths_and_names):
    print(f"[{index}/{len(midi_file_paths_and_names)}]", end=" ")

    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
        data_tensor = midi_utils.convert_midi_to_tensor(midi_data)
        data_tensors.append((midi_file_name, data_tensor))
    except Exception as exception:
        print(f"exception {exception}, midi_file_path: {midi_file_path}", end=" ")

    print()

try:
    torch.save(data_tensors, data_file_path)
except Exception as exception:
    rescue_file_path = "rescue.dat"
    torch.save(data_tensors, rescue_file_path)

    print(f"exception {exception}, saved data to a rescue file {rescue_file_path}")

print("done")
