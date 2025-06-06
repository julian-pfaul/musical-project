import os
import pretty_midi

import torch

from tqdm import tqdm

import mupo

print = lambda x: tqdm.write(f"{x}")

load_directory = "./data/midi-classical-music/data/"
save_directory = "./workspace/data/"

if not os.path.exists(save_directory):
    os.mkdir(save_directory)

paths = [os.path.join(load_directory, fn) for fn in os.listdir(load_directory)]
paths = [path for path in paths if not os.path.isdir(path)]

iterator = tqdm(enumerate(paths), desc="paths", leave=False)





for index, path in iterator:
    midi = None

    try:
        midi = pretty_midi.PrettyMIDI(path)
    except Exception as e:
        print(e)

    if midi is not None:
        base_name = os.path.basename(path)
        name, _ = os.path.splitext(base_name)

        midi_data = mupo.encode_midi(midi)

        torch.save(midi_data, os.path.join(save_directory, name+".dat"))





































