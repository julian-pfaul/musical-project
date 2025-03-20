import os
import shutil

import mupo

import torch
import pretty_midi

import alive_progress as ap


if not os.path.exists("workspace"):
    os.mkdir("workspace")

if not os.path.exists("workspace/.tmp"):
    os.mkdir("workspace/.tmp")

if os.path.exists("data/midi-classical-music/data"):
    mupo.copy_files("data/midi-classical-music/data", "workspace/.tmp")   
    
    contents = []

    names_and_paths = mupo.directory_file_names_and_paths("workspace/.tmp")

    with ap.alive_bar(len(names_and_paths), spinner=None) as bar:
        for index, (file_name, file_path) in enumerate(names_and_paths):
            try:
                contents.append((file_name, mupo.midi_to_data(pretty_midi.PrettyMIDI(file_path))))
            except Exception as exception:
                print(f"exception '{exception}' raised for {file_path}")

            bar()

    if not os.path.exists("workspace/dataset"):
        os.mkdir("workspace/dataset")

    torch.save(contents, "workspace/dataset/large-dataset.dat")

    shutil.rmtree("workspace/.tmp")
