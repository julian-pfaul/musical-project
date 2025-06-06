import torch

import os
import shutil

import pretty_midi

from tqdm import tqdm

import alive_progress as ap

def sequence_length(x):
    if len(x.shape) == 3: # batched
        return x.shape[1]
    elif len(x.shape) == 2: # unbatched
        return x.shape[0]
    else:
        raise RuntimeError(f"Invalid sequence tensor with shape {x.shape}.")


def is_batched(x):
    if len(x.shape) == 3:
        return True
    elif len(x.shape) == 2:
        return False
    else:
        raise RuntimeError(f"Invalid sequence tensor with shape {x.shape}.")


def learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def copy_files(src_dir, dest_dir, filter_substr = None, with_p_bar=False):
    fnames = os.listdir(src_dir)

    if filter_substr is not None:
        fnames = [fn for fn in fnames if fn.find(filter_substr) != -1]

    for fn in tqdm(fnames, leave=False):
        src_path = os.path.join(src_dir, fn)
        dest_path = os.path.join(dest_dir, fn)

        try:
            shutil.copy(src_path, dest_path)
        except Exception:
            ...

    return

def directory_file_names_and_paths(directory):
    return [
        (file_name, os.path.join(directory, file_name))
        for file_name in os.listdir(directory)
    ]


def convert_midi_to_tensor(midi_data):
    notes = []

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append([float(note.pitch), float(note.start), float(note.get_duration())])

    notes = sorted(notes, key=lambda x: x[1]) # sort with note.start as criteria

    note_tensors = []

    for note in notes:
        pitch = torch.tensor(note[0])
        start = torch.tensor(note[1])
        duration = torch.tensor(note[2])

        note_tensors.append(torch.hstack((pitch, start, duration)))

    unified_data_tensor = torch.stack(note_tensors)

    return unified_data_tensor

def convert_tensor_to_midi(input_tensor):
    midi_data = pretty_midi.PrettyMIDI()

    instrument = pretty_midi.Instrument(program=0)

    for tensor_note in input_tensor:
        pitch = tensor_note[0].item()
        start = tensor_note[1].item()
        duration = tensor_note[2].item()

        midi_note = pretty_midi.Note(velocity=int(64), pitch=max(0, min(int(pitch), 127)), start=float(start), end=float(start+duration))

        instrument.notes.append(midi_note)

    midi_data.instruments.append(instrument)

    return midi_data


def encode_midi(midi):
    midi_data = []

    for instrument in midi.instruments:
        instrument_data = []

        for note in instrument.notes:
            start = note.start
            end = note.end
            pitch = note.pitch
            velocity = note.velocity

            note_data = torch.tensor([start, end, pitch, velocity])

            instrument_data.append(note_data)

        instrument_data = torch.stack(instrument_data)

        midi_data.append(instrument_data)

    target_length = max([t.size(0) for t in midi_data])
    midi_data = torch.stack([torch.cat([t, torch.zeros(target_length-t.size(0), t.size(1))], dim=0) for t in midi_data])

    return midi_data


def decode_midi(data):
    midi_data = pretty_midi.PrettyMIDI()

    instrument = pretty_midi.Instrument(program=0)

    for note in data:
        start = note[0].item()
        end = note[1].item()
        pitch = int(note[2].item())
        velocity = int(note[3].item())

        if pitch <= 0:
            break

        #print(start, end, end - start)

        midi_note = pretty_midi.Note(velocity=max(0, min(velocity, 127)), pitch=max(0, min(pitch, 127)), start=float(start), end=float(end))

        instrument.notes.append(midi_note)

    midi_data.instruments.append(instrument)

    return midi_data
