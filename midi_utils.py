import torch
import pretty_midi

def convert_midi_to_tensor(midi_data):
    notes = []

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append([float(note.pitch), float(note.velocity), float(note.start), float(note.get_duration())])

    notes = sorted(notes, key=lambda x: x[2]) # sort with note.start as criteria

    note_tensors = []

    for note in notes:
        pitch = torch.tensor(note[0])
        velocity = torch.tensor(note[1])
        start = torch.tensor(note[2])
        duration = torch.tensor(note[3])

        note_tensors.append(torch.hstack((start, duration, pitch, velocity)))

    unified_data_tensor = torch.stack(note_tensors)

    return unified_data_tensor

def convert_tensor_to_midi(input_tensor):
    midi_data = pretty_midi.PrettyMIDI()

    instrument = pretty_midi.Instrument(program=0)

    for tensor_note in input_tensor:
        start = tensor_note[0].item()
        duration = tensor_note[1].item()
        pitch = tensor_note[2].item()
        velocity = tensor_note[3].item()

        midi_note = pretty_midi.Note(velocity=int(velocity), pitch=int(pitch), start=float(start), end=float(start+duration))

        instrument.notes.append(midi_note)

    midi_data.instruments.append(instrument)
    
    return midi_data            
