import torch
import pretty_midi

def midi_to_data(midi_data):
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

def data_to_midi(data):
    midi_data = pretty_midi.PrettyMIDI()

    instrument = pretty_midi.Instrument(program=0)

    for tensor_note in data:
        pitch = tensor_note[0].item()
        start = tensor_note[1].item()
        duration = tensor_note[2].item()

        DEFAULT_MIDI_VELOCITY = 64

        midi_note = pretty_midi.Note(
            velocity=int(DEFAULT_MIDI_VELOCITY), 
            pitch=int(pitch), 
            start=float(start), 
            end=float(start+duration)
        )

        instrument.notes.append(midi_note)

    midi_data.instruments.append(instrument)
    
    return midi_data            
