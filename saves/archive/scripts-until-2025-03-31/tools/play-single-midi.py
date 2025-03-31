import sys
import os

path = sys.argv[1]

soundfont_set = False
soundfont_use_default = False
soundfont = None

if len(sys.argv) == 3:
    if sys.argv[2] == "Default":
        soundfont_use_default = True
    soundfont_set = True
    soundfont = sys.argv[2]

if not os.path.isfile(path):
    raise RuntimeError(f"{sys.argv[1]} is not a valid path")

import pretty_midi
import fluidsynth
import sounddevice

midi_data = pretty_midi.PrettyMIDI(path)

audio_data = None

if soundfont_set:
    if soundfont_use_default:
        audio_data = midi_data.fluidsynth()
    else:
        audio_data = midi_data.fluidsynth(sf2_path=soundfont)
else:
    audio_data = midi_data.synthesize()

sounddevice.play(audio_data)
sounddevice.wait()
