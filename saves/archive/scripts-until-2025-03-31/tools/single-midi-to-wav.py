import sys

source = sys.argv[1]
destination = sys.argv[2]

soundfont_set = False
soundfont = None

if len(sys.argv) == 4:
    soundfont_set = True
    soundfont = sys.argv[3]

import pretty_midi
import soundfile
import fluidsynth
import sounddevice

sample_rate = 44100

midi_data = pretty_midi.PrettyMIDI(source)

audio_data = None

if soundfont_set:
    audio_data = midi_data.fluidsynth(sample_rate, sf2_path=soundfont)
else:
    audio_data = midi_data.synthesize(sample_rate)

soundfile.write(destination, audio_data, sample_rate)
