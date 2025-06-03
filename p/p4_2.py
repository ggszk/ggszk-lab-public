import numpy as np
from wave_file import wave_write_16bit_mono
import MIDI_file
from musical_instruments import pipe_organ
from sound_effects import reverb

division, tempo, number_of_track, end_of_track, score = MIDI_file.decode('canon.mid')

tempo = 60 / (tempo / 1000000)
number_of_track = int(number_of_track - 1)
end_of_track = (end_of_track / division) * (60 / tempo)
number_of_note = score.shape[0]

fs = 44100
length_of_s_master = int(fs * (end_of_track + 2))
track = np.zeros((length_of_s_master, number_of_track))
s_master = np.zeros(length_of_s_master)

for i in range(number_of_note):
    j = int(score[i, 0] - 1)
    onset = (score[i, 1] / division) * (60 / tempo)
    note_number = score[i, 2]
    velocity = score[i, 3]
    gate = (score[i, 4] / division) * (60 / tempo)
    s = pipe_organ(fs, note_number, velocity, gate)
    length_of_s = len(s)
    offset = int(fs * onset)
    for n in range(length_of_s):
        track[offset + n, j] += s[n]

for j in range(number_of_track):
    for n in range(length_of_s_master):
        s_master[n] += track[n, j]

reverb_time = 2
level = 0.1
s_master = reverb(fs, reverb_time, level, s_master)

master_volume = 0.5
s_master /= np.max(np.abs(s_master))
s_master *= master_volume

wave_write_16bit_mono(fs, s_master.copy(), 'p4_2(output).wav')
