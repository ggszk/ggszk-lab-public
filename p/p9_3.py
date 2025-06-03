import numpy as np
from wave_file import wave_write_16bit_mono
from musical_instruments import tubular_bells

fs = 44100

note_number = 60 # 60(C4) - 77(F5)
velocity = 100
gate = 0.1

s = tubular_bells(fs, note_number, velocity, gate)
length_of_s = len(s)

length_of_s_master = int(length_of_s + fs * 3)
s_master = np.zeros(length_of_s_master)

offset = int(fs * 1)
for n in range(length_of_s):
    s_master[offset + n] += s[n]

wave_write_16bit_mono(fs, s_master.copy(), 'tubular_bells' + str(note_number) + '.wav')
