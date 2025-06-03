import numpy as np
from wave_file import wave_write_16bit_mono
from musical_instruments import pipe_organ
from sound_effects import reverb

fs = 44100

note_number = 60 # 21(A0) - 108(C8)
velocity = 100
gate = 1

s = pipe_organ(fs, note_number, velocity, gate)
length_of_s = len(s)

length_of_s_master = int(length_of_s + fs * 3)
s_master = np.zeros(length_of_s_master)

offset = int(fs * 1)
for n in range(length_of_s):
    s_master[offset + n] += s[n]

# reverb
reverb_time = 2
level = 0.1
s_master = reverb(fs, reverb_time, level, s_master)

wave_write_16bit_mono(fs, s_master.copy(), 'pipe_organ' + str(note_number) + '.wav')
