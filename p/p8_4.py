import numpy as np
from wave_file import wave_read_16bit_mono
from wave_file import wave_write_16bit_stereo
from sound_effects import reverb

number_of_track = 10

fs = 44100
length_of_s_master = int(fs * 28)
track = np.zeros((length_of_s_master, number_of_track))
s_master = np.zeros((length_of_s_master, 2))

fs, s = wave_read_16bit_mono('track1.wav')
track[:, 0] = s

fs, s = wave_read_16bit_mono('track2.wav')
track[:, 1] = s

fs, s = wave_read_16bit_mono('track3.wav')
track[:, 2] = s

fs, s = wave_read_16bit_mono('track4.wav')
track[:, 3] = s

fs, s = wave_read_16bit_mono('track5.wav')
track[:, 4] = s

fs, s = wave_read_16bit_mono('track6.wav')
track[:, 5] = s

fs, s = wave_read_16bit_mono('track7.wav')
track[:, 6] = s

fs, s = wave_read_16bit_mono('track8.wav')
track[:, 7] = s

fs, s = wave_read_16bit_mono('track9.wav')
track[:, 8] = s

fs, s = wave_read_16bit_mono('track10.wav')
track[:, 9] = s

v = np.array([0.7, 0.5, 0.4, 0.9, 0.7, 0.4, 0.4, 0.4, 0.4, 1.0])
p = np.array([0.5, 0.5, 0.6, 0.4, 0.5, 0.7, 0.8, 0.3, 0.2, 0.5])

for i in range(number_of_track):
    s_master[:, 0] += track[:, i] * v[i] * np.cos(np.pi * p[i] / 2)
    s_master[:, 1] += track[:, i] * v[i] * np.sin(np.pi * p[i] / 2)

reverb_time = 2
level = 0.1
s_master[:, 0] = reverb(fs, reverb_time, level, s_master[:, 0])

reverb_time = 2
level = 0.1
s_master[:, 1] = reverb(fs, reverb_time, level, s_master[:, 1])

master_volume = 0.9
s_master /= np.max(np.abs(s_master))
s_master *= master_volume

wave_write_16bit_stereo(fs, s_master.copy(), 'p8_4(output).wav')
