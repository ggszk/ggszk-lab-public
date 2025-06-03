import numpy as np
from wave_file import wave_write_16bit_mono
from oscillator import sawtooth_wave

fs = 44100
duration = 4

length_of_s = int(fs * duration)
vco0 = np.zeros(length_of_s)
vca0 = np.zeros(length_of_s)
vco1 = np.zeros(length_of_s)
vca1 = np.zeros(length_of_s)

for n in range(length_of_s):
    vco0[n] = 440
    vca0[n] = 1
    vco1[n] = 440.5
    vca1[n] = 1

s0 = sawtooth_wave(fs, vco0, duration)
s1 = sawtooth_wave(fs, vco1, duration)

for n in range(length_of_s):
    s0[n] *= vca0[n]
    s1[n] *= vca1[n]

s2 = s0 + s1

for n in range(int(fs * 0.01)):
    s2[n] *= n / (fs * 0.01)
    s2[length_of_s - n - 1] *= n / (fs * 0.01)

length_of_s_master = int(fs * (duration + 2))
s_master = np.zeros(length_of_s_master)

offset = int(fs * 1)
for n in range(length_of_s):
    s_master[offset + n] += s2[n]

master_volume = 0.5
s_master /= np.max(np.abs(s_master))
s_master *= master_volume

wave_write_16bit_mono(fs, s_master.copy(), 'p7_8(output).wav')
