import numpy as np
from wave_file import wave_write_16bit_mono
from oscillator import sawtooth_wave

fs = 44100
duration = 4

depth = 50
rate = 1

length_of_s = int(fs * duration)
vco = np.zeros(length_of_s)
vca = np.zeros(length_of_s)

for n in range(length_of_s):
    vco[n] = 440 + depth * np.sin(2 * np.pi * rate * n / fs)
    vca[n] = 1

s = sawtooth_wave(fs, vco, duration)

for n in range(length_of_s):
    s[n] *= vca[n]

for n in range(int(fs * 0.01)):
    s[n] *= n / (fs * 0.01)
    s[length_of_s - n - 1] *= n / (fs * 0.01)

length_of_s_master = int(fs * (duration + 2))
s_master = np.zeros(length_of_s_master)

offset = int(fs * 1)
for n in range(length_of_s):
    s_master[offset + n] += s[n]

master_volume = 0.5
s_master /= np.max(np.abs(s_master))
s_master *= master_volume

wave_write_16bit_mono(fs, s_master.copy(), 'p7_7(output).wav')
