import numpy as np
from wave_file import wave_read_16bit_mono
from wave_file import wave_write_16bit_mono
from biquad_filter import LPF

fs, s0 = wave_read_16bit_mono('p5_3(input).wav')
length_of_s = len(s0)

s1 = np.zeros(length_of_s)
fc = 1000
Q = 1 / np.sqrt(2)
a, b = LPF(fs, fc, Q)
for n in range(length_of_s):
    for m in range(0, 3):
        if n - m >= 0:
            s1[n] += b[m] * s0[n - m]

    for m in range(1, 3):
        if n - m >= 0:
            s1[n] += -a[m] * s1[n - m]

wave_write_16bit_mono(fs, s1.copy(), 'p5_3(output).wav')
