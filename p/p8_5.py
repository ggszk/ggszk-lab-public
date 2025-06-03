import numpy as np
from wave_file import wave_read_16bit_stereo
from wave_file import wave_write_16bit_mono

fs, s0 = wave_read_16bit_stereo('p8_4(output).wav')
length_of_s = len(s0)

s0L = s0[:, 0]
s0R = s0[:, 1]

s1 = np.zeros(length_of_s)

for n in range(length_of_s):
    s1[n] = s0L[n] - s0R[n]

wave_write_16bit_mono(fs, s1.copy(), 'p8_5(output).wav')
