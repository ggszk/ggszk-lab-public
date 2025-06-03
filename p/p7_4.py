import numpy as np
from wave_file import wave_read_16bit_mono
from wave_file import wave_write_16bit_mono
from biquad_filter import LSF
from biquad_filter import HSF
from biquad_filter import PF
from biquad_filter import filter

fs, s0 = wave_read_16bit_mono('p7_4(input).wav')

fc = 500
Q = 1 / np.sqrt(2)
g = -1
a, b = LSF(fs, fc, Q, g)
s1 = filter(a, b, s0)

s0 = s1

fc = 1000
Q = 1 / np.sqrt(2)
g = 1
a, b = PF(fs, fc, Q, g)
s1 = filter(a, b, s0)

s0 = s1

fc = 2000
Q = 1 / np.sqrt(2)
g = -1
a, b = HSF(fs, fc, Q, g)
s1 = filter(a, b, s0)

master_volume = 1
s1 /= np.max(np.abs(s1))
s1 *= master_volume

wave_write_16bit_mono(fs, s1.copy(), 'p7_4(output).wav')
