import numpy as np
from wave_file import wave_read_16bit_mono
from wave_file import wave_write_16bit_mono
from sound_effects import distortion

fs, s = wave_read_16bit_mono('p7_2(input).wav')

gain = 1000
level = 0.2
s = distortion(fs, gain, level, s)

wave_write_16bit_mono(fs, s.copy(), 'p7_2(output).wav')
