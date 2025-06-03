import numpy as np
from wave_file import wave_read_16bit_mono
from wave_file import wave_write_16bit_mono
from sound_effects import reverb

fs, s = wave_read_16bit_mono('p7_1(input).wav')

reverb_time = 2
level = 0.1
s = reverb(fs, reverb_time, level, s)

wave_write_16bit_mono(fs, s.copy(), 'p7_1(output).wav')
