import numpy as np
from wave_file import wave_read_16bit_mono
from wave_file import wave_write_16bit_mono
from sound_effects import compressor

fs, s = wave_read_16bit_mono('p7_3(input).wav')

threshold = 0.2
width = 0.1
ratio = 8
s = compressor(threshold, width, ratio, s)

wave_write_16bit_mono(fs, s.copy(), 'p7_3(output).wav')
