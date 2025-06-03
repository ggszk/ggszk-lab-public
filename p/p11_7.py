import numpy as np
from wave_file import wave_write_16bit_mono
from musical_instruments import electric_guitar
from sound_effects import distortion

fs = 44100

note_number = 40 # 40(E2) - 86(D6)
velocity = 100
gate = 4

s = electric_guitar(fs, note_number, velocity, gate)
length_of_s = len(s)

length_of_s_master = int(length_of_s + fs * 3)
s_master = np.zeros(length_of_s_master)

offset = int(fs * 1)
for n in range(length_of_s):
    s_master[offset + n] += s[n]

wave_write_16bit_mono(fs, s_master.copy(), 'electric_guitar' + str(note_number) + '.wav')

# distortion
gain = 1000
level = 0.2
s_master = distortion(fs, gain, level, s_master)

wave_write_16bit_mono(fs, s_master.copy(), 'electric_guitar_distorted' + str(note_number) + '.wav')
