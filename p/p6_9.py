import numpy as np
from wave_file import wave_write_16bit_mono
from envelope import ADSR
from biquad_filter import HPF
from biquad_filter import filter

fs = 44100
f0 = 440

gate = 3
duration = 4

decay = 8
d = 0.5

T = 1 / f0

num = np.power(10, -3 * T / decay)
den = np.sqrt((1 - d) * (1 - d) + 2 * d * (1 - d) * np.cos((2 * np.pi * f0) / fs) + d * d)

c = num / den
if c > 1:
    c = 1

D = int(T * fs - d)
e = T * fs - d - int(T * fs - d)
g = (1 - e) / (1 + e)

length_of_s = int(fs * duration)
s0 = np.zeros(length_of_s)
s1 = np.zeros(length_of_s)
s2 = np.zeros(length_of_s)

np.random.seed(0)
mean_of_s0 = 0
for n in range(D + 1):
    s0[n] = (np.random.rand() * 2) - 1
    mean_of_s0 += s0[n]

mean_of_s0 /= D + 1
for n in range(D + 1):
    s0[n] -= mean_of_s0

for n in range(D + 1, length_of_s):
    # fractional delay
    s1[n] = -g * s1[n - 1] + g * s0[n - D] + s0[n - D - 1]

    # filter
    s2[n] = c * ((1 - d) * s1[n] + d * s1[n - 1])

    # feedback
    s0[n] += s2[n]

# DC cancel
fc = 5
Q = 1 / np.sqrt(2)
a, b = HPF(fs, fc, Q)
s3 = filter(a, b, s0)

VCA_A = np.array([0])
VCA_D = np.array([0])
VCA_S = np.array([1])
VCA_R = np.array([0.1])
VCA_gate = np.array([gate])
VCA_duration = np.array([duration])
VCA_offset = np.array([0])
VCA_depth = np.array([1])

vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
for n in range(length_of_s):
    vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

for n in range(length_of_s):
    s3[n] *= vca[n]

length_of_s_master = int(fs * (duration + 2))
s_master = np.zeros(length_of_s_master)

offset = int(fs * 1)
for n in range(length_of_s):
    s_master[offset + n] += s3[n]

master_volume = 0.5
s_master /= np.max(np.abs(s_master))
s_master *= master_volume

wave_write_16bit_mono(fs, s_master.copy(), 'p6_9(output).wav')
