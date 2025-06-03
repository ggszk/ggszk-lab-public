import numpy as np
from wave_file import wave_write_16bit_mono
from oscillator import sawtooth_wave
from envelope import ADSR
from biquad_filter import LPF

fs = 44100
f0 = 440

gate = 3
duration = 4

VCO_A = np.array([0])
VCO_D = np.array([0])
VCO_S = np.array([1])
VCO_R = np.array([0])
VCO_gate = np.array([duration])
VCO_duration = np.array([duration])
VCO_offset = np.array([f0])
VCO_depth = np.array([0])

VCF_A = np.array([0])
VCF_D = np.array([0])
VCF_S = np.array([1])
VCF_R = np.array([0])
VCF_gate = np.array([duration])
VCF_duration = np.array([duration])
VCF_offset = np.array([f0 * 2])
VCF_depth = np.array([0])

VCA_A = np.array([0.01])
VCA_D = np.array([0])
VCA_S = np.array([1])
VCA_R = np.array([0.01])
VCA_gate = np.array([gate])
VCA_duration = np.array([duration])
VCA_offset = np.array([0])
VCA_depth = np.array([1])

length_of_s = int(fs * duration)
s0 = np.zeros(length_of_s)
s1 = np.zeros(length_of_s)

vco = ADSR(fs, VCO_A[0], VCO_D[0], VCO_S[0], VCO_R[0], VCO_gate[0], VCO_duration[0])
for n in range(length_of_s):
    vco[n] = VCO_offset[0] + vco[n] * VCO_depth[0]

s0 = sawtooth_wave(fs, vco, duration)

vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCO_gate[0], VCO_duration[0])
for n in range(length_of_s):
    vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]
    if vcf[n] > fs / 2:
        vcf[n] = fs / 2

Q = 1 / np.sqrt(2)
for n in range(length_of_s):
    a, b = LPF(fs, vcf[n], Q)
    for m in range(0, 3):
        if n - m >= 0:
            s1[n] += b[m] * s0[n - m]

    for m in range(1, 3):
        if n - m >= 0:
            s1[n] += -a[m] * s1[n - m]

vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
for n in range(length_of_s):
    vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

for n in range(length_of_s):
    s1[n] *= vca[n]

length_of_s_master = int(fs * (duration + 2))
s_master = np.zeros(length_of_s_master)

offset = int(fs * 1)
for n in range(length_of_s):
    s_master[offset + n] += s1[n]

master_volume = 0.5
s_master /= np.max(np.abs(s_master))
s_master *= master_volume

wave_write_16bit_mono(fs, s_master.copy(), 'p6_7(output).wav')
