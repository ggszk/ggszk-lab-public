import numpy as np
from window_function import Hanning_window
from envelope import ADSR
from envelope import cosine_envelope
from biquad_filter import LPF
from biquad_filter import HPF
from biquad_filter import BPF
from biquad_filter import filter
from sound_effects import compressor

def glockenspiel(fs, note_number, velocity, gate):
    duration = 4 + 1

    length_of_s = int(fs * duration)
    s = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    number_of_partial = 7

    VCO_A = np.repeat(0, number_of_partial)
    VCO_D = np.repeat(0, number_of_partial)
    VCO_S = np.repeat(1, number_of_partial)
    VCO_R = np.repeat(0, number_of_partial)
    VCO_gate = np.repeat(duration, number_of_partial)
    VCO_duration = np.repeat(duration, number_of_partial)
    VCO_offset = np.array([1, 2.8, 5.4, 8.9, 13.3, 18.6, 24.8]) * f0
    VCO_depth = np.repeat(0, number_of_partial)

    VCA_A = np.repeat(0.01, number_of_partial)
    VCA_D = np.array([4, 1, 0.8, 0.6, 0.5, 0.4, 0.3])
    VCA_S = np.repeat(0, number_of_partial)
    VCA_R = np.array([4, 1, 0.8, 0.6, 0.5, 0.4, 0.3])
    VCA_gate = np.repeat(gate, number_of_partial)
    VCA_duration = np.repeat(duration, number_of_partial)
    VCA_offset = np.repeat(0, number_of_partial)
    VCA_depth = np.repeat(1, number_of_partial)

    for i in range(number_of_partial):
        vco = ADSR(fs, VCO_A[i], VCO_D[i], VCO_S[i], VCO_R[i], VCO_gate[i], VCO_duration[i])
        for n in range(length_of_s):
            vco[n] = VCO_offset[i] + vco[n] * VCO_depth[i]

        if np.max(vco) < 20000:
            p = np.zeros(length_of_s)
            x = 0
            for n in range(length_of_s):
                p[n] = np.sin(2 * np.pi * x)
                delta = vco[n] / fs
                x += delta
                if x >= 1:
                    x -= 1

            vca = ADSR(fs, VCA_A[i], VCA_D[i], VCA_S[i], VCA_R[i], VCA_gate[i], VCA_duration[i])
            for n in range(length_of_s):
                vca[n] = VCA_offset[i] + vca[n] * VCA_depth[i]

            for n in range(length_of_s):
                s[n] += p[n] * vca[n]

    s *= velocity / 127 / np.max(np.abs(s))

    return s

def triangle_in(fs, note_number, velocity, gate):
    duration = 8 + 1

    length_of_s = int(fs * duration)
    s = np.zeros(length_of_s)

    number_of_partial = 25

    VCO_A = np.repeat(0, number_of_partial)
    VCO_D = np.repeat(0, number_of_partial)
    VCO_S = np.repeat(1, number_of_partial)
    VCO_R = np.repeat(0, number_of_partial)
    VCO_gate = np.repeat(duration, number_of_partial)
    VCO_duration = np.repeat(duration, number_of_partial)
    VCO_offset = np.array([172.4, 297.8, 1066, 1639, 1827, 3035, 3428, 4208, 5072, 6856, 7001, 8577, 9474, 11442, 11616, 12753, 13836, 15367, 16085, 16383, 16961, 18217, 19221, 20309, 21818])
    VCO_depth = np.repeat(0, number_of_partial)

    VCA_A = np.repeat(0.01, number_of_partial)
    VCA_D = np.array([8, 8, 6, 6, 4, 4, 4, 4, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    VCA_S = np.repeat(0, number_of_partial)
    VCA_R = np.array([8, 8, 6, 6, 4, 4, 4, 4, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    VCA_gate = np.repeat(gate, number_of_partial)
    VCA_duration = np.repeat(duration, number_of_partial)
    VCA_offset = np.repeat(0, number_of_partial)
    VCA_depth = np.array([0.01, 0.01, 0.07, 0.15, 0.29, 0.02, 0.08, 0.02, 0.56, 0.37, 0.04, 0.41, 1, 0.01, 0.05, 0.05, 0.01, 0.04, 0.01, 0.02, 0.01, 0.01, 0.03, 0.03, 0.01])

    for i in range(number_of_partial):
        vco = ADSR(fs, VCO_A[i], VCO_D[i], VCO_S[i], VCO_R[i], VCO_gate[i], VCO_duration[i])
        for n in range(length_of_s):
            vco[n] = VCO_offset[i] + vco[n] * VCO_depth[i]

        if np.max(vco) < 20000:
            p = np.zeros(length_of_s)
            x = 0
            for n in range(length_of_s):
                p[n] = np.sin(2 * np.pi * x)
                delta = vco[n] / fs
                x += delta
                if x >= 1:
                    x -= 1

            vca = ADSR(fs, VCA_A[i], VCA_D[i], VCA_S[i], VCA_R[i], VCA_gate[i], VCA_duration[i])
            for n in range(length_of_s):
                vca[n] = VCA_offset[i] + vca[n] * VCA_depth[i]

            for n in range(length_of_s):
                s[n] += p[n] * vca[n]

    s *= velocity / 127 / np.max(np.abs(s))

    return s

def triangle_out(fs, note_number, velocity, gate):
    duration = 8 + 1

    length_of_s = int(fs * duration)
    s = np.zeros(length_of_s)

    number_of_partial = 25

    VCO_A = np.repeat(0, number_of_partial)
    VCO_D = np.repeat(0, number_of_partial)
    VCO_S = np.repeat(1, number_of_partial)
    VCO_R = np.repeat(0, number_of_partial)
    VCO_gate = np.repeat(duration, number_of_partial)
    VCO_duration = np.repeat(duration, number_of_partial)
    VCO_offset = np.array([176.5, 872.7, 1593, 1791, 3035, 3928, 4817, 4873, 5345, 6856, 7001, 8441, 8770, 10213, 11442, 11616, 13836, 14834, 15259, 15737, 16961, 18217, 18227, 21691, 21818])
    VCO_depth = np.repeat(0, number_of_partial)

    VCA_A = np.repeat(0.01, number_of_partial)
    VCA_D = np.array([8, 8, 6, 6, 4, 4, 4, 4, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    VCA_S = np.repeat(0, number_of_partial)
    VCA_R = np.array([8, 8, 6, 6, 4, 4, 4, 4, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    VCA_gate = np.repeat(gate, number_of_partial)
    VCA_duration = np.repeat(duration, number_of_partial)
    VCA_offset = np.repeat(0, number_of_partial)
    VCA_depth = np.array([0.01, 0.01, 0.06, 0.12, 0.02, 0.04, 0.03, 0.01, 0.02, 0.02, 0.41, 1, 0.39, 0.57, 0.43, 0.04, 0.06, 0.01, 0.01, 0.01, 0.01, 0.01, 0.04, 0.01, 0.01])

    for i in range(number_of_partial):
        vco = ADSR(fs, VCO_A[i], VCO_D[i], VCO_S[i], VCO_R[i], VCO_gate[i], VCO_duration[i])
        for n in range(length_of_s):
            vco[n] = VCO_offset[i] + vco[n] * VCO_depth[i]

        if np.max(vco) < 20000:
            p = np.zeros(length_of_s)
            x = 0
            for n in range(length_of_s):
                p[n] = np.sin(2 * np.pi * x)
                delta = vco[n] / fs
                x += delta
                if x >= 1:
                    x -= 1

            vca = ADSR(fs, VCA_A[i], VCA_D[i], VCA_S[i], VCA_R[i], VCA_gate[i], VCA_duration[i])
            for n in range(length_of_s):
                vca[n] = VCA_offset[i] + vca[n] * VCA_depth[i]

            for n in range(length_of_s):
                s[n] += p[n] * vca[n]

    s *= velocity / 127 / np.max(np.abs(s))

    return s

def tubular_bells(fs, note_number, velocity, gate):
    duration = 4 + 1

    length_of_s = int(fs * duration)
    s = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    VCO_A = np.array([0, 0])
    VCO_D = np.array([0, 0])
    VCO_S = np.array([1, 1])
    VCO_R = np.array([0, 0])
    VCO_gate = np.repeat(duration, 2)
    VCO_duration = np.repeat(duration, 2)
    VCO_offset = np.array([3.5, 1]) * f0
    VCO_depth = np.array([0, 0])

    VCA_A = np.array([0, 0])
    VCA_D = np.array([2, 4])
    VCA_S = np.array([0, 0])
    VCA_R = np.array([2, 4])
    VCA_gate = np.repeat(gate, 2)
    VCA_duration = np.repeat(duration, 2)
    VCA_offset = np.array([0, 0])
    VCA_depth = np.array([1, 1])

    vco_m = ADSR(fs, VCO_A[0], VCO_D[0], VCO_S[0], VCO_R[0], VCO_gate[0], VCO_duration[0])
    for n in range(length_of_s):
        vco_m[n] = VCO_offset[0] + vco_m[n] * VCO_depth[0]

    vca_m = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca_m[n] = VCA_offset[0] + vca_m[n] * VCA_depth[0]

    vco_c = ADSR(fs, VCO_A[1], VCO_D[1], VCO_S[1], VCO_R[1], VCO_gate[1], VCO_duration[1])
    for n in range(length_of_s):
        vco_c[n] = VCO_offset[1] + vco_c[n] * VCO_depth[1]

    vca_c = ADSR(fs, VCA_A[1], VCA_D[1], VCA_S[1], VCA_R[1], VCA_gate[1], VCA_duration[1])
    for n in range(length_of_s):
        vca_c[n] = VCA_offset[1] + vca_c[n] * VCA_depth[1]

    xm = 0
    xc = 0
    for n in range(length_of_s):
        s[n] = vca_c[n] * np.sin(2 * np.pi * xc + vca_m[n] * np.sin(2 * np.pi * xm))
        delta_m = vco_m[n] / fs
        xm += delta_m
        if xm >= 1:
            xm -= 1

        delta_c = vco_c[n] / fs
        xc += delta_c
        if xc >= 1:
            xc -= 1

    s *= velocity / 127 / np.max(np.abs(s))

    return s

def marimba(fs, note_number, velocity, gate):
    duration = 0.8 + 1

    length_of_s = int(fs * duration)
    s0 = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    np.random.seed(0)
    for n in range(length_of_s):
        s0[n] = np.random.rand() * 2 - 1

    fc = f0
    if fc < 20000:
        Q = 1 / np.sqrt(2)
        a, b = LPF(fs, fc, Q)
        s1 = filter(a, b, s0)
        s0 = s1

    VCF_A = np.array([0])
    VCF_D = np.array([0.2])
    VCF_S = np.array([0])
    VCF_R = np.array([0.2])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([500])
    VCF_depth = np.array([2000])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]

    s1 = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                s1[n] += b[m] * s0[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                s1[n] += -a[m] * s1[n - m]

    s3 = np.zeros(length_of_s)

    fc = f0
    if fc < 20000:
        Q = 200
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        s3 += s2

    fc = f0 * 4
    if fc < 20000:
        Q = 200
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        s3 += s2

    fc = f0 * 10
    if fc < 20000:
        Q = 200
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        s3 += s2

    VCA_A = np.array([0])
    VCA_D = np.array([0.8])
    VCA_S = np.array([0])
    VCA_R = np.array([0.8])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        s3[n] *= vca[n]

    s3 *= velocity / 127 / np.max(np.abs(s3))

    return s3

def xylophone(fs, note_number, velocity, gate):
    duration = 0.8 + 1

    length_of_s = int(fs * duration)
    s0 = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    np.random.seed(0)
    for n in range(length_of_s):
        s0[n] = np.random.rand() * 2 - 1

    #fc = f0
    #if fc < 20000:
    #    Q = 1 / np.sqrt(2)
    #    a, b = LPF(fs, fc, Q)
    #    s1 = filter(a, b, s0)
    #    s0 = s1

    VCF_A = np.array([0])
    VCF_D = np.array([0.2])
    VCF_S = np.array([0])
    VCF_R = np.array([0.2])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([500])
    VCF_depth = np.array([2000])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]

    s1 = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                s1[n] += b[m] * s0[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                s1[n] += -a[m] * s1[n - m]

    s3 = np.zeros(length_of_s)

    fc = f0
    if fc < 20000:
        Q = 200
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        s3 += s2

    fc = f0 * 3
    if fc < 20000:
        Q = 200
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        s3 += s2

    fc = f0 * 6.5
    if fc < 20000:
        Q = 200
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        s3 += s2

    fc = f0 * 10
    if fc < fs * 0.45:
        Q = 200
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        s3 += s2

    VCA_A = np.array([0])
    VCA_D = np.array([0.8])
    VCA_S = np.array([0])
    VCA_R = np.array([0.8])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        s3[n] *= vca[n]

    s3 *= velocity / 127 / np.max(np.abs(s3))

    return s3

def timpani(fs, note_number, velocity, gate):
    duration = 2 + 1

    length_of_s = int(fs * duration)
    sa = np.zeros(length_of_s)
    sb = np.zeros(length_of_s)
    s0 = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    np.random.seed(0)
    for n in range(length_of_s):
        s0[n] = np.random.rand() * 2 - 1

    # A part
    VCF_A = np.array([0])
    VCF_D = np.array([0.5])
    VCF_S = np.array([0])
    VCF_R = np.array([0.5])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([f0 * 2])
    VCF_depth = np.array([5000])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]

    s1 = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                s1[n] += b[m] * s0[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                s1[n] += -a[m] * s1[n - m]

    fc = f0 * 1.7
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 2.7
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 2.9
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 3.4
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 3.8
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 4.2
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 4.4
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 4.8
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 5.2
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 5.6
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 5.9
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 6.3
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 6.7
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 7.3
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 7.6
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 7.8
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 8.6
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 8.9
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 9.3
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 9.6
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 10.1
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 10.4
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 11.1
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    fc = f0 * 11.7
    if fc < 20000:
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sa += s2

    VCA_A = np.array([0])
    VCA_D = np.array([1])
    VCA_S = np.array([0])
    VCA_R = np.array([1])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        sa[n] *= vca[n]

    sa /= np.max(np.abs(sa))

    # B part
    VCF_A = np.array([0])
    VCF_D = np.array([1])
    VCF_S = np.array([0])
    VCF_R = np.array([1])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([f0 * 2])
    VCF_depth = np.array([5000])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]

    s1 = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                s1[n] += b[m] * s0[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                s1[n] += -a[m] * s1[n - m]

    fc = f0
    if fc < 20000:
        Q = 200
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sb += s2

    fc = f0 * 1.5
    if fc < 20000:
        Q = 200
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sb += s2

    fc = f0 * 2
    if fc < 20000:
        Q = 200
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sb += s2

    fc = f0 * 2.5
    if fc < 20000:
        Q = 200
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        sb += s2

    VCA_A = np.array([0])
    VCA_D = np.array([2])
    VCA_S = np.array([0])
    VCA_R = np.array([2])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        sb[n] *= vca[n]

    sb /= np.max(np.abs(sb))

    s = sa * 0.3 + sb * 0.7

    # compressor
    s = s / np.max(np.abs(s))
    threshold = 0.5
    width = 0.4
    ratio = 8
    s = compressor(threshold, width, ratio, s)

    s *= velocity / 127 / np.max(np.abs(s))

    return s

def cymbal(fs, note_number, velocity, gate):
    duration = 3 + 1

    length_of_s = int(fs * duration)
    s0 = np.zeros(length_of_s)

    np.random.seed(0)
    for n in range(length_of_s):
        s0[n] = np.random.rand() * 2 - 1

    VCF_A = np.array([0.1])
    VCF_D = np.array([3])
    VCF_S = np.array([0])
    VCF_R = np.array([3])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([50])
    VCF_depth = np.array([16000])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]

    s1 = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                s1[n] += b[m] * s0[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                s1[n] += -a[m] * s1[n - m]

    s3 = np.zeros(length_of_s)

    fL = 500
    fH = 20000
    np.random.seed(0)
    for i in range(100):
        r = -np.log(1 - np.random.rand()) / 10
        if r > 1:
            r = 1

        fc = r * (fH - fL) + fL
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        s3 += s2

    VCA_A = np.array([0])
    VCA_D = np.array([3])
    VCA_S = np.array([0])
    VCA_R = np.array([3])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        s3[n] *= vca[n]

    s3 *= velocity / 127 / np.max(np.abs(s3))

    return s3

def tamtam(fs, note_number, velocity, gate):
    duration = 8 + 1

    length_of_s = int(fs * duration)
    s0 = np.zeros(length_of_s)

    np.random.seed(0)
    for n in range(length_of_s):
        s0[n] = np.random.rand() * 2 - 1

    VCF_A = np.array([2.5])
    VCF_D = np.array([5])
    VCF_S = np.array([0])
    VCF_R = np.array([5])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([50])
    VCF_depth = np.array([16000])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]

    s1 = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                s1[n] += b[m] * s0[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                s1[n] += -a[m] * s1[n - m]

    s3 = np.zeros(length_of_s)

    fL = 150
    fH = 20000
    np.random.seed(0)
    for i in range(100):
        r = -np.log(1 - np.random.rand()) / 10
        if r > 1:
            r = 1

        fc = r * (fH - fL) + fL
        Q = 100
        a, b = BPF(fs, fc, Q)
        s2 = filter(a, b, s1)
        s3 += s2

    VCA_A = np.array([0])
    VCA_D = np.array([7])
    VCA_S = np.array([0])
    VCA_R = np.array([7])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        s3[n] *= vca[n]

    s3 *= velocity / 127 / np.max(np.abs(s3))

    return s3

def hihat_cymbal_close(fs, note_number, velocity, gate):
    duration = 0.1 + 1

    length_of_s = int(fs * duration)
    s0 = np.zeros(length_of_s)

    np.random.seed(0)
    for n in range(length_of_s):
        s0[n] = np.random.rand() * 2 - 1

    VCF_A = np.array([0])
    VCF_D = np.array([0])
    VCF_S = np.array([1])
    VCF_R = np.array([0])
    VCF_gate = np.array([duration])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([10000])
    VCF_depth = np.array([0])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]

    s1 = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = HPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                s1[n] += b[m] * s0[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                s1[n] += -a[m] * s1[n - m]

    VCA_A = np.array([0])
    VCA_D = np.array([0.1])
    VCA_S = np.array([0])
    VCA_R = np.array([0.1])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        s1[n] *= vca[n]

    s1 *= velocity / 127 / np.max(np.abs(s1))

    return s1

def bass_drum(fs, note_number, velocity, gate):
    duration = 0.3 + 1

    length_of_s = int(fs * duration)
    sa = np.zeros(length_of_s)
    sb = np.zeros(length_of_s)
    s0 = np.zeros(length_of_s)

    # A part
    np.random.seed(0)
    for n in range(length_of_s):
        sa[n] = np.random.rand() * 2 - 1

    VCA_A = np.array([0])
    VCA_D = np.array([0.2])
    VCA_S = np.array([0])
    VCA_R = np.array([0.2])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        sa[n] *= vca[n]

    # B part
    VCO_A = np.array([0])
    VCO_D = np.array([0.1])
    VCO_S = np.array([0])
    VCO_R = np.array([0.1])
    VCO_gate = np.array([gate])
    VCO_duration = np.array([duration])
    VCO_offset = np.array([55])
    VCO_depth = np.array([50])

    vco = ADSR(fs, VCO_A[0], VCO_D[0], VCO_S[0], VCO_R[0], VCO_gate[0], VCO_duration[0])
    for n in range(length_of_s):
        vco[n] = VCO_offset[0] + vco[n] * VCO_depth[0]

    x = 0
    for n in range(length_of_s):
        sb[n] = np.sin(2 * np.pi * x)
        delta = vco[n] / fs
        x += delta
        if x >= 1:
            x -= 1

    for n in range(length_of_s):
        s0[n] = sa[n] * 0.4 + sb[n] * 0.6

    VCF_A = np.array([0])
    VCF_D = np.array([0.08])
    VCF_S = np.array([0])
    VCF_R = np.array([0.08])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([200])
    VCF_depth = np.array([6000])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]

    s1 = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                s1[n] += b[m] * s0[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                s1[n] += -a[m] * s1[n - m]

    VCA_A = np.array([0])
    VCA_D = np.array([0.3])
    VCA_S = np.array([0])
    VCA_R = np.array([0.3])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        s1[n] *= vca[n]

    # compressor
    s1 = s1 / np.max(np.abs(s1))
    threshold = 0.5
    width = 0.4
    ratio = 8
    s1 = compressor(threshold, width, ratio, s1)

    s1 *= velocity / 127 / np.max(np.abs(s1))

    return s1

def tom_drum(fs, note_number, velocity, gate):
    duration = 0.4 + 1

    length_of_s = int(fs * duration)
    sa = np.zeros(length_of_s)
    sb = np.zeros(length_of_s)
    s0 = np.zeros(length_of_s)

    # A part
    np.random.seed(0)
    for n in range(length_of_s):
        sa[n] = np.random.rand() * 2 - 1

    VCA_A = np.array([0])
    VCA_D = np.array([0.2])
    VCA_S = np.array([0])
    VCA_R = np.array([0.2])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        sa[n] *= vca[n]

    # B part
    VCO_A = np.array([0])
    VCO_D = np.array([0.1])
    VCO_S = np.array([0])
    VCO_R = np.array([0.1])
    VCO_gate = np.array([gate])
    VCO_duration = np.array([duration])
    VCO_offset = np.array([150])
    VCO_depth = np.array([50])

    vco = ADSR(fs, VCO_A[0], VCO_D[0], VCO_S[0], VCO_R[0], VCO_gate[0], VCO_duration[0])
    for n in range(length_of_s):
        vco[n] = VCO_offset[0] + vco[n] * VCO_depth[0]

    x = 0
    for n in range(length_of_s):
        sb[n] = np.sin(2 * np.pi * x)
        delta = vco[n] / fs
        x += delta
        if x >= 1:
            x -= 1

    for n in range(length_of_s):
        s0[n] = sa[n] * 0.6 + sb[n] * 0.4

    VCF_A = np.array([0])
    VCF_D = np.array([0.08])
    VCF_S = np.array([0])
    VCF_R = np.array([0.08])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([200])
    VCF_depth = np.array([6000])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]

    s1 = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                s1[n] += b[m] * s0[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                s1[n] += -a[m] * s1[n - m]

    VCA_A = np.array([0])
    VCA_D = np.array([0.4])
    VCA_S = np.array([0])
    VCA_R = np.array([0.4])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        s1[n] *= vca[n]

    # compressor
    s1 = s1 / np.max(np.abs(s1))
    threshold = 0.5
    width = 0.4
    ratio = 8
    s1 = compressor(threshold, width, ratio, s1)

    s1 *= velocity / 127 / np.max(np.abs(s1))

    return s1

def snare_drum(fs, note_number, velocity, gate):
    duration = 0.2 + 1

    length_of_s = int(fs * duration)
    sa = np.zeros(length_of_s)
    sb = np.zeros(length_of_s)
    s0 = np.zeros(length_of_s)

    # A part
    np.random.seed(0)
    for n in range(length_of_s):
        sa[n] = np.random.rand() * 2 - 1

    VCA_A = np.array([0])
    VCA_D = np.array([0])
    VCA_S = np.array([1])
    VCA_R = np.array([0])
    VCA_gate = np.array([duration])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_duration[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        sa[n] *= vca[n]

    # B part
    VCO_A = np.array([0])
    VCO_D = np.array([0])
    VCO_S = np.array([1])
    VCO_R = np.array([0])
    VCO_gate = np.array([duration])
    VCO_duration = np.array([duration])
    VCO_offset = np.array([150])
    VCO_depth = np.array([0])

    vco = ADSR(fs, VCO_A[0], VCO_D[0], VCO_S[0], VCO_R[0], VCO_duration[0], VCO_duration[0])
    for n in range(length_of_s):
        vco[n] = VCO_offset[0] + vco[n] * VCO_depth[0]

    x = 0
    for n in range(length_of_s):
        if x < 0.5:
            sb[n] = 1
        else:
            sb[n] = -1

        delta = vco[n] / fs

        if 1 - delta <= x and x < 1:
            t = (x - 1) / delta
            d = t * t + 2 * t + 1
            sb[n] += d
        elif 0 <= x and x < delta:
            t = x / delta
            d = -t * t + 2 * t - 1
            sb[n] += d

        if 0.5 - delta <= x and x < 0.5:
            t = (x - 0.5) / delta
            d = t * t + 2 * t + 1
            sb[n] -= d
        elif 0.5 <= x and x < 0.5 + delta:
            t = (x - 0.5) / delta
            d = -t * t + 2 * t - 1
            sb[n] -= d

        x += delta
        if x >= 1:
            x -= 1

    for n in range(length_of_s):
        s0[n] = sa[n] * 0.7 + sb[n] * 0.3

    VCF_A = np.array([0])
    VCF_D = np.array([0.1])
    VCF_S = np.array([0])
    VCF_R = np.array([0.1])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([8000])
    VCF_depth = np.array([-7800])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]

    s1 = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                s1[n] += b[m] * s0[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                s1[n] += -a[m] * s1[n - m]

    VCA_A = np.array([0])
    VCA_D = np.array([0.2])
    VCA_S = np.array([0])
    VCA_R = np.array([0.2])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        s1[n] *= vca[n]

    # compressor
    s1 = s1 / np.max(np.abs(s1))
    threshold = 0.5
    width = 0.4
    ratio = 8
    s1 = compressor(threshold, width, ratio, s1)

    s1 *= velocity / 127 / np.max(np.abs(s1))

    return s1

def flute(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    s = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    number_of_partial = 30

    a = np.array([[0.932, 0.042, 0.175, 0.014, 0.029, 0.008, 0.026, 0.015, 0.019, 0.010, 0.008, 0.004, 0.003, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.001, 0.000, 0.000, 0.001, 0.000, 0.001, 0.000, 0.001],
                  [0.932, 0.091, 0.159, 0.021, 0.035, 0.015, 0.039, 0.016, 0.026, 0.010, 0.006, 0.004, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.001, 0.000, 0.001, 0.001, 0.001, 0.001, 0.000, 0.001, 0.000, 0.001],
                  [0.932, 0.150, 0.108, 0.024, 0.040, 0.026, 0.039, 0.014, 0.018, 0.009, 0.002, 0.004, 0.002, 0.003, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.165, 0.086, 0.027, 0.050, 0.033, 0.026, 0.010, 0.007, 0.007, 0.002, 0.003, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.215, 0.095, 0.032, 0.049, 0.047, 0.015, 0.011, 0.006, 0.006, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.448, 0.116, 0.070, 0.079, 0.058, 0.010, 0.009, 0.004, 0.005, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.685, 0.149, 0.130, 0.119, 0.047, 0.012, 0.006, 0.005, 0.005, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.001, 0.001],
                  [0.932, 0.594, 0.226, 0.147, 0.118, 0.046, 0.017, 0.007, 0.009, 0.004, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.001, 0.001, 0.001, 0.001],
                  [0.932, 0.532, 0.267, 0.116, 0.113, 0.057, 0.020, 0.007, 0.008, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
                  [0.932, 0.583, 0.190, 0.075, 0.074, 0.036, 0.013, 0.005, 0.004, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
                  [0.932, 0.407, 0.128, 0.052, 0.025, 0.010, 0.005, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.213, 0.118, 0.039, 0.015, 0.006, 0.003, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.156, 0.105, 0.042, 0.014, 0.004, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.128, 0.110, 0.063, 0.014, 0.004, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.092, 0.120, 0.051, 0.012, 0.004, 0.002, 0.001, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.063, 0.107, 0.024, 0.007, 0.002, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.057, 0.093, 0.017, 0.004, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.068, 0.077, 0.013, 0.004, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.105, 0.051, 0.007, 0.004, 0.002, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.161, 0.053, 0.005, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.176, 0.065, 0.003, 0.003, 0.002, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.139, 0.049, 0.003, 0.003, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.087, 0.038, 0.002, 0.002, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.089, 0.052, 0.006, 0.004, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.194, 0.089, 0.017, 0.009, 0.003, 0.003, 0.002, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.248, 0.103, 0.022, 0.011, 0.004, 0.004, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.148, 0.067, 0.013, 0.006, 0.003, 0.003, 0.001, 0.001, 0.002, 0.001, 0.001, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.070, 0.036, 0.006, 0.004, 0.002, 0.001, 0.001, 0.002, 0.002, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.082, 0.030, 0.006, 0.006, 0.003, 0.001, 0.001, 0.005, 0.002, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.077, 0.034, 0.007, 0.005, 0.002, 0.001, 0.002, 0.005, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.041, 0.035, 0.006, 0.005, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.020, 0.024, 0.003, 0.003, 0.002, 0.002, 0.003, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.019, 0.024, 0.005, 0.003, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.019, 0.028, 0.008, 0.003, 0.002, 0.003, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.023, 0.023, 0.006, 0.003, 0.009, 0.003, 0.002, 0.002, 0.002, 0.003, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.038, 0.029, 0.006, 0.005, 0.021, 0.005, 0.004, 0.004, 0.004, 0.006, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.046, 0.037, 0.007, 0.006, 0.026, 0.007, 0.005, 0.005, 0.005, 0.006, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.046, 0.037, 0.007, 0.006, 0.026, 0.007, 0.005, 0.005, 0.004, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.932, 0.046, 0.037, 0.007, 0.006, 0.026, 0.007, 0.005, 0.005, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]])

    VCO_A = np.repeat(0, number_of_partial)
    VCO_D = np.repeat(0, number_of_partial)
    VCO_S = np.repeat(1, number_of_partial)
    VCO_R = np.repeat(0, number_of_partial)
    VCO_gate = np.repeat(duration, number_of_partial)
    VCO_duration = np.repeat(duration, number_of_partial)
    VCO_offset = np.arange(1, number_of_partial + 1) * f0
    VCO_depth = np.repeat(0, number_of_partial)

    p1 = 0
    p2 = -0.1
    p3 = 8000 / 12
    p4 = 0.2
    VCA_delay = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_delay[i] = 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3)) * p4 + p2

    for i in range(1, number_of_partial):
        VCA_delay[i] = VCA_delay[i] - VCA_delay[0]

    VCA_delay[0] = 0

    p1 = 0
    p2 = 0.1
    p3 = 8000 / 12
    p4 = 0.2
    VCA_A = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_A[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA_S = np.repeat(1, number_of_partial)

    p1 = 0
    p2 = 0.2
    p3 = 8000 / 12
    p4 = 0.4
    VCA_R = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_R[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA_gate = np.repeat(gate, number_of_partial)
    VCA_duration = np.repeat(duration, number_of_partial)
    VCA_offset = np.repeat(0, number_of_partial)
    VCA_depth = a[note_number - 59, :]

    np.random.seed(0)

    for i in range(number_of_partial):
        vco = ADSR(fs, VCO_A[i], VCO_D[i], VCO_S[i], VCO_R[i], VCO_gate[i], VCO_duration[i])
        for n in range(length_of_s):
            vco[n] = VCO_offset[i] + vco[n] * VCO_depth[i]

        w = np.zeros(length_of_s)
        for n in range(length_of_s):
            w[n] = np.random.rand() * 2 - 1

        fc = 40
        Q = 1 / np.sqrt(2)
        a, b = LPF(fs, fc, Q)
        jitter = filter(a, b, w)
        jitter /= np.max(np.abs(jitter))

        p1 = 108
        p2 = 1
        p3 = 150 / 12
        p4 = 20
        jitter_depth = 1 / (1 + np.exp(-(note_number - p1) / p3)) * p4 + p2

        for n in range(length_of_s):
            vco[n] += jitter[n] * jitter_depth

        if np.max(vco) < 20000:
            p = np.zeros(length_of_s)
            x = 0
            for n in range(length_of_s):
                p[n] = np.sin(2 * np.pi * x)
                delta = vco[n] / fs
                x += delta
                if x >= 1:
                    x -= 1

            vca = cosine_envelope(fs, VCA_delay[i], VCA_A[i], VCA_S[i], VCA_R[i], VCA_gate[i], VCA_duration[i])
            for n in range(length_of_s):
                vca[n] = VCA_offset[i] + vca[n] * VCA_depth[i]

            w = np.zeros(length_of_s)
            for n in range(length_of_s):
                w[n] = np.random.rand() * 2 - 1

            fc = 40
            Q = 1 / np.sqrt(2)
            a, b = LPF(fs, fc, Q)
            shimmer = filter(a, b, w)
            shimmer /= np.max(np.abs(shimmer))

            p1 = 1
            p2 = -0.3
            p3 = 10 / 12
            p4 = 0.8
            shimmer_depth = 1 / (1 + np.exp(-(VCO_offset[i] / f0 - p1) / p3)) * p4 + p2

            for n in range(length_of_s):
                vca[n] *= 1 + shimmer[n] * shimmer_depth
                if vca[n] < 0:
                    vca[n] = 0

            for n in range(length_of_s):
                s[n] += p[n] * vca[n]

    s *= velocity / 127 / np.max(np.abs(s))

    return s

def piccolo(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    s = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    number_of_partial = 30

    a = np.array([[0.523, 0.053, 0.067, 0.058, 0.067, 0.015, 0.006, 0.009, 0.002, 0.002, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.058, 0.085, 0.061, 0.060, 0.015, 0.005, 0.008, 0.001, 0.001, 0.002, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.086, 0.079, 0.051, 0.047, 0.014, 0.006, 0.006, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.342, 0.077, 0.123, 0.089, 0.029, 0.011, 0.004, 0.002, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.572, 0.111, 0.235, 0.148, 0.045, 0.011, 0.003, 0.002, 0.001, 0.001, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.347, 0.117, 0.197, 0.104, 0.027, 0.006, 0.003, 0.002, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.114, 0.106, 0.117, 0.064, 0.007, 0.005, 0.002, 0.002, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.096, 0.111, 0.094, 0.083, 0.004, 0.007, 0.003, 0.002, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.105, 0.113, 0.074, 0.075, 0.003, 0.007, 0.004, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.131, 0.108, 0.052, 0.045, 0.004, 0.005, 0.003, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.121, 0.094, 0.032, 0.025, 0.005, 0.004, 0.003, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.090, 0.083, 0.021, 0.012, 0.004, 0.004, 0.002, 0.002, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.134, 0.102, 0.029, 0.007, 0.003, 0.004, 0.002, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.221, 0.135, 0.051, 0.011, 0.003, 0.004, 0.002, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.196, 0.115, 0.046, 0.010, 0.002, 0.003, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.095, 0.061, 0.019, 0.005, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.045, 0.053, 0.006, 0.003, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.029, 0.060, 0.006, 0.005, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.032, 0.048, 0.004, 0.005, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.059, 0.031, 0.002, 0.005, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.058, 0.020, 0.002, 0.004, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.033, 0.013, 0.003, 0.002, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.022, 0.009, 0.003, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.020, 0.009, 0.003, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.021, 0.008, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.029, 0.011, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.030, 0.014, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.024, 0.010, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.034, 0.006, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.058, 0.005, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.053, 0.005, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.021, 0.006, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.007, 0.005, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.019, 0.004, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.036, 0.004, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.045, 0.002, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.523, 0.048, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]])

    VCO_A = np.repeat(0, number_of_partial)
    VCO_D = np.repeat(0, number_of_partial)
    VCO_S = np.repeat(1, number_of_partial)
    VCO_R = np.repeat(0, number_of_partial)
    VCO_gate = np.repeat(duration, number_of_partial)
    VCO_duration = np.repeat(duration, number_of_partial)
    VCO_offset = np.arange(1, number_of_partial + 1) * f0
    VCO_depth = np.repeat(0, number_of_partial)

    p1 = 0
    p2 = -0.1
    p3 = 8000 / 12
    p4 = 0.2
    VCA_delay = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_delay[i] = 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3)) * p4 + p2

    for i in range(1, number_of_partial):
        VCA_delay[i] = VCA_delay[i] - VCA_delay[0]

    VCA_delay[0] = 0

    p1 = 0
    p2 = 0.1
    p3 = 8000 / 12
    p4 = 0.2
    VCA_A = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_A[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA_S = np.repeat(1, number_of_partial)

    p1 = 0
    p2 = 0.2
    p3 = 8000 / 12
    p4 = 0.4
    VCA_R = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_R[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA_gate = np.repeat(gate, number_of_partial)
    VCA_duration = np.repeat(duration, number_of_partial)
    VCA_offset = np.repeat(0, number_of_partial)
    VCA_depth = a[note_number - 72, :]

    np.random.seed(0)

    for i in range(number_of_partial):
        vco = ADSR(fs, VCO_A[i], VCO_D[i], VCO_S[i], VCO_R[i], VCO_gate[i], VCO_duration[i])
        for n in range(length_of_s):
            vco[n] = VCO_offset[i] + vco[n] * VCO_depth[i]

        w = np.zeros(length_of_s)
        for n in range(length_of_s):
            w[n] = np.random.rand() * 2 - 1

        fc = 40
        Q = 1 / np.sqrt(2)
        a, b = LPF(fs, fc, Q)
        jitter = filter(a, b, w)
        jitter /= np.max(np.abs(jitter))

        p1 = 108
        p2 = 1
        p3 = 150 / 12
        p4 = 20
        jitter_depth = 1 / (1 + np.exp(-(note_number - p1) / p3)) * p4 + p2

        for n in range(length_of_s):
            vco[n] += jitter[n] * jitter_depth

        if np.max(vco) < 20000:
            p = np.zeros(length_of_s)
            x = 0
            for n in range(length_of_s):
                p[n] = np.sin(2 * np.pi * x)
                delta = vco[n] / fs
                x += delta
                if x >= 1:
                    x -= 1

            vca = cosine_envelope(fs, VCA_delay[i], VCA_A[i], VCA_S[i], VCA_R[i], VCA_gate[i], VCA_duration[i])
            for n in range(length_of_s):
                vca[n] = VCA_offset[i] + vca[n] * VCA_depth[i]

            w = np.zeros(length_of_s)
            for n in range(length_of_s):
                w[n] = np.random.rand() * 2 - 1

            fc = 40
            Q = 1 / np.sqrt(2)
            a, b = LPF(fs, fc, Q)
            shimmer = filter(a, b, w)
            shimmer /= np.max(np.abs(shimmer))

            p1 = 1
            p2 = -0.3
            p3 = 50 / 12
            p4 = 0.8
            shimmer_depth = 1 / (1 + np.exp(-(VCO_offset[i] / f0 - p1) / p3)) * p4 + p2

            for n in range(length_of_s):
                vca[n] *= 1 + shimmer[n] * shimmer_depth
                if vca[n] < 0:
                    vca[n] = 0

            for n in range(length_of_s):
                s[n] += p[n] * vca[n]

    s *= velocity / 127 / np.max(np.abs(s))

    return s

def clarinet(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    s = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    number_of_partial = 30

    a = np.array([[0.302, 0.016, 0.205, 0.034, 0.214, 0.051, 0.119, 0.092, 0.074, 0.037, 0.068, 0.008, 0.011, 0.015, 0.022, 0.010, 0.015, 0.009, 0.014, 0.004, 0.006, 0.005, 0.004, 0.004, 0.003, 0.002, 0.003, 0.004, 0.002, 0.003],
                  [0.302, 0.011, 0.193, 0.022, 0.205, 0.031, 0.157, 0.039, 0.069, 0.022, 0.053, 0.011, 0.017, 0.015, 0.030, 0.013, 0.016, 0.014, 0.010, 0.007, 0.003, 0.005, 0.003, 0.004, 0.003, 0.002, 0.003, 0.002, 0.003, 0.002],
                  [0.302, 0.007, 0.200, 0.013, 0.204, 0.018, 0.168, 0.006, 0.045, 0.011, 0.034, 0.011, 0.030, 0.011, 0.030, 0.014, 0.017, 0.013, 0.010, 0.011, 0.003, 0.005, 0.004, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002, 0.001],
                  [0.302, 0.005, 0.216, 0.011, 0.192, 0.024, 0.107, 0.006, 0.040, 0.009, 0.030, 0.014, 0.030, 0.006, 0.013, 0.009, 0.014, 0.013, 0.012, 0.016, 0.005, 0.005, 0.004, 0.002, 0.002, 0.002, 0.001, 0.002, 0.002, 0.002],
                  [0.302, 0.005, 0.230, 0.012, 0.162, 0.030, 0.051, 0.009, 0.049, 0.011, 0.033, 0.018, 0.016, 0.004, 0.003, 0.005, 0.009, 0.014, 0.013, 0.014, 0.004, 0.005, 0.002, 0.001, 0.001, 0.002, 0.002, 0.002, 0.003, 0.001],
                  [0.302, 0.004, 0.274, 0.012, 0.162, 0.025, 0.034, 0.013, 0.041, 0.010, 0.020, 0.015, 0.011, 0.008, 0.005, 0.007, 0.005, 0.010, 0.009, 0.006, 0.003, 0.003, 0.001, 0.001, 0.001, 0.002, 0.002, 0.002, 0.002, 0.001],
                  [0.302, 0.004, 0.306, 0.011, 0.196, 0.017, 0.036, 0.023, 0.029, 0.007, 0.012, 0.012, 0.014, 0.012, 0.008, 0.010, 0.006, 0.009, 0.005, 0.003, 0.003, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.001, 0.001, 0.001],
                  [0.302, 0.004, 0.295, 0.009, 0.168, 0.018, 0.063, 0.030, 0.022, 0.009, 0.019, 0.012, 0.010, 0.013, 0.006, 0.012, 0.009, 0.006, 0.004, 0.003, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000],
                  [0.302, 0.003, 0.260, 0.010, 0.086, 0.024, 0.088, 0.018, 0.022, 0.014, 0.019, 0.008, 0.003, 0.012, 0.010, 0.010, 0.004, 0.002, 0.002, 0.001, 0.002, 0.002, 0.001, 0.002, 0.002, 0.000, 0.001, 0.000, 0.000, 0.000],
                  [0.302, 0.004, 0.227, 0.020, 0.075, 0.029, 0.078, 0.008, 0.019, 0.018, 0.009, 0.005, 0.003, 0.015, 0.012, 0.006, 0.001, 0.001, 0.000, 0.001, 0.002, 0.002, 0.001, 0.001, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000],
                  [0.302, 0.005, 0.229, 0.034, 0.119, 0.047, 0.044, 0.011, 0.015, 0.017, 0.003, 0.006, 0.006, 0.014, 0.008, 0.004, 0.001, 0.001, 0.002, 0.001, 0.002, 0.001, 0.001, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000],
                  [0.302, 0.005, 0.262, 0.036, 0.131, 0.061, 0.024, 0.012, 0.015, 0.016, 0.006, 0.010, 0.011, 0.010, 0.004, 0.003, 0.001, 0.002, 0.003, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.006, 0.339, 0.029, 0.115, 0.041, 0.027, 0.009, 0.015, 0.012, 0.010, 0.015, 0.012, 0.006, 0.002, 0.002, 0.001, 0.002, 0.002, 0.001, 0.001, 0.000, 0.000, 0.001, 0.000, 0.001, 0.000, 0.001, 0.000, 0.000],
                  [0.302, 0.005, 0.443, 0.026, 0.150, 0.015, 0.038, 0.007, 0.014, 0.007, 0.009, 0.014, 0.006, 0.002, 0.002, 0.002, 0.001, 0.002, 0.001, 0.000, 0.000, 0.000, 0.000, 0.001, 0.000, 0.001, 0.000, 0.001, 0.000, 0.000],
                  [0.302, 0.004, 0.422, 0.024, 0.175, 0.008, 0.032, 0.009, 0.013, 0.012, 0.013, 0.011, 0.003, 0.001, 0.002, 0.002, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.000, 0.001, 0.000, 0.001, 0.000, 0.000],
                  [0.302, 0.006, 0.255, 0.022, 0.113, 0.011, 0.029, 0.017, 0.011, 0.020, 0.014, 0.007, 0.001, 0.002, 0.003, 0.002, 0.001, 0.001, 0.000, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000],
                  [0.302, 0.006, 0.269, 0.034, 0.059, 0.017, 0.041, 0.023, 0.010, 0.018, 0.009, 0.003, 0.001, 0.001, 0.003, 0.002, 0.001, 0.001, 0.000, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.007, 0.394, 0.051, 0.041, 0.019, 0.036, 0.020, 0.011, 0.007, 0.006, 0.002, 0.001, 0.001, 0.002, 0.002, 0.001, 0.001, 0.000, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.001, 0.000, 0.001, 0.001],
                  [0.302, 0.030, 0.285, 0.053, 0.032, 0.017, 0.020, 0.011, 0.009, 0.002, 0.004, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.001, 0.001],
                  [0.302, 0.061, 0.142, 0.055, 0.035, 0.026, 0.018, 0.007, 0.007, 0.002, 0.003, 0.002, 0.001, 0.000, 0.001, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000],
                  [0.302, 0.061, 0.104, 0.053, 0.045, 0.045, 0.021, 0.007, 0.005, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.001, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.054, 0.070, 0.033, 0.043, 0.045, 0.013, 0.006, 0.002, 0.002, 0.000, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.057, 0.087, 0.018, 0.024, 0.027, 0.010, 0.007, 0.002, 0.001, 0.000, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.043, 0.113, 0.016, 0.014, 0.013, 0.012, 0.010, 0.003, 0.001, 0.000, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.027, 0.091, 0.014, 0.014, 0.008, 0.007, 0.008, 0.002, 0.000, 0.001, 0.000, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.021, 0.059, 0.009, 0.014, 0.009, 0.004, 0.004, 0.002, 0.000, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.018, 0.035, 0.004, 0.014, 0.010, 0.004, 0.002, 0.001, 0.001, 0.000, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.022, 0.033, 0.004, 0.017, 0.008, 0.005, 0.002, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.027, 0.040, 0.010, 0.021, 0.009, 0.005, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.042, 0.030, 0.017, 0.016, 0.011, 0.003, 0.001, 0.001, 0.001, 0.000, 0.001, 0.001, 0.000, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.060, 0.017, 0.025, 0.008, 0.010, 0.003, 0.002, 0.001, 0.001, 0.000, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.057, 0.018, 0.042, 0.009, 0.008, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.045, 0.022, 0.045, 0.011, 0.005, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.042, 0.022, 0.028, 0.013, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.046, 0.034, 0.020, 0.010, 0.004, 0.002, 0.002, 0.001, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.000, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.044, 0.055, 0.019, 0.006, 0.003, 0.003, 0.001, 0.001, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.000, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.035, 0.060, 0.019, 0.004, 0.002, 0.003, 0.001, 0.001, 0.001, 0.000, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.035, 0.044, 0.017, 0.002, 0.001, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.094, 0.020, 0.017, 0.004, 0.002, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.255, 0.019, 0.023, 0.008, 0.003, 0.003, 0.002, 0.003, 0.001, 0.002, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.426, 0.065, 0.023, 0.009, 0.004, 0.005, 0.006, 0.005, 0.003, 0.006, 0.005, 0.002, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.440, 0.103, 0.016, 0.012, 0.005, 0.008, 0.010, 0.005, 0.005, 0.007, 0.009, 0.002, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.400, 0.083, 0.015, 0.012, 0.005, 0.007, 0.010, 0.004, 0.005, 0.005, 0.006, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.593, 0.079, 0.019, 0.015, 0.008, 0.007, 0.007, 0.004, 0.004, 0.003, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.737, 0.126, 0.043, 0.033, 0.028, 0.015, 0.022, 0.008, 0.011, 0.006, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.302, 0.599, 0.169, 0.083, 0.051, 0.053, 0.023, 0.050, 0.012, 0.023, 0.011, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]])

    VCO_A = np.repeat(0, number_of_partial)
    VCO_D = np.repeat(0, number_of_partial)
    VCO_S = np.repeat(1, number_of_partial)
    VCO_R = np.repeat(0, number_of_partial)
    VCO_gate = np.repeat(duration, number_of_partial)
    VCO_duration = np.repeat(duration, number_of_partial)
    VCO_offset = np.arange(1, number_of_partial + 1) * f0
    VCO_depth = np.repeat(0, number_of_partial)

    p1 = 0
    p2 = -0.05
    p3 = 8000 / 12
    p4 = 0.1
    VCA_delay = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_delay[i] = 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3)) * p4 + p2

    for i in range(1, number_of_partial):
        VCA_delay[i] = VCA_delay[i] - VCA_delay[0]

    VCA_delay[0] = 0

    p1 = 0
    p2 = 0.05
    p3 = 8000 / 12
    p4 = 0.1
    VCA_A = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_A[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA_S = np.repeat(1, number_of_partial)

    p1 = 0
    p2 = 0.2
    p3 = 8000 / 12
    p4 = 0.4
    VCA_R = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_R[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA_gate = np.repeat(gate, number_of_partial)
    VCA_duration = np.repeat(duration, number_of_partial)
    VCA_offset = np.repeat(0, number_of_partial)
    VCA_depth = a[note_number - 50, :]

    np.random.seed(0)

    for i in range(number_of_partial):
        vco = ADSR(fs, VCO_A[i], VCO_D[i], VCO_S[i], VCO_R[i], VCO_gate[i], VCO_duration[i])
        for n in range(length_of_s):
            vco[n] = VCO_offset[i] + vco[n] * VCO_depth[i]

        w = np.zeros(length_of_s)
        for n in range(length_of_s):
            w[n] = np.random.rand() * 2 - 1

        fc = 40
        Q = 1 / np.sqrt(2)
        a, b = LPF(fs, fc, Q)
        jitter = filter(a, b, w)
        jitter /= np.max(np.abs(jitter))

        p1 = 108
        p2 = 1
        p3 = 150 / 12
        p4 = 20
        jitter_depth = 1 / (1 + np.exp(-(note_number - p1) / p3)) * p4 + p2

        for n in range(length_of_s):
            vco[n] += jitter[n] * jitter_depth

        if np.max(vco) < 20000:
            p = np.zeros(length_of_s)
            x = 0
            for n in range(length_of_s):
                p[n] = np.sin(2 * np.pi * x)
                delta = vco[n] / fs
                x += delta
                if x >= 1:
                    x -= 1

            vca = cosine_envelope(fs, VCA_delay[i], VCA_A[i], VCA_S[i], VCA_R[i], VCA_gate[i], VCA_duration[i])
            for n in range(length_of_s):
                vca[n] = VCA_offset[i] + vca[n] * VCA_depth[i]

            w = np.zeros(length_of_s)
            for n in range(length_of_s):
                w[n] = np.random.rand() * 2 - 1

            fc = 40
            Q = 1 / np.sqrt(2)
            a, b = LPF(fs, fc, Q)
            shimmer = filter(a, b, w)
            shimmer /= np.max(np.abs(shimmer))

            p1 = 1
            p2 = -0.3
            p3 = 100 / 12
            p4 = 0.8
            shimmer_depth = 1 / (1 + np.exp(-(VCO_offset[i] / f0 - p1) / p3)) * p4 + p2

            for n in range(length_of_s):
                vca[n] *= 1 + shimmer[n] * shimmer_depth
                if vca[n] < 0:
                    vca[n] = 0

            for n in range(length_of_s):
                s[n] += p[n] * vca[n]

    s *= velocity / 127 / np.max(np.abs(s))

    return s

def oboe(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    s = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    number_of_partial = 30

    a = np.array([[0.079, 0.107, 0.068, 0.138, 0.232, 0.145, 0.082, 0.028, 0.018, 0.014, 0.023, 0.026, 0.013, 0.022, 0.023, 0.010, 0.007, 0.004, 0.002, 0.003, 0.000, 0.001, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001],
                  [0.079, 0.106, 0.055, 0.134, 0.268, 0.142, 0.059, 0.023, 0.016, 0.019, 0.025, 0.021, 0.015, 0.017, 0.016, 0.012, 0.007, 0.004, 0.002, 0.003, 0.001, 0.002, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001],
                  [0.079, 0.107, 0.050, 0.149, 0.327, 0.135, 0.053, 0.020, 0.015, 0.026, 0.035, 0.018, 0.032, 0.019, 0.018, 0.021, 0.011, 0.008, 0.003, 0.004, 0.003, 0.003, 0.003, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000],
                  [0.079, 0.108, 0.076, 0.190, 0.306, 0.113, 0.055, 0.013, 0.014, 0.026, 0.041, 0.021, 0.046, 0.024, 0.021, 0.022, 0.014, 0.010, 0.004, 0.005, 0.003, 0.003, 0.002, 0.001, 0.001, 0.001, 0.001, 0.000, 0.001, 0.000],
                  [0.079, 0.092, 0.112, 0.258, 0.244, 0.068, 0.031, 0.007, 0.017, 0.027, 0.034, 0.023, 0.033, 0.017, 0.011, 0.011, 0.009, 0.006, 0.004, 0.004, 0.002, 0.002, 0.001, 0.001, 0.000, 0.001, 0.001, 0.000, 0.000, 0.000],
                  [0.079, 0.076, 0.135, 0.343, 0.193, 0.034, 0.012, 0.019, 0.034, 0.025, 0.028, 0.020, 0.014, 0.007, 0.004, 0.004, 0.003, 0.003, 0.003, 0.002, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.073, 0.155, 0.442, 0.126, 0.020, 0.013, 0.041, 0.052, 0.018, 0.025, 0.015, 0.010, 0.005, 0.003, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.000, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.081, 0.176, 0.458, 0.074, 0.008, 0.019, 0.057, 0.043, 0.021, 0.023, 0.013, 0.013, 0.007, 0.004, 0.003, 0.002, 0.003, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.098, 0.218, 0.304, 0.038, 0.006, 0.041, 0.064, 0.026, 0.029, 0.029, 0.015, 0.015, 0.009, 0.006, 0.006, 0.004, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000],
                  [0.079, 0.117, 0.289, 0.180, 0.025, 0.014, 0.078, 0.054, 0.038, 0.042, 0.033, 0.016, 0.016, 0.011, 0.009, 0.008, 0.003, 0.001, 0.001, 0.002, 0.001, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.000, 0.001, 0.001],
                  [0.079, 0.136, 0.364, 0.173, 0.029, 0.025, 0.084, 0.043, 0.048, 0.052, 0.025, 0.019, 0.016, 0.011, 0.009, 0.005, 0.002, 0.001, 0.001, 0.003, 0.002, 0.002, 0.001, 0.001, 0.000, 0.001, 0.001, 0.000, 0.001, 0.001],
                  [0.079, 0.150, 0.550, 0.126, 0.021, 0.047, 0.049, 0.045, 0.037, 0.042, 0.013, 0.016, 0.009, 0.007, 0.004, 0.002, 0.002, 0.001, 0.001, 0.002, 0.002, 0.001, 0.001, 0.001, 0.000, 0.000, 0.001, 0.001, 0.001, 0.001],
                  [0.079, 0.152, 0.666, 0.052, 0.016, 0.065, 0.023, 0.038, 0.025, 0.028, 0.008, 0.009, 0.003, 0.003, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.001, 0.001, 0.001, 0.000, 0.000, 0.001, 0.001, 0.001, 0.000],
                  [0.079, 0.167, 0.407, 0.033, 0.015, 0.073, 0.030, 0.021, 0.019, 0.020, 0.009, 0.006, 0.002, 0.002, 0.002, 0.001, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.001, 0.001, 0.001, 0.001, 0.000],
                  [0.079, 0.175, 0.139, 0.036, 0.014, 0.079, 0.037, 0.016, 0.019, 0.011, 0.008, 0.005, 0.002, 0.001, 0.002, 0.001, 0.002, 0.003, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.001, 0.001, 0.000],
                  [0.079, 0.140, 0.078, 0.020, 0.019, 0.052, 0.021, 0.015, 0.014, 0.005, 0.005, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.172, 0.047, 0.009, 0.027, 0.019, 0.008, 0.010, 0.005, 0.003, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.284, 0.022, 0.012, 0.027, 0.027, 0.011, 0.008, 0.004, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.420, 0.011, 0.019, 0.027, 0.043, 0.012, 0.006, 0.003, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.454, 0.012, 0.032, 0.026, 0.035, 0.010, 0.004, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.254, 0.019, 0.037, 0.019, 0.020, 0.009, 0.004, 0.002, 0.001, 0.001, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.092, 0.023, 0.031, 0.021, 0.014, 0.008, 0.002, 0.001, 0.001, 0.001, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.101, 0.022, 0.021, 0.024, 0.010, 0.006, 0.000, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.001, 0.000, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.086, 0.022, 0.015, 0.016, 0.005, 0.004, 0.000, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.039, 0.016, 0.015, 0.012, 0.002, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.026, 0.015, 0.012, 0.012, 0.002, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.045, 0.024, 0.012, 0.011, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.056, 0.024, 0.016, 0.011, 0.001, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.035, 0.015, 0.013, 0.007, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.015, 0.010, 0.011, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.020, 0.007, 0.009, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.045, 0.029, 0.006, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.085, 0.063, 0.008, 0.002, 0.002, 0.002, 0.003, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.085, 0.053, 0.008, 0.002, 0.002, 0.002, 0.003, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.079, 0.048, 0.024, 0.005, 0.001, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]])

    VCO_A = np.repeat(0, number_of_partial)
    VCO_D = np.repeat(0, number_of_partial)
    VCO_S = np.repeat(1, number_of_partial)
    VCO_R = np.repeat(0, number_of_partial)
    VCO_gate = np.repeat(duration, number_of_partial)
    VCO_duration = np.repeat(duration, number_of_partial)
    VCO_offset = np.arange(1, number_of_partial + 1) * f0
    VCO_depth = np.repeat(0, number_of_partial)

    p1 = 0
    p2 = -0.05
    p3 = 8000 / 12
    p4 = 0.1
    VCA_delay = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_delay[i] = 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3)) * p4 + p2

    for i in range(1, number_of_partial):
        VCA_delay[i] = VCA_delay[i] - VCA_delay[0]

    VCA_delay[0] = 0

    p1 = 0
    p2 = 0.05
    p3 = 8000 / 12
    p4 = 0.1
    VCA_A = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_A[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA_S = np.repeat(1, number_of_partial)

    p1 = 0
    p2 = 0.2
    p3 = 8000 / 12
    p4 = 0.4
    VCA_R = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_R[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA_gate = np.repeat(gate, number_of_partial)
    VCA_duration = np.repeat(duration, number_of_partial)
    VCA_offset = np.repeat(0, number_of_partial)
    VCA_depth = a[note_number - 58, :]

    np.random.seed(0)

    for i in range(number_of_partial):
        vco = ADSR(fs, VCO_A[i], VCO_D[i], VCO_S[i], VCO_R[i], VCO_gate[i], VCO_duration[i])
        for n in range(length_of_s):
            vco[n] = VCO_offset[i] + vco[n] * VCO_depth[i]

        w = np.zeros(length_of_s)
        for n in range(length_of_s):
            w[n] = np.random.rand() * 2 - 1

        fc = 40
        Q = 1 / np.sqrt(2)
        a, b = LPF(fs, fc, Q)
        jitter = filter(a, b, w)
        jitter /= np.max(np.abs(jitter))

        p1 = 108
        p2 = 1
        p3 = 150 / 12
        p4 = 20
        jitter_depth = 1 / (1 + np.exp(-(note_number - p1) / p3)) * p4 + p2

        for n in range(length_of_s):
            vco[n] += jitter[n] * jitter_depth

        if np.max(vco) < 20000:
            p = np.zeros(length_of_s)
            x = 0
            for n in range(length_of_s):
                p[n] = np.sin(2 * np.pi * x)
                delta = vco[n] / fs
                x += delta
                if x >= 1:
                    x -= 1

            vca = cosine_envelope(fs, VCA_delay[i], VCA_A[i], VCA_S[i], VCA_R[i], VCA_gate[i], VCA_duration[i])
            for n in range(length_of_s):
                vca[n] = VCA_offset[i] + vca[n] * VCA_depth[i]

            w = np.zeros(length_of_s)
            for n in range(length_of_s):
                w[n] = np.random.rand() * 2 - 1

            fc = 40
            Q = 1 / np.sqrt(2)
            a, b = LPF(fs, fc, Q)
            shimmer = filter(a, b, w)
            shimmer /= np.max(np.abs(shimmer))

            p1 = 1
            p2 = -0.3
            p3 = 100 / 12
            p4 = 0.8
            shimmer_depth = 1 / (1 + np.exp(-(VCO_offset[i] / f0 - p1) / p3)) * p4 + p2

            for n in range(length_of_s):
                vca[n] *= 1 + shimmer[n] * shimmer_depth
                if vca[n] < 0:
                    vca[n] = 0

            for n in range(length_of_s):
                s[n] += p[n] * vca[n]

    s *= velocity / 127 / np.max(np.abs(s))

    return s

def bassoon(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    s = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    number_of_partial = 30

    a = np.array([[0.069, 0.095, 0.101, 0.094, 0.146, 0.153, 0.176, 0.194, 0.214, 0.144, 0.068, 0.046, 0.041, 0.036, 0.030, 0.023, 0.032, 0.043, 0.039, 0.031, 0.022, 0.027, 0.017, 0.011, 0.015, 0.014, 0.012, 0.009, 0.011, 0.012],
                  [0.069, 0.092, 0.104, 0.095, 0.143, 0.186, 0.256, 0.249, 0.237, 0.143, 0.072, 0.054, 0.060, 0.043, 0.025, 0.022, 0.028, 0.039, 0.036, 0.042, 0.030, 0.037, 0.019, 0.013, 0.020, 0.018, 0.013, 0.012, 0.014, 0.018],
                  [0.069, 0.094, 0.105, 0.093, 0.122, 0.207, 0.374, 0.259, 0.220, 0.105, 0.053, 0.034, 0.044, 0.036, 0.022, 0.027, 0.033, 0.038, 0.035, 0.036, 0.027, 0.025, 0.016, 0.015, 0.016, 0.015, 0.012, 0.010, 0.009, 0.013],
                  [0.069, 0.100, 0.113, 0.116, 0.119, 0.262, 0.429, 0.195, 0.168, 0.073, 0.038, 0.018, 0.017, 0.021, 0.025, 0.032, 0.035, 0.034, 0.027, 0.018, 0.013, 0.008, 0.010, 0.010, 0.008, 0.008, 0.008, 0.005, 0.004, 0.005],
                  [0.069, 0.098, 0.130, 0.161, 0.158, 0.349, 0.348, 0.122, 0.068, 0.050, 0.035, 0.022, 0.012, 0.026, 0.032, 0.036, 0.032, 0.027, 0.015, 0.008, 0.007, 0.007, 0.007, 0.006, 0.005, 0.004, 0.005, 0.003, 0.004, 0.003],
                  [0.069, 0.095, 0.146, 0.204, 0.204, 0.431, 0.270, 0.108, 0.015, 0.029, 0.031, 0.030, 0.014, 0.045, 0.041, 0.045, 0.031, 0.020, 0.012, 0.006, 0.009, 0.007, 0.005, 0.005, 0.004, 0.002, 0.005, 0.003, 0.003, 0.003],
                  [0.069, 0.098, 0.152, 0.222, 0.253, 0.533, 0.251, 0.109, 0.035, 0.030, 0.035, 0.033, 0.018, 0.053, 0.040, 0.041, 0.022, 0.013, 0.010, 0.006, 0.009, 0.004, 0.005, 0.008, 0.005, 0.002, 0.004, 0.002, 0.002, 0.003],
                  [0.069, 0.106, 0.150, 0.231, 0.331, 0.548, 0.188, 0.082, 0.047, 0.056, 0.038, 0.027, 0.031, 0.045, 0.026, 0.020, 0.011, 0.007, 0.005, 0.005, 0.007, 0.004, 0.007, 0.010, 0.005, 0.002, 0.002, 0.003, 0.003, 0.004],
                  [0.069, 0.113, 0.145, 0.247, 0.455, 0.430, 0.094, 0.075, 0.058, 0.073, 0.025, 0.026, 0.045, 0.037, 0.017, 0.009, 0.006, 0.004, 0.004, 0.004, 0.007, 0.004, 0.007, 0.006, 0.003, 0.002, 0.002, 0.003, 0.004, 0.003],
                  [0.069, 0.105, 0.127, 0.228, 0.637, 0.275, 0.038, 0.078, 0.066, 0.054, 0.016, 0.024, 0.039, 0.029, 0.016, 0.011, 0.005, 0.003, 0.005, 0.004, 0.005, 0.003, 0.004, 0.002, 0.003, 0.002, 0.003, 0.003, 0.003, 0.003],
                  [0.069, 0.094, 0.109, 0.256, 0.674, 0.140, 0.071, 0.088, 0.051, 0.028, 0.026, 0.036, 0.031, 0.020, 0.014, 0.008, 0.006, 0.006, 0.008, 0.003, 0.002, 0.003, 0.002, 0.003, 0.003, 0.004, 0.003, 0.003, 0.002, 0.003],
                  [0.069, 0.090, 0.103, 0.388, 0.499, 0.057, 0.114, 0.112, 0.030, 0.029, 0.037, 0.049, 0.030, 0.015, 0.008, 0.006, 0.008, 0.012, 0.010, 0.002, 0.002, 0.003, 0.002, 0.004, 0.004, 0.005, 0.002, 0.003, 0.003, 0.002],
                  [0.069, 0.099, 0.150, 0.608, 0.375, 0.057, 0.080, 0.108, 0.020, 0.054, 0.044, 0.031, 0.022, 0.009, 0.008, 0.008, 0.011, 0.010, 0.006, 0.003, 0.003, 0.003, 0.003, 0.004, 0.005, 0.005, 0.003, 0.003, 0.004, 0.002],
                  [0.069, 0.112, 0.266, 0.769, 0.251, 0.104, 0.112, 0.073, 0.024, 0.072, 0.050, 0.022, 0.016, 0.006, 0.008, 0.014, 0.011, 0.004, 0.003, 0.004, 0.003, 0.005, 0.003, 0.005, 0.005, 0.006, 0.005, 0.005, 0.004, 0.001],
                  [0.069, 0.124, 0.401, 0.604, 0.080, 0.114, 0.163, 0.044, 0.047, 0.071, 0.044, 0.024, 0.014, 0.008, 0.010, 0.012, 0.005, 0.002, 0.004, 0.005, 0.004, 0.006, 0.002, 0.005, 0.005, 0.010, 0.004, 0.004, 0.004, 0.002],
                  [0.069, 0.142, 0.572, 0.439, 0.045, 0.111, 0.099, 0.058, 0.067, 0.059, 0.028, 0.017, 0.014, 0.015, 0.013, 0.005, 0.002, 0.002, 0.004, 0.003, 0.005, 0.006, 0.004, 0.007, 0.003, 0.007, 0.004, 0.003, 0.004, 0.002],
                  [0.069, 0.164, 0.704, 0.447, 0.156, 0.188, 0.042, 0.096, 0.065, 0.045, 0.017, 0.016, 0.019, 0.021, 0.008, 0.004, 0.004, 0.004, 0.007, 0.003, 0.005, 0.005, 0.007, 0.008, 0.003, 0.005, 0.006, 0.003, 0.003, 0.002],
                  [0.069, 0.177, 0.752, 0.285, 0.210, 0.228, 0.039, 0.095, 0.046, 0.037, 0.011, 0.019, 0.022, 0.018, 0.005, 0.006, 0.006, 0.005, 0.009, 0.003, 0.004, 0.005, 0.008, 0.007, 0.005, 0.006, 0.004, 0.004, 0.003, 0.002],
                  [0.069, 0.170, 0.767, 0.111, 0.113, 0.123, 0.054, 0.062, 0.030, 0.022, 0.014, 0.016, 0.014, 0.009, 0.006, 0.006, 0.004, 0.003, 0.006, 0.005, 0.003, 0.005, 0.006, 0.005, 0.003, 0.004, 0.002, 0.003, 0.003, 0.002],
                  [0.069, 0.164, 0.658, 0.096, 0.053, 0.040, 0.066, 0.034, 0.018, 0.011, 0.017, 0.011, 0.006, 0.005, 0.006, 0.005, 0.003, 0.002, 0.004, 0.005, 0.003, 0.004, 0.005, 0.003, 0.001, 0.002, 0.001, 0.001, 0.002, 0.001],
                  [0.069, 0.169, 0.437, 0.088, 0.054, 0.041, 0.045, 0.015, 0.009, 0.010, 0.012, 0.005, 0.003, 0.004, 0.004, 0.003, 0.002, 0.001, 0.003, 0.003, 0.002, 0.002, 0.003, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
                  [0.069, 0.172, 0.254, 0.054, 0.036, 0.050, 0.022, 0.005, 0.005, 0.011, 0.007, 0.002, 0.002, 0.004, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.001],
                  [0.069, 0.262, 0.147, 0.035, 0.024, 0.059, 0.020, 0.003, 0.008, 0.010, 0.005, 0.003, 0.003, 0.004, 0.002, 0.002, 0.002, 0.002, 0.001, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.001],
                  [0.069, 0.418, 0.081, 0.048, 0.035, 0.063, 0.021, 0.006, 0.012, 0.005, 0.004, 0.005, 0.004, 0.003, 0.002, 0.002, 0.002, 0.004, 0.001, 0.001, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.000, 0.001, 0.000, 0.001],
                  [0.069, 0.472, 0.052, 0.056, 0.058, 0.045, 0.011, 0.011, 0.012, 0.003, 0.002, 0.005, 0.004, 0.002, 0.002, 0.001, 0.001, 0.003, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000, 0.001],
                  [0.069, 0.474, 0.041, 0.038, 0.064, 0.024, 0.008, 0.013, 0.007, 0.003, 0.001, 0.004, 0.005, 0.002, 0.002, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.001, 0.001, 0.000, 0.000, 0.000],
                  [0.069, 0.406, 0.031, 0.021, 0.052, 0.013, 0.010, 0.009, 0.003, 0.002, 0.001, 0.003, 0.005, 0.002, 0.002, 0.002, 0.000, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.001, 0.001, 0.000, 0.000, 0.000],
                  [0.069, 0.267, 0.033, 0.025, 0.033, 0.007, 0.008, 0.004, 0.002, 0.002, 0.001, 0.003, 0.003, 0.001, 0.001, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.069, 0.260, 0.043, 0.051, 0.017, 0.008, 0.008, 0.002, 0.002, 0.002, 0.001, 0.003, 0.002, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.069, 0.303, 0.043, 0.073, 0.010, 0.011, 0.008, 0.002, 0.005, 0.002, 0.002, 0.002, 0.003, 0.000, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.069, 0.238, 0.028, 0.065, 0.006, 0.012, 0.005, 0.002, 0.006, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.000, 0.001, 0.000, 0.000, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000],
                  [0.069, 0.131, 0.021, 0.037, 0.006, 0.010, 0.003, 0.002, 0.004, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.001, 0.000, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.069, 0.096, 0.037, 0.015, 0.010, 0.004, 0.002, 0.001, 0.002, 0.001, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.069, 0.090, 0.053, 0.006, 0.011, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.069, 0.069, 0.051, 0.005, 0.009, 0.001, 0.002, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.069, 0.068, 0.042, 0.005, 0.004, 0.001, 0.002, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.069, 0.050, 0.023, 0.006, 0.002, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.069, 0.021, 0.010, 0.007, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.069, 0.011, 0.010, 0.005, 0.002, 0.001, 0.001, 0.001, 0.000, 0.001, 0.001, 0.000, 0.000, 0.001, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.069, 0.015, 0.006, 0.002, 0.002, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.069, 0.024, 0.005, 0.001, 0.003, 0.001, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]])

    VCO_A = np.repeat(0, number_of_partial)
    VCO_D = np.repeat(0, number_of_partial)
    VCO_S = np.repeat(1, number_of_partial)
    VCO_R = np.repeat(0, number_of_partial)
    VCO_gate = np.repeat(duration, number_of_partial)
    VCO_duration = np.repeat(duration, number_of_partial)
    VCO_offset = np.arange(1, number_of_partial + 1) * f0
    VCO_depth = np.repeat(0, number_of_partial)

    p1 = 0
    p2 = -0.05
    p3 = 8000 / 12
    p4 = 0.1
    VCA_delay = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_delay[i] = 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3)) * p4 + p2

    for i in range(1, number_of_partial):
        VCA_delay[i] = VCA_delay[i] - VCA_delay[0]

    VCA_delay[0] = 0

    p1 = 0
    p2 = 0.05
    p3 = 8000 / 12
    p4 = 0.1
    VCA_A = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_A[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA_S = np.repeat(1, number_of_partial)

    p1 = 0
    p2 = 0.2
    p3 = 8000 / 12
    p4 = 0.4
    VCA_R = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_R[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA_gate = np.repeat(gate, number_of_partial)
    VCA_duration = np.repeat(duration, number_of_partial)
    VCA_offset = np.repeat(0, number_of_partial)
    VCA_depth = a[note_number - 34, :]

    np.random.seed(0)

    for i in range(number_of_partial):
        vco = ADSR(fs, VCO_A[i], VCO_D[i], VCO_S[i], VCO_R[i], VCO_gate[i], VCO_duration[i])
        for n in range(length_of_s):
            vco[n] = VCO_offset[i] + vco[n] * VCO_depth[i]

        w = np.zeros(length_of_s)
        for n in range(length_of_s):
            w[n] = np.random.rand() * 2 - 1

        fc = 40
        Q = 1 / np.sqrt(2)
        a, b = LPF(fs, fc, Q)
        jitter = filter(a, b, w)
        jitter /= np.max(np.abs(jitter))

        p1 = 108
        p2 = 1
        p3 = 150 / 12
        p4 = 20
        jitter_depth = 1 / (1 + np.exp(-(note_number - p1) / p3)) * p4 + p2

        for n in range(length_of_s):
            vco[n] += jitter[n] * jitter_depth

        if np.max(vco) < 20000:
            p = np.zeros(length_of_s)
            x = 0
            for n in range(length_of_s):
                p[n] = np.sin(2 * np.pi * x)
                delta = vco[n] / fs
                x += delta
                if x >= 1:
                    x -= 1

            vca = cosine_envelope(fs, VCA_delay[i], VCA_A[i], VCA_S[i], VCA_R[i], VCA_gate[i], VCA_duration[i])
            for n in range(length_of_s):
                vca[n] = VCA_offset[i] + vca[n] * VCA_depth[i]

            w = np.zeros(length_of_s)
            for n in range(length_of_s):
                w[n] = np.random.rand() * 2 - 1

            fc = 40
            Q = 1 / np.sqrt(2)
            a, b = LPF(fs, fc, Q)
            shimmer = filter(a, b, w)
            shimmer /= np.max(np.abs(shimmer))

            p1 = 1
            p2 = -0.3
            p3 = 100 / 12
            p4 = 0.8
            shimmer_depth = 1 / (1 + np.exp(-(VCO_offset[i] / f0 - p1) / p3)) * p4 + p2

            for n in range(length_of_s):
                vca[n] *= 1 + shimmer[n] * shimmer_depth
                if vca[n] < 0:
                    vca[n] = 0

            for n in range(length_of_s):
                s[n] += p[n] * vca[n]

    s *= velocity / 127 / np.max(np.abs(s))

    return s

def saxophone(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    s = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    number_of_partial = 30

    a = np.array([[0.239, 0.363, 0.172, 0.381, 0.506, 0.233, 0.106, 0.025, 0.054, 0.070, 0.068, 0.065, 0.034, 0.046, 0.055, 0.061, 0.069, 0.065, 0.051, 0.049, 0.045, 0.040, 0.032, 0.030, 0.019, 0.013, 0.008, 0.008, 0.008, 0.006],
                  [0.239, 0.377, 0.226, 0.368, 0.543, 0.204, 0.113, 0.057, 0.051, 0.058, 0.056, 0.059, 0.040, 0.053, 0.059, 0.062, 0.066, 0.052, 0.046, 0.042, 0.035, 0.035, 0.032, 0.027, 0.016, 0.013, 0.009, 0.007, 0.006, 0.005],
                  [0.239, 0.360, 0.238, 0.372, 0.452, 0.123, 0.067, 0.094, 0.059, 0.048, 0.034, 0.037, 0.039, 0.061, 0.068, 0.052, 0.047, 0.035, 0.031, 0.027, 0.020, 0.019, 0.016, 0.014, 0.011, 0.009, 0.008, 0.005, 0.004, 0.003],
                  [0.239, 0.316, 0.215, 0.328, 0.258, 0.085, 0.059, 0.094, 0.056, 0.034, 0.027, 0.029, 0.035, 0.064, 0.061, 0.038, 0.032, 0.029, 0.023, 0.017, 0.011, 0.008, 0.006, 0.007, 0.007, 0.005, 0.006, 0.003, 0.003, 0.002],
                  [0.239, 0.303, 0.261, 0.235, 0.097, 0.099, 0.086, 0.062, 0.035, 0.023, 0.032, 0.028, 0.034, 0.047, 0.036, 0.024, 0.023, 0.024, 0.019, 0.010, 0.006, 0.007, 0.006, 0.006, 0.004, 0.003, 0.004, 0.003, 0.003, 0.002],
                  [0.239, 0.309, 0.297, 0.161, 0.052, 0.075, 0.061, 0.029, 0.023, 0.024, 0.031, 0.021, 0.031, 0.027, 0.022, 0.014, 0.015, 0.017, 0.011, 0.006, 0.004, 0.007, 0.004, 0.003, 0.001, 0.002, 0.003, 0.004, 0.003, 0.002],
                  [0.239, 0.307, 0.203, 0.087, 0.068, 0.048, 0.032, 0.021, 0.028, 0.031, 0.021, 0.021, 0.025, 0.023, 0.020, 0.013, 0.011, 0.012, 0.006, 0.005, 0.004, 0.004, 0.002, 0.002, 0.001, 0.002, 0.002, 0.003, 0.002, 0.002],
                  [0.239, 0.309, 0.096, 0.050, 0.076, 0.055, 0.030, 0.027, 0.034, 0.032, 0.019, 0.025, 0.030, 0.022, 0.018, 0.012, 0.009, 0.008, 0.005, 0.004, 0.003, 0.003, 0.002, 0.001, 0.001, 0.002, 0.003, 0.002, 0.002, 0.001],
                  [0.239, 0.326, 0.053, 0.065, 0.083, 0.062, 0.024, 0.032, 0.038, 0.025, 0.024, 0.031, 0.037, 0.022, 0.013, 0.007, 0.007, 0.006, 0.005, 0.003, 0.001, 0.002, 0.002, 0.002, 0.002, 0.002, 0.003, 0.003, 0.003, 0.002],
                  [0.239, 0.320, 0.039, 0.072, 0.070, 0.063, 0.026, 0.031, 0.039, 0.024, 0.028, 0.036, 0.032, 0.019, 0.007, 0.004, 0.004, 0.003, 0.004, 0.001, 0.001, 0.002, 0.002, 0.003, 0.002, 0.002, 0.002, 0.003, 0.003, 0.003],
                  [0.239, 0.238, 0.035, 0.085, 0.042, 0.052, 0.029, 0.030, 0.033, 0.026, 0.025, 0.032, 0.021, 0.010, 0.004, 0.003, 0.003, 0.002, 0.002, 0.001, 0.001, 0.002, 0.002, 0.002, 0.001, 0.002, 0.003, 0.003, 0.002, 0.003],
                  [0.239, 0.125, 0.028, 0.102, 0.029, 0.036, 0.025, 0.028, 0.030, 0.024, 0.020, 0.019, 0.011, 0.004, 0.004, 0.003, 0.002, 0.001, 0.001, 0.001, 0.002, 0.003, 0.002, 0.001, 0.001, 0.002, 0.002, 0.002, 0.002, 0.003],
                  [0.239, 0.067, 0.026, 0.077, 0.038, 0.031, 0.025, 0.028, 0.027, 0.021, 0.019, 0.009, 0.007, 0.004, 0.003, 0.002, 0.001, 0.001, 0.001, 0.001, 0.002, 0.003, 0.003, 0.002, 0.001, 0.002, 0.002, 0.001, 0.002, 0.002],
                  [0.239, 0.080, 0.047, 0.054, 0.060, 0.022, 0.032, 0.031, 0.021, 0.018, 0.017, 0.005, 0.006, 0.005, 0.003, 0.002, 0.001, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.002, 0.002, 0.001, 0.001, 0.001],
                  [0.239, 0.107, 0.073, 0.064, 0.076, 0.018, 0.035, 0.038, 0.032, 0.022, 0.010, 0.008, 0.008, 0.008, 0.005, 0.002, 0.003, 0.004, 0.003, 0.003, 0.003, 0.001, 0.002, 0.002, 0.001, 0.003, 0.002, 0.002, 0.001, 0.001],
                  [0.239, 0.119, 0.094, 0.062, 0.070, 0.022, 0.027, 0.039, 0.039, 0.023, 0.006, 0.011, 0.008, 0.007, 0.005, 0.003, 0.004, 0.005, 0.003, 0.004, 0.003, 0.001, 0.003, 0.002, 0.001, 0.002, 0.001, 0.003, 0.002, 0.001],
                  [0.239, 0.113, 0.111, 0.058, 0.047, 0.019, 0.022, 0.027, 0.023, 0.012, 0.005, 0.007, 0.005, 0.003, 0.004, 0.003, 0.003, 0.003, 0.001, 0.002, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.002, 0.001, 0.000],
                  [0.239, 0.082, 0.082, 0.066, 0.034, 0.026, 0.024, 0.020, 0.010, 0.006, 0.004, 0.003, 0.003, 0.002, 0.003, 0.003, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000],
                  [0.239, 0.051, 0.050, 0.051, 0.031, 0.035, 0.025, 0.019, 0.007, 0.006, 0.005, 0.002, 0.003, 0.003, 0.002, 0.002, 0.001, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.001, 0.000],
                  [0.239, 0.040, 0.056, 0.027, 0.027, 0.029, 0.024, 0.013, 0.006, 0.003, 0.004, 0.002, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.002, 0.001, 0.001, 0.000, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.239, 0.040, 0.063, 0.019, 0.025, 0.021, 0.018, 0.006, 0.005, 0.003, 0.002, 0.002, 0.003, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                  [0.239, 0.044, 0.094, 0.015, 0.035, 0.025, 0.013, 0.007, 0.004, 0.003, 0.002, 0.003, 0.002, 0.002, 0.001, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.001, 0.001, 0.001, 0.000, 0.000],
                  [0.239, 0.067, 0.174, 0.024, 0.074, 0.044, 0.015, 0.010, 0.004, 0.004, 0.004, 0.005, 0.002, 0.003, 0.002, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.002, 0.001, 0.001, 0.000, 0.001],
                  [0.239, 0.100, 0.431, 0.117, 0.173, 0.095, 0.038, 0.022, 0.009, 0.011, 0.011, 0.010, 0.014, 0.004, 0.006, 0.005, 0.004, 0.005, 0.004, 0.003, 0.005, 0.002, 0.002, 0.002, 0.002, 0.002, 0.003, 0.001, 0.001, 0.001],
                  [0.239, 0.103, 0.614, 0.221, 0.227, 0.123, 0.055, 0.038, 0.015, 0.018, 0.015, 0.017, 0.028, 0.007, 0.011, 0.009, 0.007, 0.007, 0.007, 0.004, 0.008, 0.003, 0.003, 0.002, 0.003, 0.002, 0.004, 0.002, 0.001, 0.001],
                  [0.239, 0.093, 0.353, 0.215, 0.150, 0.071, 0.033, 0.028, 0.011, 0.014, 0.010, 0.012, 0.019, 0.008, 0.009, 0.010, 0.005, 0.004, 0.004, 0.003, 0.005, 0.002, 0.002, 0.002, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001],
                  [0.239, 0.098, 0.089, 0.207, 0.090, 0.024, 0.010, 0.012, 0.015, 0.010, 0.005, 0.004, 0.004, 0.008, 0.006, 0.007, 0.003, 0.002, 0.001, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.001],
                  [0.239, 0.126, 0.065, 0.191, 0.074, 0.015, 0.005, 0.008, 0.019, 0.010, 0.004, 0.005, 0.004, 0.009, 0.005, 0.005, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.000, 0.001, 0.001, 0.001, 0.000, 0.001],
                  [0.239, 0.155, 0.076, 0.111, 0.040, 0.011, 0.004, 0.006, 0.011, 0.008, 0.003, 0.007, 0.007, 0.008, 0.003, 0.005, 0.002, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.000, 0.001, 0.001, 0.001, 0.000, 0.001],
                  [0.239, 0.128, 0.074, 0.062, 0.019, 0.009, 0.007, 0.007, 0.006, 0.008, 0.004, 0.006, 0.004, 0.004, 0.002, 0.003, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000],
                  [0.239, 0.075, 0.061, 0.056, 0.014, 0.008, 0.009, 0.008, 0.007, 0.006, 0.005, 0.005, 0.002, 0.002, 0.001, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000],
                  [0.239, 0.054, 0.064, 0.057, 0.014, 0.007, 0.007, 0.008, 0.008, 0.004, 0.004, 0.004, 0.002, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000]])

    VCO_A = np.repeat(0, number_of_partial)
    VCO_D = np.repeat(0, number_of_partial)
    VCO_S = np.repeat(1, number_of_partial)
    VCO_R = np.repeat(0, number_of_partial)
    VCO_gate = np.repeat(duration, number_of_partial)
    VCO_duration = np.repeat(duration, number_of_partial)
    VCO_offset = np.arange(1, number_of_partial + 1) * f0
    VCO_depth = np.repeat(0, number_of_partial)

    p1 = 0
    p2 = -0.05
    p3 = 8000 / 12
    p4 = 0.1
    VCA_delay = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_delay[i] = 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3)) * p4 + p2

    for i in range(1, number_of_partial):
        VCA_delay[i] = VCA_delay[i] - VCA_delay[0]

    VCA_delay[0] = 0

    p1 = 0
    p2 = 0.05
    p3 = 8000 / 12
    p4 = 0.1
    VCA_A = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_A[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA_S = np.repeat(1, number_of_partial)

    p1 = 0
    p2 = 0.2
    p3 = 8000 / 12
    p4 = 0.4
    VCA_R = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_R[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA_gate = np.repeat(gate, number_of_partial)
    VCA_duration = np.repeat(duration, number_of_partial)
    VCA_offset = np.repeat(0, number_of_partial)
    VCA_depth = a[note_number - 49, :]

    np.random.seed(0)

    for i in range(number_of_partial):
        vco = ADSR(fs, VCO_A[i], VCO_D[i], VCO_S[i], VCO_R[i], VCO_gate[i], VCO_duration[i])
        for n in range(length_of_s):
            vco[n] = VCO_offset[i] + vco[n] * VCO_depth[i]

        w = np.zeros(length_of_s)
        for n in range(length_of_s):
            w[n] = np.random.rand() * 2 - 1

        fc = 40
        Q = 1 / np.sqrt(2)
        a, b = LPF(fs, fc, Q)
        jitter = filter(a, b, w)
        jitter /= np.max(np.abs(jitter))

        p1 = 108
        p2 = 1
        p3 = 150 / 12
        p4 = 20
        jitter_depth = 1 / (1 + np.exp(-(note_number - p1) / p3)) * p4 + p2

        for n in range(length_of_s):
            vco[n] += jitter[n] * jitter_depth

        if np.max(vco) < 20000:
            p = np.zeros(length_of_s)
            x = 0
            for n in range(length_of_s):
                p[n] = np.sin(2 * np.pi * x)
                delta = vco[n] / fs
                x += delta
                if x >= 1:
                    x -= 1

            vca = cosine_envelope(fs, VCA_delay[i], VCA_A[i], VCA_S[i], VCA_R[i], VCA_gate[i], VCA_duration[i])
            for n in range(length_of_s):
                vca[n] = VCA_offset[i] + vca[n] * VCA_depth[i]

            w = np.zeros(length_of_s)
            for n in range(length_of_s):
                w[n] = np.random.rand() * 2 - 1

            fc = 40
            Q = 1 / np.sqrt(2)
            a, b = LPF(fs, fc, Q)
            shimmer = filter(a, b, w)
            shimmer /= np.max(np.abs(shimmer))

            p1 = 1
            p2 = -0.3
            p3 = 100 / 12
            p4 = 0.8
            shimmer_depth = 1 / (1 + np.exp(-(VCO_offset[i] / f0 - p1) / p3)) * p4 + p2

            for n in range(length_of_s):
                vca[n] *= 1 + shimmer[n] * shimmer_depth
                if vca[n] < 0:
                    vca[n] = 0

            for n in range(length_of_s):
                s[n] += p[n] * vca[n]

    s *= velocity / 127 / np.max(np.abs(s))

    return s

def trumpet(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    sa = np.zeros(length_of_s)
    sb = np.zeros(length_of_s)
    s0 = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    number_of_partial= int(6000 / f0)
    if number_of_partial < 5:
        number_of_partial = 5
    elif number_of_partial > 30:
        number_of_partial = 30

    VCO_A = np.repeat(0, number_of_partial)
    VCO_D = np.repeat(0.15, number_of_partial)
    VCO_S = np.repeat(0, number_of_partial)
    VCO_R = np.repeat(0.15, number_of_partial)
    VCO_gate = np.repeat(duration, number_of_partial)
    VCO_duration = np.repeat(duration, number_of_partial)
    VCO_offset = np.arange(1, number_of_partial + 1) * f0
    VCO_depth = np.arange(1, number_of_partial + 1) * (-f0 * 0.15)

    VCA0_A = np.repeat(0.01, number_of_partial)
    VCA0_D = np.repeat(0.02, number_of_partial)
    VCA0_S = np.repeat(0, number_of_partial)
    VCA0_R = np.repeat(0.02, number_of_partial)
    VCA0_gate = np.repeat(gate, number_of_partial)
    VCA0_duration = np.repeat(duration, number_of_partial)
    VCA0_offset = np.repeat(0, number_of_partial)
    VCA0_depth = np.repeat(1, number_of_partial)

    p1 = 3000
    p2 = 0
    p3 = 3000 / 12
    p4 = 0.03
    VCA1_delay = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA1_delay[i] = 1 / (1 + np.exp(-((VCO_offset[i] + VCO_depth[i]) - p1) / p3)) * p4 + p2

    for i in range(1, number_of_partial):
        VCA1_delay[i] = VCA1_delay[i] - VCA1_delay[0]

    VCA1_delay[0] = 0

    p1 = 3000
    p2 = 0.02
    p3 = 3000 / 12
    p4 = 0.02
    VCA1_A = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA1_A[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA1_S = np.repeat(1, number_of_partial)

    p1 = 3000
    p2 = 0.2
    p3 = 3000 / 12
    p4 = 0.1
    VCA1_R = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA1_R[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA1_gate = np.repeat(gate, number_of_partial)
    VCA1_duration = np.repeat(duration, number_of_partial)
    VCA1_offset = np.repeat(0, number_of_partial)
    VCA1_depth = np.repeat(1, number_of_partial)

    np.random.seed(0)

    for i in range(number_of_partial):
        vco = ADSR(fs, VCO_A[i], VCO_D[i], VCO_S[i], VCO_R[i], VCO_gate[i], VCO_duration[i])
        for n in range(length_of_s):
            vco[n] = VCO_offset[i] + vco[n] * VCO_depth[i]

        w = np.zeros(length_of_s)
        for n in range(length_of_s):
            w[n] = np.random.rand() * 2 - 1

        fc = 40
        Q = 1 / np.sqrt(2)
        a, b = LPF(fs, fc, Q)
        jitter = filter(a, b, w)
        jitter /= np.max(np.abs(jitter))

        p1 = 108
        p2 = 1
        p3 = 150 / 12
        p4 = 20
        jitter_depth = 1 / (1 + np.exp(-(note_number - p1) / p3)) * p4 + p2

        for n in range(length_of_s):
            vco[n] += jitter[n] * jitter_depth

        if np.max(vco) < 20000:
            p = np.zeros(length_of_s)
            x = 0
            for n in range(length_of_s):
                p[n] = np.sin(2 * np.pi * x)
                delta = vco[n] / fs
                x += delta
                if x >= 1:
                    x -= 1

            vca0 = ADSR(fs, VCA0_A[i], VCA0_D[i], VCA0_S[i], VCA0_R[i], VCA0_gate[i], VCA0_duration[i])
            for n in range(length_of_s):
                vca0[n] = VCA0_offset[i] + vca0[n] * VCA0_depth[i]

            vca1 = cosine_envelope(fs, VCA1_delay[i], VCA1_A[i], VCA1_S[i], VCA1_R[i], VCA1_gate[i], VCA1_duration[i])
            for n in range(length_of_s):
                vca1[n] = VCA1_offset[i] + vca1[n] * VCA1_depth[i]

            w = np.zeros(length_of_s)
            for n in range(length_of_s):
                w[n] = np.random.rand() * 2 - 1

            fc = 40
            Q = 1 / np.sqrt(2)
            a, b = LPF(fs, fc, Q)
            shimmer = filter(a, b, w)
            shimmer /= np.max(np.abs(shimmer))

            p1 = 1
            p2 = -0.3
            p3 = 100 / 12
            p4 = 0.8
            shimmer_depth = 1 / (1 + np.exp(-(VCO_offset[i] / f0 - p1) / p3)) * p4 + p2

            for n in range(length_of_s):
                vca1[n] *= 1 + shimmer[n] * shimmer_depth
                if vca1[n] < 0:
                    vca1[n] = 0

            # A part
            for n in range(length_of_s):
                sa[n] += p[n] * vca0[n]

            # B part
            for n in range(length_of_s):
                sb[n] += p[n] * vca1[n]

    s0 = sa * 0.3 + sb * 0.7

    fc = 1500
    Q = 1 / np.sqrt(2)
    a, b = BPF(fs, fc, Q)
    s1 = filter(a, b, s0)

    fc = 3000
    Q = 1 / np.sqrt(2)
    a, b = LPF(fs, fc, Q)
    s2 = filter(a, b, s1)

    s2 *= velocity / 127 / np.max(np.abs(s2))

    return s2

def trombone(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    sa = np.zeros(length_of_s)
    sb = np.zeros(length_of_s)
    s0 = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    number_of_partial= int(3600 / f0)
    if number_of_partial < 5:
        number_of_partial = 5
    elif number_of_partial > 30:
        number_of_partial = 30

    VCO_A = np.repeat(0, number_of_partial)
    VCO_D = np.repeat(0.15, number_of_partial)
    VCO_S = np.repeat(0, number_of_partial)
    VCO_R = np.repeat(0.15, number_of_partial)
    VCO_gate = np.repeat(duration, number_of_partial)
    VCO_duration = np.repeat(duration, number_of_partial)
    VCO_offset = np.arange(1, number_of_partial + 1) * f0
    VCO_depth = np.arange(1, number_of_partial + 1) * (-f0 * 0.15)

    VCA0_A = np.repeat(0.01, number_of_partial)
    VCA0_D = np.repeat(0.02, number_of_partial)
    VCA0_S = np.repeat(0, number_of_partial)
    VCA0_R = np.repeat(0.02, number_of_partial)
    VCA0_gate = np.repeat(gate, number_of_partial)
    VCA0_duration = np.repeat(duration, number_of_partial)
    VCA0_offset = np.repeat(0, number_of_partial)
    VCA0_depth = np.repeat(1, number_of_partial)

    p1 = 1800
    p2 = 0
    p3 = 1800 / 12
    p4 = 0.03
    VCA1_delay = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA1_delay[i] = 1 / (1 + np.exp(-((VCO_offset[i] + VCO_depth[i]) - p1) / p3)) * p4 + p2

    for i in range(1, number_of_partial):
        VCA1_delay[i] = VCA1_delay[i] - VCA1_delay[0]

    VCA1_delay[0] = 0

    p1 = 1800
    p2 = 0.02
    p3 = 1800 / 12
    p4 = 0.02
    VCA1_A = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA1_A[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA1_S = np.repeat(1, number_of_partial)

    p1 = 1800
    p2 = 0.2
    p3 = 1800 / 12
    p4 = 0.1
    VCA1_R = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA1_R[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA1_gate = np.repeat(gate, number_of_partial)
    VCA1_duration = np.repeat(duration, number_of_partial)
    VCA1_offset = np.repeat(0, number_of_partial)
    VCA1_depth = np.repeat(1, number_of_partial)

    np.random.seed(0)

    for i in range(number_of_partial):
        vco = ADSR(fs, VCO_A[i], VCO_D[i], VCO_S[i], VCO_R[i], VCO_gate[i], VCO_duration[i])
        for n in range(length_of_s):
            vco[n] = VCO_offset[i] + vco[n] * VCO_depth[i]

        w = np.zeros(length_of_s)
        for n in range(length_of_s):
            w[n] = np.random.rand() * 2 - 1

        fc = 40
        Q = 1 / np.sqrt(2)
        a, b = LPF(fs, fc, Q)
        jitter = filter(a, b, w)
        jitter /= np.max(np.abs(jitter))

        p1 = 108
        p2 = 1
        p3 = 150 / 12
        p4 = 20
        jitter_depth = 1 / (1 + np.exp(-(note_number - p1) / p3)) * p4 + p2

        for n in range(length_of_s):
            vco[n] += jitter[n] * jitter_depth

        if np.max(vco) < 20000:
            p = np.zeros(length_of_s)
            x = 0
            for n in range(length_of_s):
                p[n] = np.sin(2 * np.pi * x)
                delta = vco[n] / fs
                x += delta
                if x >= 1:
                    x -= 1

            vca0 = ADSR(fs, VCA0_A[i], VCA0_D[i], VCA0_S[i], VCA0_R[i], VCA0_gate[i], VCA0_duration[i])
            for n in range(length_of_s):
                vca0[n] = VCA0_offset[i] + vca0[n] * VCA0_depth[i]

            vca1 = cosine_envelope(fs, VCA1_delay[i], VCA1_A[i], VCA1_S[i], VCA1_R[i], VCA1_gate[i], VCA1_duration[i])
            for n in range(length_of_s):
                vca1[n] = VCA1_offset[i] + vca1[n] * VCA1_depth[i]

            w = np.zeros(length_of_s)
            for n in range(length_of_s):
                w[n] = np.random.rand() * 2 - 1

            fc = 40
            Q = 1 / np.sqrt(2)
            a, b = LPF(fs, fc, Q)
            shimmer = filter(a, b, w)
            shimmer /= np.max(np.abs(shimmer))

            p1 = 1
            p2 = -0.3
            p3 = 100 / 12
            p4 = 0.8
            shimmer_depth = 1 / (1 + np.exp(-(VCO_offset[i] / f0 - p1) / p3)) * p4 + p2

            for n in range(length_of_s):
                vca1[n] *= 1 + shimmer[n] * shimmer_depth
                if vca1[n] < 0:
                    vca1[n] = 0

            # A part
            for n in range(length_of_s):
                sa[n] += p[n] * vca0[n]

            # B part
            for n in range(length_of_s):
                sb[n] += p[n] * vca1[n]

    s0 = sa * 0.3 + sb * 0.7

    fc = 900
    Q = 1 / np.sqrt(2)
    a, b = BPF(fs, fc, Q)
    s1 = filter(a, b, s0)

    fc = 1800
    Q = 1 / np.sqrt(2)
    a, b = LPF(fs, fc, Q)
    s2 = filter(a, b, s1)

    s2 *= velocity / 127 / np.max(np.abs(s2))

    return s2

def horn(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    sa = np.zeros(length_of_s)
    sb = np.zeros(length_of_s)
    s0 = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    number_of_partial= int(2400 / f0)
    if number_of_partial < 5:
        number_of_partial = 5
    elif number_of_partial > 30:
        number_of_partial = 30

    VCO_A = np.repeat(0, number_of_partial)
    VCO_D = np.repeat(0.15, number_of_partial)
    VCO_S = np.repeat(0, number_of_partial)
    VCO_R = np.repeat(0.15, number_of_partial)
    VCO_gate = np.repeat(duration, number_of_partial)
    VCO_duration = np.repeat(duration, number_of_partial)
    VCO_offset = np.arange(1, number_of_partial + 1) * f0
    VCO_depth = np.arange(1, number_of_partial + 1) * (-f0 * 0.15)

    VCA0_A = np.repeat(0.01, number_of_partial)
    VCA0_D = np.repeat(0.02, number_of_partial)
    VCA0_S = np.repeat(0, number_of_partial)
    VCA0_R = np.repeat(0.02, number_of_partial)
    VCA0_gate = np.repeat(gate, number_of_partial)
    VCA0_duration = np.repeat(duration, number_of_partial)
    VCA0_offset = np.repeat(0, number_of_partial)
    VCA0_depth = np.repeat(1, number_of_partial)

    p1 = 1200
    p2 = 0
    p3 = 1200 / 12
    p4 = 0.03
    VCA1_delay = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA1_delay[i] = 1 / (1 + np.exp(-((VCO_offset[i] + VCO_depth[i]) - p1) / p3)) * p4 + p2

    for i in range(1, number_of_partial):
        VCA1_delay[i] = VCA1_delay[i] - VCA1_delay[0]

    VCA1_delay[0] = 0

    p1 = 1200
    p2 = 0.02
    p3 = 1200 / 12
    p4 = 0.02
    VCA1_A = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA1_A[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA1_S = np.repeat(1, number_of_partial)

    p1 = 1200
    p2 = 0.2
    p3 = 1200 / 12
    p4 = 0.1
    VCA1_R = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA1_R[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA1_gate = np.repeat(gate, number_of_partial)
    VCA1_duration = np.repeat(duration, number_of_partial)
    VCA1_offset = np.repeat(0, number_of_partial)
    VCA1_depth = np.repeat(1, number_of_partial)

    np.random.seed(0)

    for i in range(number_of_partial):
        vco = ADSR(fs, VCO_A[i], VCO_D[i], VCO_S[i], VCO_R[i], VCO_gate[i], VCO_duration[i])
        for n in range(length_of_s):
            vco[n] = VCO_offset[i] + vco[n] * VCO_depth[i]

        w = np.zeros(length_of_s)
        for n in range(length_of_s):
            w[n] = np.random.rand() * 2 - 1

        fc = 40
        Q = 1 / np.sqrt(2)
        a, b = LPF(fs, fc, Q)
        jitter = filter(a, b, w)
        jitter /= np.max(np.abs(jitter))

        p1 = 108
        p2 = 1
        p3 = 150 / 12
        p4 = 20
        jitter_depth = 1 / (1 + np.exp(-(note_number - p1) / p3)) * p4 + p2

        for n in range(length_of_s):
            vco[n] += jitter[n] * jitter_depth

        if np.max(vco) < 20000:
            p = np.zeros(length_of_s)
            x = 0
            for n in range(length_of_s):
                p[n] = np.sin(2 * np.pi * x)
                delta = vco[n] / fs
                x += delta
                if x >= 1:
                    x -= 1

            vca0 = ADSR(fs, VCA0_A[i], VCA0_D[i], VCA0_S[i], VCA0_R[i], VCA0_gate[i], VCA0_duration[i])
            for n in range(length_of_s):
                vca0[n] = VCA0_offset[i] + vca0[n] * VCA0_depth[i]

            vca1 = cosine_envelope(fs, VCA1_delay[i], VCA1_A[i], VCA1_S[i], VCA1_R[i], VCA1_gate[i], VCA1_duration[i])
            for n in range(length_of_s):
                vca1[n] = VCA1_offset[i] + vca1[n] * VCA1_depth[i]

            w = np.zeros(length_of_s)
            for n in range(length_of_s):
                w[n] = np.random.rand() * 2 - 1

            fc = 40
            Q = 1 / np.sqrt(2)
            a, b = LPF(fs, fc, Q)
            shimmer = filter(a, b, w)
            shimmer /= np.max(np.abs(shimmer))

            p1 = 1
            p2 = -0.3
            p3 = 100 / 12
            p4 = 0.8
            shimmer_depth = 1 / (1 + np.exp(-(VCO_offset[i] / f0 - p1) / p3)) * p4 + p2

            for n in range(length_of_s):
                vca1[n] *= 1 + shimmer[n] * shimmer_depth
                if vca1[n] < 0:
                    vca1[n] = 0

            # A part
            for n in range(length_of_s):
                sa[n] += p[n] * vca0[n]

            # B part
            for n in range(length_of_s):
                sb[n] += p[n] * vca1[n]

    s0 = sa * 0.3 + sb * 0.7

    fc = 600
    Q = 1 / np.sqrt(2)
    a, b = BPF(fs, fc, Q)
    s1 = filter(a, b, s0)

    fc = 1200
    Q = 1 / np.sqrt(2)
    a, b = LPF(fs, fc, Q)
    s2 = filter(a, b, s1)

    s2 *= velocity / 127 / np.max(np.abs(s2))

    return s2

def tuba(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    sa = np.zeros(length_of_s)
    sb = np.zeros(length_of_s)
    s0 = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    number_of_partial= int(1600 / f0)
    if number_of_partial < 5:
        number_of_partial = 5
    elif number_of_partial > 30:
        number_of_partial = 30

    VCO_A = np.repeat(0, number_of_partial)
    VCO_D = np.repeat(0.15, number_of_partial)
    VCO_S = np.repeat(0, number_of_partial)
    VCO_R = np.repeat(0.15, number_of_partial)
    VCO_gate = np.repeat(duration, number_of_partial)
    VCO_duration = np.repeat(duration, number_of_partial)
    VCO_offset = np.arange(1, number_of_partial + 1) * f0
    VCO_depth = np.arange(1, number_of_partial + 1) * (-f0 * 0.15)

    VCA0_A = np.repeat(0.01, number_of_partial)
    VCA0_D = np.repeat(0.02, number_of_partial)
    VCA0_S = np.repeat(0, number_of_partial)
    VCA0_R = np.repeat(0.02, number_of_partial)
    VCA0_gate = np.repeat(gate, number_of_partial)
    VCA0_duration = np.repeat(duration, number_of_partial)
    VCA0_offset = np.repeat(0, number_of_partial)
    VCA0_depth = np.repeat(1, number_of_partial)

    p1 = 800
    p2 = 0
    p3 = 800 / 12
    p4 = 0.03
    VCA1_delay = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA1_delay[i] = 1 / (1 + np.exp(-((VCO_offset[i] + VCO_depth[i]) - p1) / p3)) * p4 + p2

    for i in range(1, number_of_partial):
        VCA1_delay[i] = VCA1_delay[i] - VCA1_delay[0]

    VCA1_delay[0] = 0

    p1 = 800
    p2 = 0.02
    p3 = 800 / 12
    p4 = 0.02
    VCA1_A = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA1_A[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA1_S = np.repeat(1, number_of_partial)

    p1 = 800
    p2 = 0.2
    p3 = 800 / 12
    p4 = 0.1
    VCA1_R = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA1_R[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA1_gate = np.repeat(gate, number_of_partial)
    VCA1_duration = np.repeat(duration, number_of_partial)
    VCA1_offset = np.repeat(0, number_of_partial)
    VCA1_depth = np.repeat(1, number_of_partial)

    np.random.seed(0)

    for i in range(number_of_partial):
        vco = ADSR(fs, VCO_A[i], VCO_D[i], VCO_S[i], VCO_R[i], VCO_gate[i], VCO_duration[i])
        for n in range(length_of_s):
            vco[n] = VCO_offset[i] + vco[n] * VCO_depth[i]

        w = np.zeros(length_of_s)
        for n in range(length_of_s):
            w[n] = np.random.rand() * 2 - 1

        fc = 40
        Q = 1 / np.sqrt(2)
        a, b = LPF(fs, fc, Q)
        jitter = filter(a, b, w)
        jitter /= np.max(np.abs(jitter))

        p1 = 108
        p2 = 1
        p3 = 150 / 12
        p4 = 20
        jitter_depth = 1 / (1 + np.exp(-(note_number - p1) / p3)) * p4 + p2

        for n in range(length_of_s):
            vco[n] += jitter[n] * jitter_depth

        if np.max(vco) < 20000:
            p = np.zeros(length_of_s)
            x = 0
            for n in range(length_of_s):
                p[n] = np.sin(2 * np.pi * x)
                delta = vco[n] / fs
                x += delta
                if x >= 1:
                    x -= 1

            vca0 = ADSR(fs, VCA0_A[i], VCA0_D[i], VCA0_S[i], VCA0_R[i], VCA0_gate[i], VCA0_duration[i])
            for n in range(length_of_s):
                vca0[n] = VCA0_offset[i] + vca0[n] * VCA0_depth[i]

            vca1 = cosine_envelope(fs, VCA1_delay[i], VCA1_A[i], VCA1_S[i], VCA1_R[i], VCA1_gate[i], VCA1_duration[i])
            for n in range(length_of_s):
                vca1[n] = VCA1_offset[i] + vca1[n] * VCA1_depth[i]

            w = np.zeros(length_of_s)
            for n in range(length_of_s):
                w[n] = np.random.rand() * 2 - 1

            fc = 40
            Q = 1 / np.sqrt(2)
            a, b = LPF(fs, fc, Q)
            shimmer = filter(a, b, w)
            shimmer /= np.max(np.abs(shimmer))

            p1 = 1
            p2 = -0.3
            p3 = 100 / 12
            p4 = 0.8
            shimmer_depth = 1 / (1 + np.exp(-(VCO_offset[i] / f0 - p1) / p3)) * p4 + p2

            for n in range(length_of_s):
                vca1[n] *= 1 + shimmer[n] * shimmer_depth
                if vca1[n] < 0:
                    vca1[n] = 0

            # A part
            for n in range(length_of_s):
                sa[n] += p[n] * vca0[n]

            # B part
            for n in range(length_of_s):
                sb[n] += p[n] * vca1[n]

    s0 = sa * 0.3 + sb * 0.7

    fc = 400
    Q = 1 / np.sqrt(2)
    a, b = BPF(fs, fc, Q)
    s1 = filter(a, b, s0)

    fc = 800
    Q = 1 / np.sqrt(2)
    a, b = LPF(fs, fc, Q)
    s2 = filter(a, b, s1)

    s2 *= velocity / 127 / np.max(np.abs(s2))

    return s2

def violin(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    s0 = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    number_of_partial= int(6000 / f0)
    if number_of_partial < 5:
        number_of_partial = 5
    elif number_of_partial > 30:
        number_of_partial = 30

    np.random.seed(0)
    for n in range(length_of_s):
        s0[n] = np.random.rand() * 2 - 1

    fc = 2000
    Q = 1 / np.sqrt(2)
    a, b = LPF(fs, fc, Q)
    s1 = filter(a, b, s0)
    s2 = filter(a, b, s1)

    fc = 250
    Q = 1 / np.sqrt(2)
    a, b = HPF(fs, fc, Q)
    s3 = filter(a, b, s2)
    s4 = filter(a, b, s3)

    p1 = 69
    p2 = 0.02
    p3 = 70 / 12
    p4 = 0.18
    ratio = (1 - 1 / (1 + np.exp(-(note_number - p1) / p3))) * p4 + p2

    vibrato_depth = 440 * np.power(2, (note_number + ratio - 69) / 12) - f0
    vibrato_rate = 8
    vibrato = np.zeros(length_of_s)
    for n in range(length_of_s):
        vibrato[n] = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * n / fs)

    w = np.zeros(length_of_s)
    for n in range(length_of_s):
        w[n] = np.random.rand() * 2 - 1

    fc = 8
    Q = 1 / np.sqrt(2)
    a, b = LPF(fs, fc, Q)
    f0_fluctuation = filter(a, b, w)
    f0_fluctuation /= np.max(np.abs(f0_fluctuation))

    f0_fluctuation_depth = vibrato_depth * 0.5

    s7 = np.zeros(length_of_s)
    for i in range(number_of_partial):
        vcf = np.zeros(length_of_s)
        for n in range(length_of_s):
            vcf[n] = (f0 + vibrato[n] + f0_fluctuation[n] * f0_fluctuation_depth) * (i + 1)

        if np.max(vcf) < 20000:
            Q = 200
            s5 = np.zeros(length_of_s)
            for n in range(length_of_s):
                a, b = BPF(fs, vcf[n], Q)
                for m in range(0, 3):
                    if n - m >= 0:
                        s5[n] += b[m] * s4[n - m]

                for m in range(1, 3):
                    if n - m >= 0:
                        s5[n] += -a[m] * s5[n - m]

            s6 = np.zeros(length_of_s)
            for n in range(length_of_s):
                a, b = BPF(fs, vcf[n], Q)
                for m in range(0, 3):
                    if n - m >= 0:
                        s6[n] += b[m] * s5[n - m]

                for m in range(1, 3):
                    if n - m >= 0:
                        s6[n] += -a[m] * s6[n - m]

            s7 += s6

    VCA_A = np.array([0.2])
    VCA_D = np.array([0])
    VCA_S = np.array([1])
    VCA_R = np.array([0.4])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        s7[n] *= vca[n]

    s7 *= velocity / 127 / np.max(np.abs(s7))

    return s7

def viola(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    s0 = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    number_of_partial= int(6000 / f0)
    if number_of_partial < 5:
        number_of_partial = 5
    elif number_of_partial > 30:
        number_of_partial = 30

    np.random.seed(0)
    for n in range(length_of_s):
        s0[n] = np.random.rand() * 2 - 1

    fc = 1400
    Q = 1 / np.sqrt(2)
    a, b = LPF(fs, fc, Q)
    s1 = filter(a, b, s0)
    s2 = filter(a, b, s1)

    fc = 200
    Q = 1 / np.sqrt(2)
    a, b = HPF(fs, fc, Q)
    s3 = filter(a, b, s2)
    s4 = filter(a, b, s3)

    p1 = 69
    p2 = 0.02
    p3 = 70 / 12
    p4 = 0.18
    ratio = (1 - 1 / (1 + np.exp(-(note_number - p1) / p3))) * p4 + p2

    vibrato_depth = 440 * np.power(2, (note_number + ratio - 69) / 12) - f0
    vibrato_rate = 8
    vibrato = np.zeros(length_of_s)
    for n in range(length_of_s):
        vibrato[n] = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * n / fs)

    w = np.zeros(length_of_s)
    for n in range(length_of_s):
        w[n] = np.random.rand() * 2 - 1

    fc = 8
    Q = 1 / np.sqrt(2)
    a, b = LPF(fs, fc, Q)
    f0_fluctuation = filter(a, b, w)
    f0_fluctuation /= np.max(np.abs(f0_fluctuation))

    f0_fluctuation_depth = vibrato_depth * 0.5

    s7 = np.zeros(length_of_s)
    for i in range(number_of_partial):
        vcf = np.zeros(length_of_s)
        for n in range(length_of_s):
            vcf[n] = (f0 + vibrato[n] + f0_fluctuation[n] * f0_fluctuation_depth) * (i + 1)

        if np.max(vcf) < 20000:
            Q = 200
            s5 = np.zeros(length_of_s)
            for n in range(length_of_s):
                a, b = BPF(fs, vcf[n], Q)
                for m in range(0, 3):
                    if n - m >= 0:
                        s5[n] += b[m] * s4[n - m]

                for m in range(1, 3):
                    if n - m >= 0:
                        s5[n] += -a[m] * s5[n - m]

            s6 = np.zeros(length_of_s)
            for n in range(length_of_s):
                a, b = BPF(fs, vcf[n], Q)
                for m in range(0, 3):
                    if n - m >= 0:
                        s6[n] += b[m] * s5[n - m]

                for m in range(1, 3):
                    if n - m >= 0:
                        s6[n] += -a[m] * s6[n - m]

            s7 += s6

    VCA_A = np.array([0.2])
    VCA_D = np.array([0])
    VCA_S = np.array([1])
    VCA_R = np.array([0.4])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])
    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        s7[n] *= vca[n]

    s7 *= velocity / 127 / np.max(np.abs(s7))

    return s7

def cello(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    s0 = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    number_of_partial= int(6000 / f0)
    if number_of_partial < 5:
        number_of_partial = 5
    elif number_of_partial > 30:
        number_of_partial = 30

    np.random.seed(0)
    for n in range(length_of_s):
        s0[n] = np.random.rand() * 2 - 1

    fc = 900
    Q = 1 / np.sqrt(2)
    a, b = LPF(fs, fc, Q)
    s1 = filter(a, b, s0)
    s2 = filter(a, b, s1)

    fc = 150
    Q = 1 / np.sqrt(2)
    a, b = HPF(fs, fc, Q)
    s3 = filter(a, b, s2)
    s4 = filter(a, b, s3)

    p1 = 69
    p2 = 0.02
    p3 = 70 / 12
    p4 = 0.18
    ratio = (1 - 1 / (1 + np.exp(-(note_number - p1) / p3))) * p4 + p2

    vibrato_depth = 440 * np.power(2, (note_number + ratio - 69) / 12) - f0
    vibrato_rate = 8
    vibrato = np.zeros(length_of_s)
    for n in range(length_of_s):
        vibrato[n] = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * n / fs)

    w = np.zeros(length_of_s)
    for n in range(length_of_s):
        w[n] = np.random.rand() * 2 - 1

    fc = 8
    Q = 1 / np.sqrt(2)
    a, b = LPF(fs, fc, Q)
    f0_fluctuation = filter(a, b, w)
    f0_fluctuation /= np.max(np.abs(f0_fluctuation))

    f0_fluctuation_depth = vibrato_depth * 0.5

    s7 = np.zeros(length_of_s)
    for i in range(number_of_partial):
        vcf = np.zeros(length_of_s)
        for n in range(length_of_s):
            vcf[n] = (f0 + vibrato[n] + f0_fluctuation[n] * f0_fluctuation_depth) * (i + 1)

        if np.max(vcf) < 20000:
            Q = 200
            s5 = np.zeros(length_of_s)
            for n in range(length_of_s):
                a, b = BPF(fs, vcf[n], Q)
                for m in range(0, 3):
                    if n - m >= 0:
                        s5[n] += b[m] * s4[n - m]

                for m in range(1, 3):
                    if n - m >= 0:
                        s5[n] += -a[m] * s5[n - m]

            s6 = np.zeros(length_of_s)
            for n in range(length_of_s):
                a, b = BPF(fs, vcf[n], Q)
                for m in range(0, 3):
                    if n - m >= 0:
                        s6[n] += b[m] * s5[n - m]

                for m in range(1, 3):
                    if n - m >= 0:
                        s6[n] += -a[m] * s6[n - m]

            s7 += s6

    VCA_A = np.array([0.2])
    VCA_D = np.array([0])
    VCA_S = np.array([1])
    VCA_R = np.array([0.4])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])
    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        s7[n] *= vca[n]

    s7 *= velocity / 127 / np.max(np.abs(s7))

    return s7

def contrabass(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    s0 = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    number_of_partial= int(6000 / f0)
    if number_of_partial < 5:
        number_of_partial = 5
    elif number_of_partial > 30:
        number_of_partial = 30

    np.random.seed(0)
    for n in range(length_of_s):
        s0[n] = np.random.rand() * 2 - 1

    fc = 500
    Q = 1 / np.sqrt(2)
    a, b = LPF(fs, fc, Q)
    s1 = filter(a, b, s0)
    s2 = filter(a, b, s1)

    fc = 100
    Q = 1 / np.sqrt(2)
    a, b = HPF(fs, fc, Q)
    s3 = filter(a, b, s2)
    s4 = filter(a, b, s3)

    p1 = 69
    p2 = 0.02
    p3 = 70 / 12
    p4 = 0.18
    ratio = (1 - 1 / (1 + np.exp(-(note_number - p1) / p3))) * p4 + p2

    vibrato_depth = 440 * np.power(2, (note_number + ratio - 69) / 12) - f0
    vibrato_rate = 8
    vibrato = np.zeros(length_of_s)
    for n in range(length_of_s):
        vibrato[n] = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * n / fs)

    w = np.zeros(length_of_s)
    for n in range(length_of_s):
        w[n] = np.random.rand() * 2 - 1

    fc = 8
    Q = 1 / np.sqrt(2)
    a, b = LPF(fs, fc, Q)
    f0_fluctuation = filter(a, b, w)
    f0_fluctuation /= np.max(np.abs(f0_fluctuation))

    f0_fluctuation_depth = vibrato_depth * 0.5

    s7 = np.zeros(length_of_s)
    for i in range(number_of_partial):
        vcf = np.zeros(length_of_s)
        for n in range(length_of_s):
            vcf[n] = (f0 + vibrato[n] + f0_fluctuation[n] * f0_fluctuation_depth) * (i + 1)

        if np.max(vcf) < 20000:
            Q = 200
            s5 = np.zeros(length_of_s)
            for n in range(length_of_s):
                a, b = BPF(fs, vcf[n], Q)
                for m in range(0, 3):
                    if n - m >= 0:
                        s5[n] += b[m] * s4[n - m]

                for m in range(1, 3):
                    if n - m >= 0:
                        s5[n] += -a[m] * s5[n - m]

            s6 = np.zeros(length_of_s)
            for n in range(length_of_s):
                a, b = BPF(fs, vcf[n], Q)
                for m in range(0, 3):
                    if n - m >= 0:
                        s6[n] += b[m] * s5[n - m]

                for m in range(1, 3):
                    if n - m >= 0:
                        s6[n] += -a[m] * s6[n - m]

            s7 += s6

    VCA_A = np.array([0.2])
    VCA_D = np.array([0])
    VCA_S = np.array([1])
    VCA_R = np.array([0.4])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])
    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        s7[n] *= vca[n]

    s7 *= velocity / 127 / np.max(np.abs(s7))

    return s7

def harp(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    s0 = np.zeros(length_of_s)

    f = np.array([0, 4, 7, 12, 16, 20, 25, 30, 35, 41, 47, 53, 60, 67, 74, 82, 90, 99, 108, 118, 128, 139, 150, 162, 175, 188, 202, 217, 233, 250, 267, 286, 306, 326, 348, 371, 396, 421, 449, 477, 508, 540, 574, 609, 647, 687, 729, 773, 820, 869, 922, 977, 1035, 1097, 1161, 1230, 1302, 1379, 1460, 1545, 1635, 1730, 1830, 1936, 2048])
    a = np.array([[0.028926, 0.053858, 0.277871, 0.086249, 1.000000, 0.711683, 0.318528, 0.207507, 0.262590, 0.451641, 0.262717, 0.114672, 0.506146, 0.176017, 0.104403, 0.025953, 0.027905, 0.039973, 0.013096, 0.057863, 0.073509, 0.012798, 0.015880, 0.019534, 0.006723, 0.014531, 0.011033, 0.008811, 0.008016, 0.015216, 0.006268, 0.002561, 0.000664, 0.002042, 0.003062, 0.001679, 0.001879, 0.000752, 0.001087, 0.001099, 0.001105, 0.001938, 0.001332, 0.000569, 0.000729, 0.000481, 0.000526, 0.000524, 0.000539, 0.000408, 0.000292, 0.000366, 0.000366, 0.000340, 0.000314, 0.000292, 0.000297, 0.000294, 0.000283, 0.000279, 0.000241, 0.000281, 0.000236, 0.000254],
                  [0.028926, 0.053858, 0.277871, 0.086249, 1.000000, 0.711683, 0.318528, 0.207507, 0.262590, 0.451641, 0.262717, 0.114672, 0.506146, 0.176017, 0.104403, 0.025953, 0.027905, 0.039973, 0.013096, 0.057863, 0.073509, 0.012798, 0.015880, 0.019534, 0.006723, 0.014531, 0.011033, 0.008811, 0.008016, 0.015216, 0.006268, 0.002561, 0.000664, 0.002042, 0.003062, 0.001679, 0.001879, 0.000752, 0.001087, 0.001099, 0.001105, 0.001938, 0.001332, 0.000569, 0.000729, 0.000481, 0.000526, 0.000524, 0.000539, 0.000408, 0.000292, 0.000366, 0.000366, 0.000340, 0.000314, 0.000292, 0.000297, 0.000294, 0.000283, 0.000279, 0.000241, 0.000281, 0.000236, 0.000254],
                  [0.028926, 0.053858, 0.277871, 0.086249, 1.000000, 0.711683, 0.318528, 0.207507, 0.262590, 0.451641, 0.262717, 0.114672, 0.506146, 0.176017, 0.104403, 0.025953, 0.027905, 0.039973, 0.013096, 0.057863, 0.073509, 0.012798, 0.015880, 0.019534, 0.006723, 0.014531, 0.011033, 0.008811, 0.008016, 0.015216, 0.006268, 0.002561, 0.000664, 0.002042, 0.003062, 0.001679, 0.001879, 0.000752, 0.001087, 0.001099, 0.001105, 0.001938, 0.001332, 0.000569, 0.000729, 0.000481, 0.000526, 0.000524, 0.000539, 0.000408, 0.000292, 0.000366, 0.000366, 0.000340, 0.000314, 0.000292, 0.000297, 0.000294, 0.000283, 0.000279, 0.000241, 0.000281, 0.000236, 0.000254],
                  [0.028926, 0.053858, 0.277871, 0.086249, 1.000000, 0.711683, 0.318528, 0.207507, 0.262590, 0.451641, 0.262717, 0.114672, 0.506146, 0.176017, 0.104403, 0.025953, 0.027905, 0.039973, 0.013096, 0.057863, 0.073509, 0.012798, 0.015880, 0.019534, 0.006723, 0.014531, 0.011033, 0.008811, 0.008016, 0.015216, 0.006268, 0.002561, 0.000664, 0.002042, 0.003062, 0.001679, 0.001879, 0.000752, 0.001087, 0.001099, 0.001105, 0.001938, 0.001332, 0.000569, 0.000729, 0.000481, 0.000526, 0.000524, 0.000539, 0.000408, 0.000292, 0.000366, 0.000366, 0.000340, 0.000314, 0.000292, 0.000297, 0.000294, 0.000283, 0.000279, 0.000241, 0.000281, 0.000236, 0.000254],
                  [0.028926, 0.053858, 0.277871, 0.086249, 1.000000, 0.711683, 0.318528, 0.207507, 0.262590, 0.451641, 0.262717, 0.114672, 0.506146, 0.176017, 0.104403, 0.025953, 0.027905, 0.039973, 0.013096, 0.057863, 0.073509, 0.012798, 0.015880, 0.019534, 0.006723, 0.014531, 0.011033, 0.008811, 0.008016, 0.015216, 0.006268, 0.002561, 0.000664, 0.002042, 0.003062, 0.001679, 0.001879, 0.000752, 0.001087, 0.001099, 0.001105, 0.001938, 0.001332, 0.000569, 0.000729, 0.000481, 0.000526, 0.000524, 0.000539, 0.000408, 0.000292, 0.000366, 0.000366, 0.000340, 0.000314, 0.000292, 0.000297, 0.000294, 0.000283, 0.000279, 0.000241, 0.000281, 0.000236, 0.000254],
                  [0.028926, 0.053858, 0.277871, 0.086249, 1.000000, 0.711683, 0.318528, 0.207507, 0.262590, 0.451641, 0.262717, 0.114672, 0.506146, 0.176017, 0.104403, 0.025953, 0.027905, 0.039973, 0.013096, 0.057863, 0.073509, 0.012798, 0.015880, 0.019534, 0.006723, 0.014531, 0.011033, 0.008811, 0.008016, 0.015216, 0.006268, 0.002561, 0.000664, 0.002042, 0.003062, 0.001679, 0.001879, 0.000752, 0.001087, 0.001099, 0.001105, 0.001938, 0.001332, 0.000569, 0.000729, 0.000481, 0.000526, 0.000524, 0.000539, 0.000408, 0.000292, 0.000366, 0.000366, 0.000340, 0.000314, 0.000292, 0.000297, 0.000294, 0.000283, 0.000279, 0.000241, 0.000281, 0.000236, 0.000254],
                  [0.028926, 0.053858, 0.277871, 0.086249, 1.000000, 0.711683, 0.318528, 0.207507, 0.262590, 0.451641, 0.262717, 0.114672, 0.506146, 0.176017, 0.104403, 0.025953, 0.027905, 0.039973, 0.013096, 0.057863, 0.073509, 0.012798, 0.015880, 0.019534, 0.006723, 0.014531, 0.011033, 0.008811, 0.008016, 0.015216, 0.006268, 0.002561, 0.000664, 0.002042, 0.003062, 0.001679, 0.001879, 0.000752, 0.001087, 0.001099, 0.001105, 0.001938, 0.001332, 0.000569, 0.000729, 0.000481, 0.000526, 0.000524, 0.000539, 0.000408, 0.000292, 0.000366, 0.000366, 0.000340, 0.000314, 0.000292, 0.000297, 0.000294, 0.000283, 0.000279, 0.000241, 0.000281, 0.000236, 0.000254],
                  [0.028926, 0.053858, 0.277871, 0.086249, 1.000000, 0.711683, 0.318528, 0.207507, 0.262590, 0.451641, 0.262717, 0.114672, 0.506146, 0.176017, 0.104403, 0.025953, 0.027905, 0.039973, 0.013096, 0.057863, 0.073509, 0.012798, 0.015880, 0.019534, 0.006723, 0.014531, 0.011033, 0.008811, 0.008016, 0.015216, 0.006268, 0.002561, 0.000664, 0.002042, 0.003062, 0.001679, 0.001879, 0.000752, 0.001087, 0.001099, 0.001105, 0.001938, 0.001332, 0.000569, 0.000729, 0.000481, 0.000526, 0.000524, 0.000539, 0.000408, 0.000292, 0.000366, 0.000366, 0.000340, 0.000314, 0.000292, 0.000297, 0.000294, 0.000283, 0.000279, 0.000241, 0.000281, 0.000236, 0.000254],
                  [0.028926, 0.053858, 0.277871, 0.086249, 1.000000, 0.711683, 0.318528, 0.207507, 0.262590, 0.451641, 0.262717, 0.114672, 0.506146, 0.176017, 0.104403, 0.025953, 0.027905, 0.039973, 0.013096, 0.057863, 0.073509, 0.012798, 0.015880, 0.019534, 0.006723, 0.014531, 0.011033, 0.008811, 0.008016, 0.015216, 0.006268, 0.002561, 0.000664, 0.002042, 0.003062, 0.001679, 0.001879, 0.000752, 0.001087, 0.001099, 0.001105, 0.001938, 0.001332, 0.000569, 0.000729, 0.000481, 0.000526, 0.000524, 0.000539, 0.000408, 0.000292, 0.000366, 0.000366, 0.000340, 0.000314, 0.000292, 0.000297, 0.000294, 0.000283, 0.000279, 0.000241, 0.000281, 0.000236, 0.000254],
                  [0.028926, 0.053858, 0.277871, 0.086249, 1.000000, 0.711683, 0.318528, 0.207507, 0.262590, 0.451641, 0.262717, 0.114672, 0.506146, 0.176017, 0.104403, 0.025953, 0.027905, 0.039973, 0.013096, 0.057863, 0.073509, 0.012798, 0.015880, 0.019534, 0.006723, 0.014531, 0.011033, 0.008811, 0.008016, 0.015216, 0.006268, 0.002561, 0.000664, 0.002042, 0.003062, 0.001679, 0.001879, 0.000752, 0.001087, 0.001099, 0.001105, 0.001938, 0.001332, 0.000569, 0.000729, 0.000481, 0.000526, 0.000524, 0.000539, 0.000408, 0.000292, 0.000366, 0.000366, 0.000340, 0.000314, 0.000292, 0.000297, 0.000294, 0.000283, 0.000279, 0.000241, 0.000281, 0.000236, 0.000254],
                  [0.028926, 0.053858, 0.277871, 0.086249, 1.000000, 0.711683, 0.318528, 0.207507, 0.262590, 0.451641, 0.262717, 0.114672, 0.506146, 0.176017, 0.104403, 0.025953, 0.027905, 0.039973, 0.013096, 0.057863, 0.073509, 0.012798, 0.015880, 0.019534, 0.006723, 0.014531, 0.011033, 0.008811, 0.008016, 0.015216, 0.006268, 0.002561, 0.000664, 0.002042, 0.003062, 0.001679, 0.001879, 0.000752, 0.001087, 0.001099, 0.001105, 0.001938, 0.001332, 0.000569, 0.000729, 0.000481, 0.000526, 0.000524, 0.000539, 0.000408, 0.000292, 0.000366, 0.000366, 0.000340, 0.000314, 0.000292, 0.000297, 0.000294, 0.000283, 0.000279, 0.000241, 0.000281, 0.000236, 0.000254],
                  [0.028926, 0.053858, 0.277871, 0.086249, 1.000000, 0.711683, 0.318528, 0.207507, 0.262590, 0.451641, 0.262717, 0.114672, 0.506146, 0.176017, 0.104403, 0.025953, 0.027905, 0.039973, 0.013096, 0.057863, 0.073509, 0.012798, 0.015880, 0.019534, 0.006723, 0.014531, 0.011033, 0.008811, 0.008016, 0.015216, 0.006268, 0.002561, 0.000664, 0.002042, 0.003062, 0.001679, 0.001879, 0.000752, 0.001087, 0.001099, 0.001105, 0.001938, 0.001332, 0.000569, 0.000729, 0.000481, 0.000526, 0.000524, 0.000539, 0.000408, 0.000292, 0.000366, 0.000366, 0.000340, 0.000314, 0.000292, 0.000297, 0.000294, 0.000283, 0.000279, 0.000241, 0.000281, 0.000236, 0.000254],
                  [0.028926, 0.053858, 0.277871, 0.086249, 1.000000, 0.711683, 0.318528, 0.207507, 0.262590, 0.451641, 0.262717, 0.114672, 0.506146, 0.176017, 0.104403, 0.025953, 0.027905, 0.039973, 0.013096, 0.057863, 0.073509, 0.012798, 0.015880, 0.019534, 0.006723, 0.014531, 0.011033, 0.008811, 0.008016, 0.015216, 0.006268, 0.002561, 0.000664, 0.002042, 0.003062, 0.001679, 0.001879, 0.000752, 0.001087, 0.001099, 0.001105, 0.001938, 0.001332, 0.000569, 0.000729, 0.000481, 0.000526, 0.000524, 0.000539, 0.000408, 0.000292, 0.000366, 0.000366, 0.000340, 0.000314, 0.000292, 0.000297, 0.000294, 0.000283, 0.000279, 0.000241, 0.000281, 0.000236, 0.000254],
                  [0.028926, 0.053858, 0.277871, 0.086249, 1.000000, 0.711683, 0.318528, 0.207507, 0.262590, 0.451641, 0.262717, 0.114672, 0.506146, 0.176017, 0.104403, 0.025953, 0.027905, 0.039973, 0.013096, 0.057863, 0.073509, 0.012798, 0.015880, 0.019534, 0.006723, 0.014531, 0.011033, 0.008811, 0.008016, 0.015216, 0.006268, 0.002561, 0.000664, 0.002042, 0.003062, 0.001679, 0.001879, 0.000752, 0.001087, 0.001099, 0.001105, 0.001938, 0.001332, 0.000569, 0.000729, 0.000481, 0.000526, 0.000524, 0.000539, 0.000408, 0.000292, 0.000366, 0.000366, 0.000340, 0.000314, 0.000292, 0.000297, 0.000294, 0.000283, 0.000279, 0.000241, 0.000281, 0.000236, 0.000254],
                  [0.028926, 0.053858, 0.277871, 0.086249, 1.000000, 0.711683, 0.318528, 0.207507, 0.262590, 0.451641, 0.262717, 0.114672, 0.506146, 0.176017, 0.104403, 0.025953, 0.027905, 0.039973, 0.013096, 0.057863, 0.073509, 0.012798, 0.015880, 0.019534, 0.006723, 0.014531, 0.011033, 0.008811, 0.008016, 0.015216, 0.006268, 0.002561, 0.000664, 0.002042, 0.003062, 0.001679, 0.001879, 0.000752, 0.001087, 0.001099, 0.001105, 0.001938, 0.001332, 0.000569, 0.000729, 0.000481, 0.000526, 0.000524, 0.000539, 0.000408, 0.000292, 0.000366, 0.000366, 0.000340, 0.000314, 0.000292, 0.000297, 0.000294, 0.000283, 0.000279, 0.000241, 0.000281, 0.000236, 0.000254],
                  [0.033399, 0.025462, 0.019387, 0.049802, 0.357963, 1.000000, 0.323696, 0.062014, 0.053868, 0.077788, 0.071108, 0.058877, 0.059175, 0.030739, 0.047140, 0.024788, 0.018791, 0.007839, 0.006454, 0.004520, 0.003856, 0.003913, 0.008032, 0.009742, 0.003604, 0.003221, 0.001299, 0.002013, 0.004373, 0.001504, 0.001618, 0.002084, 0.000994, 0.001040, 0.001969, 0.002784, 0.002299, 0.001454, 0.000802, 0.000515, 0.000321, 0.000264, 0.000505, 0.000456, 0.000415, 0.000444, 0.000395, 0.000428, 0.000461, 0.000341, 0.000346, 0.000378, 0.000422, 0.000309, 0.000382, 0.000397, 0.000288, 0.000356, 0.000311, 0.000359, 0.000360, 0.000306, 0.000302, 0.000301],
                  [0.033399, 0.025462, 0.019387, 0.049802, 0.357963, 1.000000, 0.323696, 0.062014, 0.053868, 0.077788, 0.071108, 0.058877, 0.059175, 0.030739, 0.047140, 0.024788, 0.018791, 0.007839, 0.006454, 0.004520, 0.003856, 0.003913, 0.008032, 0.009742, 0.003604, 0.003221, 0.001299, 0.002013, 0.004373, 0.001504, 0.001618, 0.002084, 0.000994, 0.001040, 0.001969, 0.002784, 0.002299, 0.001454, 0.000802, 0.000515, 0.000321, 0.000264, 0.000505, 0.000456, 0.000415, 0.000444, 0.000395, 0.000428, 0.000461, 0.000341, 0.000346, 0.000378, 0.000422, 0.000309, 0.000382, 0.000397, 0.000288, 0.000356, 0.000311, 0.000359, 0.000360, 0.000306, 0.000302, 0.000301],
                  [0.033399, 0.025462, 0.019387, 0.049802, 0.357963, 1.000000, 0.323696, 0.062014, 0.053868, 0.077788, 0.071108, 0.058877, 0.059175, 0.030739, 0.047140, 0.024788, 0.018791, 0.007839, 0.006454, 0.004520, 0.003856, 0.003913, 0.008032, 0.009742, 0.003604, 0.003221, 0.001299, 0.002013, 0.004373, 0.001504, 0.001618, 0.002084, 0.000994, 0.001040, 0.001969, 0.002784, 0.002299, 0.001454, 0.000802, 0.000515, 0.000321, 0.000264, 0.000505, 0.000456, 0.000415, 0.000444, 0.000395, 0.000428, 0.000461, 0.000341, 0.000346, 0.000378, 0.000422, 0.000309, 0.000382, 0.000397, 0.000288, 0.000356, 0.000311, 0.000359, 0.000360, 0.000306, 0.000302, 0.000301],
                  [0.033399, 0.025462, 0.019387, 0.049802, 0.357963, 1.000000, 0.323696, 0.062014, 0.053868, 0.077788, 0.071108, 0.058877, 0.059175, 0.030739, 0.047140, 0.024788, 0.018791, 0.007839, 0.006454, 0.004520, 0.003856, 0.003913, 0.008032, 0.009742, 0.003604, 0.003221, 0.001299, 0.002013, 0.004373, 0.001504, 0.001618, 0.002084, 0.000994, 0.001040, 0.001969, 0.002784, 0.002299, 0.001454, 0.000802, 0.000515, 0.000321, 0.000264, 0.000505, 0.000456, 0.000415, 0.000444, 0.000395, 0.000428, 0.000461, 0.000341, 0.000346, 0.000378, 0.000422, 0.000309, 0.000382, 0.000397, 0.000288, 0.000356, 0.000311, 0.000359, 0.000360, 0.000306, 0.000302, 0.000301],
                  [0.033399, 0.025462, 0.019387, 0.049802, 0.357963, 1.000000, 0.323696, 0.062014, 0.053868, 0.077788, 0.071108, 0.058877, 0.059175, 0.030739, 0.047140, 0.024788, 0.018791, 0.007839, 0.006454, 0.004520, 0.003856, 0.003913, 0.008032, 0.009742, 0.003604, 0.003221, 0.001299, 0.002013, 0.004373, 0.001504, 0.001618, 0.002084, 0.000994, 0.001040, 0.001969, 0.002784, 0.002299, 0.001454, 0.000802, 0.000515, 0.000321, 0.000264, 0.000505, 0.000456, 0.000415, 0.000444, 0.000395, 0.000428, 0.000461, 0.000341, 0.000346, 0.000378, 0.000422, 0.000309, 0.000382, 0.000397, 0.000288, 0.000356, 0.000311, 0.000359, 0.000360, 0.000306, 0.000302, 0.000301],
                  [0.033399, 0.025462, 0.019387, 0.049802, 0.357963, 1.000000, 0.323696, 0.062014, 0.053868, 0.077788, 0.071108, 0.058877, 0.059175, 0.030739, 0.047140, 0.024788, 0.018791, 0.007839, 0.006454, 0.004520, 0.003856, 0.003913, 0.008032, 0.009742, 0.003604, 0.003221, 0.001299, 0.002013, 0.004373, 0.001504, 0.001618, 0.002084, 0.000994, 0.001040, 0.001969, 0.002784, 0.002299, 0.001454, 0.000802, 0.000515, 0.000321, 0.000264, 0.000505, 0.000456, 0.000415, 0.000444, 0.000395, 0.000428, 0.000461, 0.000341, 0.000346, 0.000378, 0.000422, 0.000309, 0.000382, 0.000397, 0.000288, 0.000356, 0.000311, 0.000359, 0.000360, 0.000306, 0.000302, 0.000301],
                  [0.033399, 0.025462, 0.019387, 0.049802, 0.357963, 1.000000, 0.323696, 0.062014, 0.053868, 0.077788, 0.071108, 0.058877, 0.059175, 0.030739, 0.047140, 0.024788, 0.018791, 0.007839, 0.006454, 0.004520, 0.003856, 0.003913, 0.008032, 0.009742, 0.003604, 0.003221, 0.001299, 0.002013, 0.004373, 0.001504, 0.001618, 0.002084, 0.000994, 0.001040, 0.001969, 0.002784, 0.002299, 0.001454, 0.000802, 0.000515, 0.000321, 0.000264, 0.000505, 0.000456, 0.000415, 0.000444, 0.000395, 0.000428, 0.000461, 0.000341, 0.000346, 0.000378, 0.000422, 0.000309, 0.000382, 0.000397, 0.000288, 0.000356, 0.000311, 0.000359, 0.000360, 0.000306, 0.000302, 0.000301],
                  [0.033399, 0.025462, 0.019387, 0.049802, 0.357963, 1.000000, 0.323696, 0.062014, 0.053868, 0.077788, 0.071108, 0.058877, 0.059175, 0.030739, 0.047140, 0.024788, 0.018791, 0.007839, 0.006454, 0.004520, 0.003856, 0.003913, 0.008032, 0.009742, 0.003604, 0.003221, 0.001299, 0.002013, 0.004373, 0.001504, 0.001618, 0.002084, 0.000994, 0.001040, 0.001969, 0.002784, 0.002299, 0.001454, 0.000802, 0.000515, 0.000321, 0.000264, 0.000505, 0.000456, 0.000415, 0.000444, 0.000395, 0.000428, 0.000461, 0.000341, 0.000346, 0.000378, 0.000422, 0.000309, 0.000382, 0.000397, 0.000288, 0.000356, 0.000311, 0.000359, 0.000360, 0.000306, 0.000302, 0.000301],
                  [0.033399, 0.025462, 0.019387, 0.049802, 0.357963, 1.000000, 0.323696, 0.062014, 0.053868, 0.077788, 0.071108, 0.058877, 0.059175, 0.030739, 0.047140, 0.024788, 0.018791, 0.007839, 0.006454, 0.004520, 0.003856, 0.003913, 0.008032, 0.009742, 0.003604, 0.003221, 0.001299, 0.002013, 0.004373, 0.001504, 0.001618, 0.002084, 0.000994, 0.001040, 0.001969, 0.002784, 0.002299, 0.001454, 0.000802, 0.000515, 0.000321, 0.000264, 0.000505, 0.000456, 0.000415, 0.000444, 0.000395, 0.000428, 0.000461, 0.000341, 0.000346, 0.000378, 0.000422, 0.000309, 0.000382, 0.000397, 0.000288, 0.000356, 0.000311, 0.000359, 0.000360, 0.000306, 0.000302, 0.000301],
                  [0.033399, 0.025462, 0.019387, 0.049802, 0.357963, 1.000000, 0.323696, 0.062014, 0.053868, 0.077788, 0.071108, 0.058877, 0.059175, 0.030739, 0.047140, 0.024788, 0.018791, 0.007839, 0.006454, 0.004520, 0.003856, 0.003913, 0.008032, 0.009742, 0.003604, 0.003221, 0.001299, 0.002013, 0.004373, 0.001504, 0.001618, 0.002084, 0.000994, 0.001040, 0.001969, 0.002784, 0.002299, 0.001454, 0.000802, 0.000515, 0.000321, 0.000264, 0.000505, 0.000456, 0.000415, 0.000444, 0.000395, 0.000428, 0.000461, 0.000341, 0.000346, 0.000378, 0.000422, 0.000309, 0.000382, 0.000397, 0.000288, 0.000356, 0.000311, 0.000359, 0.000360, 0.000306, 0.000302, 0.000301],
                  [0.033399, 0.025462, 0.019387, 0.049802, 0.357963, 1.000000, 0.323696, 0.062014, 0.053868, 0.077788, 0.071108, 0.058877, 0.059175, 0.030739, 0.047140, 0.024788, 0.018791, 0.007839, 0.006454, 0.004520, 0.003856, 0.003913, 0.008032, 0.009742, 0.003604, 0.003221, 0.001299, 0.002013, 0.004373, 0.001504, 0.001618, 0.002084, 0.000994, 0.001040, 0.001969, 0.002784, 0.002299, 0.001454, 0.000802, 0.000515, 0.000321, 0.000264, 0.000505, 0.000456, 0.000415, 0.000444, 0.000395, 0.000428, 0.000461, 0.000341, 0.000346, 0.000378, 0.000422, 0.000309, 0.000382, 0.000397, 0.000288, 0.000356, 0.000311, 0.000359, 0.000360, 0.000306, 0.000302, 0.000301],
                  [0.033399, 0.025462, 0.019387, 0.049802, 0.357963, 1.000000, 0.323696, 0.062014, 0.053868, 0.077788, 0.071108, 0.058877, 0.059175, 0.030739, 0.047140, 0.024788, 0.018791, 0.007839, 0.006454, 0.004520, 0.003856, 0.003913, 0.008032, 0.009742, 0.003604, 0.003221, 0.001299, 0.002013, 0.004373, 0.001504, 0.001618, 0.002084, 0.000994, 0.001040, 0.001969, 0.002784, 0.002299, 0.001454, 0.000802, 0.000515, 0.000321, 0.000264, 0.000505, 0.000456, 0.000415, 0.000444, 0.000395, 0.000428, 0.000461, 0.000341, 0.000346, 0.000378, 0.000422, 0.000309, 0.000382, 0.000397, 0.000288, 0.000356, 0.000311, 0.000359, 0.000360, 0.000306, 0.000302, 0.000301],
                  [0.033399, 0.025462, 0.019387, 0.049802, 0.357963, 1.000000, 0.323696, 0.062014, 0.053868, 0.077788, 0.071108, 0.058877, 0.059175, 0.030739, 0.047140, 0.024788, 0.018791, 0.007839, 0.006454, 0.004520, 0.003856, 0.003913, 0.008032, 0.009742, 0.003604, 0.003221, 0.001299, 0.002013, 0.004373, 0.001504, 0.001618, 0.002084, 0.000994, 0.001040, 0.001969, 0.002784, 0.002299, 0.001454, 0.000802, 0.000515, 0.000321, 0.000264, 0.000505, 0.000456, 0.000415, 0.000444, 0.000395, 0.000428, 0.000461, 0.000341, 0.000346, 0.000378, 0.000422, 0.000309, 0.000382, 0.000397, 0.000288, 0.000356, 0.000311, 0.000359, 0.000360, 0.000306, 0.000302, 0.000301],
                  [0.033399, 0.025462, 0.019387, 0.049802, 0.357963, 1.000000, 0.323696, 0.062014, 0.053868, 0.077788, 0.071108, 0.058877, 0.059175, 0.030739, 0.047140, 0.024788, 0.018791, 0.007839, 0.006454, 0.004520, 0.003856, 0.003913, 0.008032, 0.009742, 0.003604, 0.003221, 0.001299, 0.002013, 0.004373, 0.001504, 0.001618, 0.002084, 0.000994, 0.001040, 0.001969, 0.002784, 0.002299, 0.001454, 0.000802, 0.000515, 0.000321, 0.000264, 0.000505, 0.000456, 0.000415, 0.000444, 0.000395, 0.000428, 0.000461, 0.000341, 0.000346, 0.000378, 0.000422, 0.000309, 0.000382, 0.000397, 0.000288, 0.000356, 0.000311, 0.000359, 0.000360, 0.000306, 0.000302, 0.000301],
                  [0.006673, 0.011714, 0.042924, 0.193967, 0.573656, 1.000000, 0.823029, 0.395012, 0.178624, 0.129677, 0.157677, 0.192950, 0.160184, 0.102019, 0.068875, 0.052161, 0.041308, 0.041617, 0.045708, 0.034254, 0.033598, 0.045841, 0.018602, 0.010012, 0.021323, 0.006818, 0.007099, 0.006677, 0.003246, 0.004373, 0.006728, 0.007424, 0.004747, 0.003134, 0.002396, 0.001811, 0.001388, 0.001820, 0.000957, 0.000711, 0.000577, 0.000576, 0.000686, 0.000450, 0.000403, 0.000634, 0.000495, 0.000377, 0.000315, 0.000377, 0.000293, 0.000364, 0.000350, 0.000296, 0.000345, 0.000325, 0.000295, 0.000307, 0.000281, 0.000283, 0.000371, 0.000293, 0.000263, 0.000270],
                  [0.006673, 0.011714, 0.042924, 0.193967, 0.573656, 1.000000, 0.823029, 0.395012, 0.178624, 0.129677, 0.157677, 0.192950, 0.160184, 0.102019, 0.068875, 0.052161, 0.041308, 0.041617, 0.045708, 0.034254, 0.033598, 0.045841, 0.018602, 0.010012, 0.021323, 0.006818, 0.007099, 0.006677, 0.003246, 0.004373, 0.006728, 0.007424, 0.004747, 0.003134, 0.002396, 0.001811, 0.001388, 0.001820, 0.000957, 0.000711, 0.000577, 0.000576, 0.000686, 0.000450, 0.000403, 0.000634, 0.000495, 0.000377, 0.000315, 0.000377, 0.000293, 0.000364, 0.000350, 0.000296, 0.000345, 0.000325, 0.000295, 0.000307, 0.000281, 0.000283, 0.000371, 0.000293, 0.000263, 0.000270],
                  [0.006673, 0.011714, 0.042924, 0.193967, 0.573656, 1.000000, 0.823029, 0.395012, 0.178624, 0.129677, 0.157677, 0.192950, 0.160184, 0.102019, 0.068875, 0.052161, 0.041308, 0.041617, 0.045708, 0.034254, 0.033598, 0.045841, 0.018602, 0.010012, 0.021323, 0.006818, 0.007099, 0.006677, 0.003246, 0.004373, 0.006728, 0.007424, 0.004747, 0.003134, 0.002396, 0.001811, 0.001388, 0.001820, 0.000957, 0.000711, 0.000577, 0.000576, 0.000686, 0.000450, 0.000403, 0.000634, 0.000495, 0.000377, 0.000315, 0.000377, 0.000293, 0.000364, 0.000350, 0.000296, 0.000345, 0.000325, 0.000295, 0.000307, 0.000281, 0.000283, 0.000371, 0.000293, 0.000263, 0.000270],
                  [0.006673, 0.011714, 0.042924, 0.193967, 0.573656, 1.000000, 0.823029, 0.395012, 0.178624, 0.129677, 0.157677, 0.192950, 0.160184, 0.102019, 0.068875, 0.052161, 0.041308, 0.041617, 0.045708, 0.034254, 0.033598, 0.045841, 0.018602, 0.010012, 0.021323, 0.006818, 0.007099, 0.006677, 0.003246, 0.004373, 0.006728, 0.007424, 0.004747, 0.003134, 0.002396, 0.001811, 0.001388, 0.001820, 0.000957, 0.000711, 0.000577, 0.000576, 0.000686, 0.000450, 0.000403, 0.000634, 0.000495, 0.000377, 0.000315, 0.000377, 0.000293, 0.000364, 0.000350, 0.000296, 0.000345, 0.000325, 0.000295, 0.000307, 0.000281, 0.000283, 0.000371, 0.000293, 0.000263, 0.000270],
                  [0.006673, 0.011714, 0.042924, 0.193967, 0.573656, 1.000000, 0.823029, 0.395012, 0.178624, 0.129677, 0.157677, 0.192950, 0.160184, 0.102019, 0.068875, 0.052161, 0.041308, 0.041617, 0.045708, 0.034254, 0.033598, 0.045841, 0.018602, 0.010012, 0.021323, 0.006818, 0.007099, 0.006677, 0.003246, 0.004373, 0.006728, 0.007424, 0.004747, 0.003134, 0.002396, 0.001811, 0.001388, 0.001820, 0.000957, 0.000711, 0.000577, 0.000576, 0.000686, 0.000450, 0.000403, 0.000634, 0.000495, 0.000377, 0.000315, 0.000377, 0.000293, 0.000364, 0.000350, 0.000296, 0.000345, 0.000325, 0.000295, 0.000307, 0.000281, 0.000283, 0.000371, 0.000293, 0.000263, 0.000270],
                  [0.006673, 0.011714, 0.042924, 0.193967, 0.573656, 1.000000, 0.823029, 0.395012, 0.178624, 0.129677, 0.157677, 0.192950, 0.160184, 0.102019, 0.068875, 0.052161, 0.041308, 0.041617, 0.045708, 0.034254, 0.033598, 0.045841, 0.018602, 0.010012, 0.021323, 0.006818, 0.007099, 0.006677, 0.003246, 0.004373, 0.006728, 0.007424, 0.004747, 0.003134, 0.002396, 0.001811, 0.001388, 0.001820, 0.000957, 0.000711, 0.000577, 0.000576, 0.000686, 0.000450, 0.000403, 0.000634, 0.000495, 0.000377, 0.000315, 0.000377, 0.000293, 0.000364, 0.000350, 0.000296, 0.000345, 0.000325, 0.000295, 0.000307, 0.000281, 0.000283, 0.000371, 0.000293, 0.000263, 0.000270],
                  [0.006673, 0.011714, 0.042924, 0.193967, 0.573656, 1.000000, 0.823029, 0.395012, 0.178624, 0.129677, 0.157677, 0.192950, 0.160184, 0.102019, 0.068875, 0.052161, 0.041308, 0.041617, 0.045708, 0.034254, 0.033598, 0.045841, 0.018602, 0.010012, 0.021323, 0.006818, 0.007099, 0.006677, 0.003246, 0.004373, 0.006728, 0.007424, 0.004747, 0.003134, 0.002396, 0.001811, 0.001388, 0.001820, 0.000957, 0.000711, 0.000577, 0.000576, 0.000686, 0.000450, 0.000403, 0.000634, 0.000495, 0.000377, 0.000315, 0.000377, 0.000293, 0.000364, 0.000350, 0.000296, 0.000345, 0.000325, 0.000295, 0.000307, 0.000281, 0.000283, 0.000371, 0.000293, 0.000263, 0.000270],
                  [0.006673, 0.011714, 0.042924, 0.193967, 0.573656, 1.000000, 0.823029, 0.395012, 0.178624, 0.129677, 0.157677, 0.192950, 0.160184, 0.102019, 0.068875, 0.052161, 0.041308, 0.041617, 0.045708, 0.034254, 0.033598, 0.045841, 0.018602, 0.010012, 0.021323, 0.006818, 0.007099, 0.006677, 0.003246, 0.004373, 0.006728, 0.007424, 0.004747, 0.003134, 0.002396, 0.001811, 0.001388, 0.001820, 0.000957, 0.000711, 0.000577, 0.000576, 0.000686, 0.000450, 0.000403, 0.000634, 0.000495, 0.000377, 0.000315, 0.000377, 0.000293, 0.000364, 0.000350, 0.000296, 0.000345, 0.000325, 0.000295, 0.000307, 0.000281, 0.000283, 0.000371, 0.000293, 0.000263, 0.000270],
                  [0.006673, 0.011714, 0.042924, 0.193967, 0.573656, 1.000000, 0.823029, 0.395012, 0.178624, 0.129677, 0.157677, 0.192950, 0.160184, 0.102019, 0.068875, 0.052161, 0.041308, 0.041617, 0.045708, 0.034254, 0.033598, 0.045841, 0.018602, 0.010012, 0.021323, 0.006818, 0.007099, 0.006677, 0.003246, 0.004373, 0.006728, 0.007424, 0.004747, 0.003134, 0.002396, 0.001811, 0.001388, 0.001820, 0.000957, 0.000711, 0.000577, 0.000576, 0.000686, 0.000450, 0.000403, 0.000634, 0.000495, 0.000377, 0.000315, 0.000377, 0.000293, 0.000364, 0.000350, 0.000296, 0.000345, 0.000325, 0.000295, 0.000307, 0.000281, 0.000283, 0.000371, 0.000293, 0.000263, 0.000270],
                  [0.006673, 0.011714, 0.042924, 0.193967, 0.573656, 1.000000, 0.823029, 0.395012, 0.178624, 0.129677, 0.157677, 0.192950, 0.160184, 0.102019, 0.068875, 0.052161, 0.041308, 0.041617, 0.045708, 0.034254, 0.033598, 0.045841, 0.018602, 0.010012, 0.021323, 0.006818, 0.007099, 0.006677, 0.003246, 0.004373, 0.006728, 0.007424, 0.004747, 0.003134, 0.002396, 0.001811, 0.001388, 0.001820, 0.000957, 0.000711, 0.000577, 0.000576, 0.000686, 0.000450, 0.000403, 0.000634, 0.000495, 0.000377, 0.000315, 0.000377, 0.000293, 0.000364, 0.000350, 0.000296, 0.000345, 0.000325, 0.000295, 0.000307, 0.000281, 0.000283, 0.000371, 0.000293, 0.000263, 0.000270],
                  [0.017745, 0.021170, 0.032848, 0.063984, 0.127747, 0.276760, 0.556973, 0.868409, 1.000000, 0.817335, 0.527598, 0.315382, 0.214253, 0.189557, 0.197282, 0.190805, 0.136301, 0.072581, 0.041272, 0.040100, 0.066894, 0.100421, 0.080420, 0.041788, 0.033754, 0.046999, 0.042883, 0.020654, 0.016134, 0.019743, 0.013403, 0.010764, 0.011333, 0.006327, 0.007855, 0.007153, 0.004524, 0.004398, 0.004477, 0.002790, 0.001640, 0.002304, 0.001327, 0.001289, 0.001276, 0.001282, 0.001437, 0.001162, 0.001126, 0.001185, 0.000845, 0.000824, 0.000681, 0.000478, 0.000507, 0.000436, 0.000542, 0.000492, 0.000483, 0.000550, 0.000498, 0.000473, 0.000382, 0.000417],
                  [0.017745, 0.021170, 0.032848, 0.063984, 0.127747, 0.276760, 0.556973, 0.868409, 1.000000, 0.817335, 0.527598, 0.315382, 0.214253, 0.189557, 0.197282, 0.190805, 0.136301, 0.072581, 0.041272, 0.040100, 0.066894, 0.100421, 0.080420, 0.041788, 0.033754, 0.046999, 0.042883, 0.020654, 0.016134, 0.019743, 0.013403, 0.010764, 0.011333, 0.006327, 0.007855, 0.007153, 0.004524, 0.004398, 0.004477, 0.002790, 0.001640, 0.002304, 0.001327, 0.001289, 0.001276, 0.001282, 0.001437, 0.001162, 0.001126, 0.001185, 0.000845, 0.000824, 0.000681, 0.000478, 0.000507, 0.000436, 0.000542, 0.000492, 0.000483, 0.000550, 0.000498, 0.000473, 0.000382, 0.000417],
                  [0.017745, 0.021170, 0.032848, 0.063984, 0.127747, 0.276760, 0.556973, 0.868409, 1.000000, 0.817335, 0.527598, 0.315382, 0.214253, 0.189557, 0.197282, 0.190805, 0.136301, 0.072581, 0.041272, 0.040100, 0.066894, 0.100421, 0.080420, 0.041788, 0.033754, 0.046999, 0.042883, 0.020654, 0.016134, 0.019743, 0.013403, 0.010764, 0.011333, 0.006327, 0.007855, 0.007153, 0.004524, 0.004398, 0.004477, 0.002790, 0.001640, 0.002304, 0.001327, 0.001289, 0.001276, 0.001282, 0.001437, 0.001162, 0.001126, 0.001185, 0.000845, 0.000824, 0.000681, 0.000478, 0.000507, 0.000436, 0.000542, 0.000492, 0.000483, 0.000550, 0.000498, 0.000473, 0.000382, 0.000417],
                  [0.017745, 0.021170, 0.032848, 0.063984, 0.127747, 0.276760, 0.556973, 0.868409, 1.000000, 0.817335, 0.527598, 0.315382, 0.214253, 0.189557, 0.197282, 0.190805, 0.136301, 0.072581, 0.041272, 0.040100, 0.066894, 0.100421, 0.080420, 0.041788, 0.033754, 0.046999, 0.042883, 0.020654, 0.016134, 0.019743, 0.013403, 0.010764, 0.011333, 0.006327, 0.007855, 0.007153, 0.004524, 0.004398, 0.004477, 0.002790, 0.001640, 0.002304, 0.001327, 0.001289, 0.001276, 0.001282, 0.001437, 0.001162, 0.001126, 0.001185, 0.000845, 0.000824, 0.000681, 0.000478, 0.000507, 0.000436, 0.000542, 0.000492, 0.000483, 0.000550, 0.000498, 0.000473, 0.000382, 0.000417],
                  [0.017745, 0.021170, 0.032848, 0.063984, 0.127747, 0.276760, 0.556973, 0.868409, 1.000000, 0.817335, 0.527598, 0.315382, 0.214253, 0.189557, 0.197282, 0.190805, 0.136301, 0.072581, 0.041272, 0.040100, 0.066894, 0.100421, 0.080420, 0.041788, 0.033754, 0.046999, 0.042883, 0.020654, 0.016134, 0.019743, 0.013403, 0.010764, 0.011333, 0.006327, 0.007855, 0.007153, 0.004524, 0.004398, 0.004477, 0.002790, 0.001640, 0.002304, 0.001327, 0.001289, 0.001276, 0.001282, 0.001437, 0.001162, 0.001126, 0.001185, 0.000845, 0.000824, 0.000681, 0.000478, 0.000507, 0.000436, 0.000542, 0.000492, 0.000483, 0.000550, 0.000498, 0.000473, 0.000382, 0.000417],
                  [0.017745, 0.021170, 0.032848, 0.063984, 0.127747, 0.276760, 0.556973, 0.868409, 1.000000, 0.817335, 0.527598, 0.315382, 0.214253, 0.189557, 0.197282, 0.190805, 0.136301, 0.072581, 0.041272, 0.040100, 0.066894, 0.100421, 0.080420, 0.041788, 0.033754, 0.046999, 0.042883, 0.020654, 0.016134, 0.019743, 0.013403, 0.010764, 0.011333, 0.006327, 0.007855, 0.007153, 0.004524, 0.004398, 0.004477, 0.002790, 0.001640, 0.002304, 0.001327, 0.001289, 0.001276, 0.001282, 0.001437, 0.001162, 0.001126, 0.001185, 0.000845, 0.000824, 0.000681, 0.000478, 0.000507, 0.000436, 0.000542, 0.000492, 0.000483, 0.000550, 0.000498, 0.000473, 0.000382, 0.000417],
                  [0.017745, 0.021170, 0.032848, 0.063984, 0.127747, 0.276760, 0.556973, 0.868409, 1.000000, 0.817335, 0.527598, 0.315382, 0.214253, 0.189557, 0.197282, 0.190805, 0.136301, 0.072581, 0.041272, 0.040100, 0.066894, 0.100421, 0.080420, 0.041788, 0.033754, 0.046999, 0.042883, 0.020654, 0.016134, 0.019743, 0.013403, 0.010764, 0.011333, 0.006327, 0.007855, 0.007153, 0.004524, 0.004398, 0.004477, 0.002790, 0.001640, 0.002304, 0.001327, 0.001289, 0.001276, 0.001282, 0.001437, 0.001162, 0.001126, 0.001185, 0.000845, 0.000824, 0.000681, 0.000478, 0.000507, 0.000436, 0.000542, 0.000492, 0.000483, 0.000550, 0.000498, 0.000473, 0.000382, 0.000417],
                  [0.017745, 0.021170, 0.032848, 0.063984, 0.127747, 0.276760, 0.556973, 0.868409, 1.000000, 0.817335, 0.527598, 0.315382, 0.214253, 0.189557, 0.197282, 0.190805, 0.136301, 0.072581, 0.041272, 0.040100, 0.066894, 0.100421, 0.080420, 0.041788, 0.033754, 0.046999, 0.042883, 0.020654, 0.016134, 0.019743, 0.013403, 0.010764, 0.011333, 0.006327, 0.007855, 0.007153, 0.004524, 0.004398, 0.004477, 0.002790, 0.001640, 0.002304, 0.001327, 0.001289, 0.001276, 0.001282, 0.001437, 0.001162, 0.001126, 0.001185, 0.000845, 0.000824, 0.000681, 0.000478, 0.000507, 0.000436, 0.000542, 0.000492, 0.000483, 0.000550, 0.000498, 0.000473, 0.000382, 0.000417],
                  [0.017745, 0.021170, 0.032848, 0.063984, 0.127747, 0.276760, 0.556973, 0.868409, 1.000000, 0.817335, 0.527598, 0.315382, 0.214253, 0.189557, 0.197282, 0.190805, 0.136301, 0.072581, 0.041272, 0.040100, 0.066894, 0.100421, 0.080420, 0.041788, 0.033754, 0.046999, 0.042883, 0.020654, 0.016134, 0.019743, 0.013403, 0.010764, 0.011333, 0.006327, 0.007855, 0.007153, 0.004524, 0.004398, 0.004477, 0.002790, 0.001640, 0.002304, 0.001327, 0.001289, 0.001276, 0.001282, 0.001437, 0.001162, 0.001126, 0.001185, 0.000845, 0.000824, 0.000681, 0.000478, 0.000507, 0.000436, 0.000542, 0.000492, 0.000483, 0.000550, 0.000498, 0.000473, 0.000382, 0.000417],
                  [0.017745, 0.021170, 0.032848, 0.063984, 0.127747, 0.276760, 0.556973, 0.868409, 1.000000, 0.817335, 0.527598, 0.315382, 0.214253, 0.189557, 0.197282, 0.190805, 0.136301, 0.072581, 0.041272, 0.040100, 0.066894, 0.100421, 0.080420, 0.041788, 0.033754, 0.046999, 0.042883, 0.020654, 0.016134, 0.019743, 0.013403, 0.010764, 0.011333, 0.006327, 0.007855, 0.007153, 0.004524, 0.004398, 0.004477, 0.002790, 0.001640, 0.002304, 0.001327, 0.001289, 0.001276, 0.001282, 0.001437, 0.001162, 0.001126, 0.001185, 0.000845, 0.000824, 0.000681, 0.000478, 0.000507, 0.000436, 0.000542, 0.000492, 0.000483, 0.000550, 0.000498, 0.000473, 0.000382, 0.000417],
                  [0.017745, 0.021170, 0.032848, 0.063984, 0.127747, 0.276760, 0.556973, 0.868409, 1.000000, 0.817335, 0.527598, 0.315382, 0.214253, 0.189557, 0.197282, 0.190805, 0.136301, 0.072581, 0.041272, 0.040100, 0.066894, 0.100421, 0.080420, 0.041788, 0.033754, 0.046999, 0.042883, 0.020654, 0.016134, 0.019743, 0.013403, 0.010764, 0.011333, 0.006327, 0.007855, 0.007153, 0.004524, 0.004398, 0.004477, 0.002790, 0.001640, 0.002304, 0.001327, 0.001289, 0.001276, 0.001282, 0.001437, 0.001162, 0.001126, 0.001185, 0.000845, 0.000824, 0.000681, 0.000478, 0.000507, 0.000436, 0.000542, 0.000492, 0.000483, 0.000550, 0.000498, 0.000473, 0.000382, 0.000417],
                  [0.017745, 0.021170, 0.032848, 0.063984, 0.127747, 0.276760, 0.556973, 0.868409, 1.000000, 0.817335, 0.527598, 0.315382, 0.214253, 0.189557, 0.197282, 0.190805, 0.136301, 0.072581, 0.041272, 0.040100, 0.066894, 0.100421, 0.080420, 0.041788, 0.033754, 0.046999, 0.042883, 0.020654, 0.016134, 0.019743, 0.013403, 0.010764, 0.011333, 0.006327, 0.007855, 0.007153, 0.004524, 0.004398, 0.004477, 0.002790, 0.001640, 0.002304, 0.001327, 0.001289, 0.001276, 0.001282, 0.001437, 0.001162, 0.001126, 0.001185, 0.000845, 0.000824, 0.000681, 0.000478, 0.000507, 0.000436, 0.000542, 0.000492, 0.000483, 0.000550, 0.000498, 0.000473, 0.000382, 0.000417],
                  [0.017745, 0.021170, 0.032848, 0.063984, 0.127747, 0.276760, 0.556973, 0.868409, 1.000000, 0.817335, 0.527598, 0.315382, 0.214253, 0.189557, 0.197282, 0.190805, 0.136301, 0.072581, 0.041272, 0.040100, 0.066894, 0.100421, 0.080420, 0.041788, 0.033754, 0.046999, 0.042883, 0.020654, 0.016134, 0.019743, 0.013403, 0.010764, 0.011333, 0.006327, 0.007855, 0.007153, 0.004524, 0.004398, 0.004477, 0.002790, 0.001640, 0.002304, 0.001327, 0.001289, 0.001276, 0.001282, 0.001437, 0.001162, 0.001126, 0.001185, 0.000845, 0.000824, 0.000681, 0.000478, 0.000507, 0.000436, 0.000542, 0.000492, 0.000483, 0.000550, 0.000498, 0.000473, 0.000382, 0.000417],
                  [0.017745, 0.021170, 0.032848, 0.063984, 0.127747, 0.276760, 0.556973, 0.868409, 1.000000, 0.817335, 0.527598, 0.315382, 0.214253, 0.189557, 0.197282, 0.190805, 0.136301, 0.072581, 0.041272, 0.040100, 0.066894, 0.100421, 0.080420, 0.041788, 0.033754, 0.046999, 0.042883, 0.020654, 0.016134, 0.019743, 0.013403, 0.010764, 0.011333, 0.006327, 0.007855, 0.007153, 0.004524, 0.004398, 0.004477, 0.002790, 0.001640, 0.002304, 0.001327, 0.001289, 0.001276, 0.001282, 0.001437, 0.001162, 0.001126, 0.001185, 0.000845, 0.000824, 0.000681, 0.000478, 0.000507, 0.000436, 0.000542, 0.000492, 0.000483, 0.000550, 0.000498, 0.000473, 0.000382, 0.000417],
                  [0.127855, 0.130257, 0.136622, 0.148241, 0.163440, 0.187014, 0.222448, 0.269666, 0.337724, 0.432226, 0.546829, 0.685677, 0.832264, 0.946577, 1.000000, 0.963572, 0.835437, 0.653470, 0.469405, 0.320701, 0.219671, 0.159117, 0.128159, 0.117012, 0.120187, 0.130645, 0.137970, 0.129624, 0.103790, 0.074582, 0.053827, 0.044338, 0.044053, 0.047408, 0.046578, 0.038284, 0.029418, 0.025242, 0.023617, 0.018886, 0.011937, 0.008524, 0.009274, 0.011287, 0.011157, 0.010323, 0.008138, 0.004112, 0.002289, 0.002544, 0.002919, 0.001904, 0.001524, 0.001581, 0.001191, 0.001085, 0.001013, 0.001076, 0.000924, 0.000740, 0.000834, 0.000857, 0.000908, 0.000774],
                  [0.127855, 0.130257, 0.136622, 0.148241, 0.163440, 0.187014, 0.222448, 0.269666, 0.337724, 0.432226, 0.546829, 0.685677, 0.832264, 0.946577, 1.000000, 0.963572, 0.835437, 0.653470, 0.469405, 0.320701, 0.219671, 0.159117, 0.128159, 0.117012, 0.120187, 0.130645, 0.137970, 0.129624, 0.103790, 0.074582, 0.053827, 0.044338, 0.044053, 0.047408, 0.046578, 0.038284, 0.029418, 0.025242, 0.023617, 0.018886, 0.011937, 0.008524, 0.009274, 0.011287, 0.011157, 0.010323, 0.008138, 0.004112, 0.002289, 0.002544, 0.002919, 0.001904, 0.001524, 0.001581, 0.001191, 0.001085, 0.001013, 0.001076, 0.000924, 0.000740, 0.000834, 0.000857, 0.000908, 0.000774],
                  [0.127855, 0.130257, 0.136622, 0.148241, 0.163440, 0.187014, 0.222448, 0.269666, 0.337724, 0.432226, 0.546829, 0.685677, 0.832264, 0.946577, 1.000000, 0.963572, 0.835437, 0.653470, 0.469405, 0.320701, 0.219671, 0.159117, 0.128159, 0.117012, 0.120187, 0.130645, 0.137970, 0.129624, 0.103790, 0.074582, 0.053827, 0.044338, 0.044053, 0.047408, 0.046578, 0.038284, 0.029418, 0.025242, 0.023617, 0.018886, 0.011937, 0.008524, 0.009274, 0.011287, 0.011157, 0.010323, 0.008138, 0.004112, 0.002289, 0.002544, 0.002919, 0.001904, 0.001524, 0.001581, 0.001191, 0.001085, 0.001013, 0.001076, 0.000924, 0.000740, 0.000834, 0.000857, 0.000908, 0.000774],
                  [0.127855, 0.130257, 0.136622, 0.148241, 0.163440, 0.187014, 0.222448, 0.269666, 0.337724, 0.432226, 0.546829, 0.685677, 0.832264, 0.946577, 1.000000, 0.963572, 0.835437, 0.653470, 0.469405, 0.320701, 0.219671, 0.159117, 0.128159, 0.117012, 0.120187, 0.130645, 0.137970, 0.129624, 0.103790, 0.074582, 0.053827, 0.044338, 0.044053, 0.047408, 0.046578, 0.038284, 0.029418, 0.025242, 0.023617, 0.018886, 0.011937, 0.008524, 0.009274, 0.011287, 0.011157, 0.010323, 0.008138, 0.004112, 0.002289, 0.002544, 0.002919, 0.001904, 0.001524, 0.001581, 0.001191, 0.001085, 0.001013, 0.001076, 0.000924, 0.000740, 0.000834, 0.000857, 0.000908, 0.000774],
                  [0.127855, 0.130257, 0.136622, 0.148241, 0.163440, 0.187014, 0.222448, 0.269666, 0.337724, 0.432226, 0.546829, 0.685677, 0.832264, 0.946577, 1.000000, 0.963572, 0.835437, 0.653470, 0.469405, 0.320701, 0.219671, 0.159117, 0.128159, 0.117012, 0.120187, 0.130645, 0.137970, 0.129624, 0.103790, 0.074582, 0.053827, 0.044338, 0.044053, 0.047408, 0.046578, 0.038284, 0.029418, 0.025242, 0.023617, 0.018886, 0.011937, 0.008524, 0.009274, 0.011287, 0.011157, 0.010323, 0.008138, 0.004112, 0.002289, 0.002544, 0.002919, 0.001904, 0.001524, 0.001581, 0.001191, 0.001085, 0.001013, 0.001076, 0.000924, 0.000740, 0.000834, 0.000857, 0.000908, 0.000774],
                  [0.127855, 0.130257, 0.136622, 0.148241, 0.163440, 0.187014, 0.222448, 0.269666, 0.337724, 0.432226, 0.546829, 0.685677, 0.832264, 0.946577, 1.000000, 0.963572, 0.835437, 0.653470, 0.469405, 0.320701, 0.219671, 0.159117, 0.128159, 0.117012, 0.120187, 0.130645, 0.137970, 0.129624, 0.103790, 0.074582, 0.053827, 0.044338, 0.044053, 0.047408, 0.046578, 0.038284, 0.029418, 0.025242, 0.023617, 0.018886, 0.011937, 0.008524, 0.009274, 0.011287, 0.011157, 0.010323, 0.008138, 0.004112, 0.002289, 0.002544, 0.002919, 0.001904, 0.001524, 0.001581, 0.001191, 0.001085, 0.001013, 0.001076, 0.000924, 0.000740, 0.000834, 0.000857, 0.000908, 0.000774],
                  [0.127855, 0.130257, 0.136622, 0.148241, 0.163440, 0.187014, 0.222448, 0.269666, 0.337724, 0.432226, 0.546829, 0.685677, 0.832264, 0.946577, 1.000000, 0.963572, 0.835437, 0.653470, 0.469405, 0.320701, 0.219671, 0.159117, 0.128159, 0.117012, 0.120187, 0.130645, 0.137970, 0.129624, 0.103790, 0.074582, 0.053827, 0.044338, 0.044053, 0.047408, 0.046578, 0.038284, 0.029418, 0.025242, 0.023617, 0.018886, 0.011937, 0.008524, 0.009274, 0.011287, 0.011157, 0.010323, 0.008138, 0.004112, 0.002289, 0.002544, 0.002919, 0.001904, 0.001524, 0.001581, 0.001191, 0.001085, 0.001013, 0.001076, 0.000924, 0.000740, 0.000834, 0.000857, 0.000908, 0.000774],
                  [0.127855, 0.130257, 0.136622, 0.148241, 0.163440, 0.187014, 0.222448, 0.269666, 0.337724, 0.432226, 0.546829, 0.685677, 0.832264, 0.946577, 1.000000, 0.963572, 0.835437, 0.653470, 0.469405, 0.320701, 0.219671, 0.159117, 0.128159, 0.117012, 0.120187, 0.130645, 0.137970, 0.129624, 0.103790, 0.074582, 0.053827, 0.044338, 0.044053, 0.047408, 0.046578, 0.038284, 0.029418, 0.025242, 0.023617, 0.018886, 0.011937, 0.008524, 0.009274, 0.011287, 0.011157, 0.010323, 0.008138, 0.004112, 0.002289, 0.002544, 0.002919, 0.001904, 0.001524, 0.001581, 0.001191, 0.001085, 0.001013, 0.001076, 0.000924, 0.000740, 0.000834, 0.000857, 0.000908, 0.000774],
                  [0.127855, 0.130257, 0.136622, 0.148241, 0.163440, 0.187014, 0.222448, 0.269666, 0.337724, 0.432226, 0.546829, 0.685677, 0.832264, 0.946577, 1.000000, 0.963572, 0.835437, 0.653470, 0.469405, 0.320701, 0.219671, 0.159117, 0.128159, 0.117012, 0.120187, 0.130645, 0.137970, 0.129624, 0.103790, 0.074582, 0.053827, 0.044338, 0.044053, 0.047408, 0.046578, 0.038284, 0.029418, 0.025242, 0.023617, 0.018886, 0.011937, 0.008524, 0.009274, 0.011287, 0.011157, 0.010323, 0.008138, 0.004112, 0.002289, 0.002544, 0.002919, 0.001904, 0.001524, 0.001581, 0.001191, 0.001085, 0.001013, 0.001076, 0.000924, 0.000740, 0.000834, 0.000857, 0.000908, 0.000774],
                  [0.127855, 0.130257, 0.136622, 0.148241, 0.163440, 0.187014, 0.222448, 0.269666, 0.337724, 0.432226, 0.546829, 0.685677, 0.832264, 0.946577, 1.000000, 0.963572, 0.835437, 0.653470, 0.469405, 0.320701, 0.219671, 0.159117, 0.128159, 0.117012, 0.120187, 0.130645, 0.137970, 0.129624, 0.103790, 0.074582, 0.053827, 0.044338, 0.044053, 0.047408, 0.046578, 0.038284, 0.029418, 0.025242, 0.023617, 0.018886, 0.011937, 0.008524, 0.009274, 0.011287, 0.011157, 0.010323, 0.008138, 0.004112, 0.002289, 0.002544, 0.002919, 0.001904, 0.001524, 0.001581, 0.001191, 0.001085, 0.001013, 0.001076, 0.000924, 0.000740, 0.000834, 0.000857, 0.000908, 0.000774],
                  [0.814140, 0.814872, 0.816761, 0.820052, 0.824078, 0.829802, 0.837526, 0.846615, 0.858062, 0.872000, 0.887171, 0.904567, 0.923773, 0.942704, 0.961645, 0.979062, 0.992720, 1.000000, 0.997601, 0.982492, 0.951704, 0.903890, 0.839093, 0.756490, 0.663943, 0.568238, 0.473243, 0.386137, 0.312327, 0.255888, 0.215065, 0.188170, 0.175036, 0.173206, 0.180502, 0.192896, 0.201907, 0.196781, 0.171804, 0.133294, 0.095473, 0.069530, 0.056973, 0.055004, 0.058731, 0.058858, 0.047788, 0.031549, 0.020627, 0.016669, 0.015825, 0.013084, 0.008511, 0.006260, 0.007079, 0.008119, 0.006066, 0.004087, 0.003685, 0.003310, 0.002991, 0.003056, 0.002761, 0.002788],
                  [0.814140, 0.814872, 0.816761, 0.820052, 0.824078, 0.829802, 0.837526, 0.846615, 0.858062, 0.872000, 0.887171, 0.904567, 0.923773, 0.942704, 0.961645, 0.979062, 0.992720, 1.000000, 0.997601, 0.982492, 0.951704, 0.903890, 0.839093, 0.756490, 0.663943, 0.568238, 0.473243, 0.386137, 0.312327, 0.255888, 0.215065, 0.188170, 0.175036, 0.173206, 0.180502, 0.192896, 0.201907, 0.196781, 0.171804, 0.133294, 0.095473, 0.069530, 0.056973, 0.055004, 0.058731, 0.058858, 0.047788, 0.031549, 0.020627, 0.016669, 0.015825, 0.013084, 0.008511, 0.006260, 0.007079, 0.008119, 0.006066, 0.004087, 0.003685, 0.003310, 0.002991, 0.003056, 0.002761, 0.002788],
                  [0.814140, 0.814872, 0.816761, 0.820052, 0.824078, 0.829802, 0.837526, 0.846615, 0.858062, 0.872000, 0.887171, 0.904567, 0.923773, 0.942704, 0.961645, 0.979062, 0.992720, 1.000000, 0.997601, 0.982492, 0.951704, 0.903890, 0.839093, 0.756490, 0.663943, 0.568238, 0.473243, 0.386137, 0.312327, 0.255888, 0.215065, 0.188170, 0.175036, 0.173206, 0.180502, 0.192896, 0.201907, 0.196781, 0.171804, 0.133294, 0.095473, 0.069530, 0.056973, 0.055004, 0.058731, 0.058858, 0.047788, 0.031549, 0.020627, 0.016669, 0.015825, 0.013084, 0.008511, 0.006260, 0.007079, 0.008119, 0.006066, 0.004087, 0.003685, 0.003310, 0.002991, 0.003056, 0.002761, 0.002788],
                  [0.814140, 0.814872, 0.816761, 0.820052, 0.824078, 0.829802, 0.837526, 0.846615, 0.858062, 0.872000, 0.887171, 0.904567, 0.923773, 0.942704, 0.961645, 0.979062, 0.992720, 1.000000, 0.997601, 0.982492, 0.951704, 0.903890, 0.839093, 0.756490, 0.663943, 0.568238, 0.473243, 0.386137, 0.312327, 0.255888, 0.215065, 0.188170, 0.175036, 0.173206, 0.180502, 0.192896, 0.201907, 0.196781, 0.171804, 0.133294, 0.095473, 0.069530, 0.056973, 0.055004, 0.058731, 0.058858, 0.047788, 0.031549, 0.020627, 0.016669, 0.015825, 0.013084, 0.008511, 0.006260, 0.007079, 0.008119, 0.006066, 0.004087, 0.003685, 0.003310, 0.002991, 0.003056, 0.002761, 0.002788],
                  [0.814140, 0.814872, 0.816761, 0.820052, 0.824078, 0.829802, 0.837526, 0.846615, 0.858062, 0.872000, 0.887171, 0.904567, 0.923773, 0.942704, 0.961645, 0.979062, 0.992720, 1.000000, 0.997601, 0.982492, 0.951704, 0.903890, 0.839093, 0.756490, 0.663943, 0.568238, 0.473243, 0.386137, 0.312327, 0.255888, 0.215065, 0.188170, 0.175036, 0.173206, 0.180502, 0.192896, 0.201907, 0.196781, 0.171804, 0.133294, 0.095473, 0.069530, 0.056973, 0.055004, 0.058731, 0.058858, 0.047788, 0.031549, 0.020627, 0.016669, 0.015825, 0.013084, 0.008511, 0.006260, 0.007079, 0.008119, 0.006066, 0.004087, 0.003685, 0.003310, 0.002991, 0.003056, 0.002761, 0.002788],
                  [0.814140, 0.814872, 0.816761, 0.820052, 0.824078, 0.829802, 0.837526, 0.846615, 0.858062, 0.872000, 0.887171, 0.904567, 0.923773, 0.942704, 0.961645, 0.979062, 0.992720, 1.000000, 0.997601, 0.982492, 0.951704, 0.903890, 0.839093, 0.756490, 0.663943, 0.568238, 0.473243, 0.386137, 0.312327, 0.255888, 0.215065, 0.188170, 0.175036, 0.173206, 0.180502, 0.192896, 0.201907, 0.196781, 0.171804, 0.133294, 0.095473, 0.069530, 0.056973, 0.055004, 0.058731, 0.058858, 0.047788, 0.031549, 0.020627, 0.016669, 0.015825, 0.013084, 0.008511, 0.006260, 0.007079, 0.008119, 0.006066, 0.004087, 0.003685, 0.003310, 0.002991, 0.003056, 0.002761, 0.002788],
                  [0.814140, 0.814872, 0.816761, 0.820052, 0.824078, 0.829802, 0.837526, 0.846615, 0.858062, 0.872000, 0.887171, 0.904567, 0.923773, 0.942704, 0.961645, 0.979062, 0.992720, 1.000000, 0.997601, 0.982492, 0.951704, 0.903890, 0.839093, 0.756490, 0.663943, 0.568238, 0.473243, 0.386137, 0.312327, 0.255888, 0.215065, 0.188170, 0.175036, 0.173206, 0.180502, 0.192896, 0.201907, 0.196781, 0.171804, 0.133294, 0.095473, 0.069530, 0.056973, 0.055004, 0.058731, 0.058858, 0.047788, 0.031549, 0.020627, 0.016669, 0.015825, 0.013084, 0.008511, 0.006260, 0.007079, 0.008119, 0.006066, 0.004087, 0.003685, 0.003310, 0.002991, 0.003056, 0.002761, 0.002788],
                  [0.814140, 0.814872, 0.816761, 0.820052, 0.824078, 0.829802, 0.837526, 0.846615, 0.858062, 0.872000, 0.887171, 0.904567, 0.923773, 0.942704, 0.961645, 0.979062, 0.992720, 1.000000, 0.997601, 0.982492, 0.951704, 0.903890, 0.839093, 0.756490, 0.663943, 0.568238, 0.473243, 0.386137, 0.312327, 0.255888, 0.215065, 0.188170, 0.175036, 0.173206, 0.180502, 0.192896, 0.201907, 0.196781, 0.171804, 0.133294, 0.095473, 0.069530, 0.056973, 0.055004, 0.058731, 0.058858, 0.047788, 0.031549, 0.020627, 0.016669, 0.015825, 0.013084, 0.008511, 0.006260, 0.007079, 0.008119, 0.006066, 0.004087, 0.003685, 0.003310, 0.002991, 0.003056, 0.002761, 0.002788],
                  [0.814140, 0.814872, 0.816761, 0.820052, 0.824078, 0.829802, 0.837526, 0.846615, 0.858062, 0.872000, 0.887171, 0.904567, 0.923773, 0.942704, 0.961645, 0.979062, 0.992720, 1.000000, 0.997601, 0.982492, 0.951704, 0.903890, 0.839093, 0.756490, 0.663943, 0.568238, 0.473243, 0.386137, 0.312327, 0.255888, 0.215065, 0.188170, 0.175036, 0.173206, 0.180502, 0.192896, 0.201907, 0.196781, 0.171804, 0.133294, 0.095473, 0.069530, 0.056973, 0.055004, 0.058731, 0.058858, 0.047788, 0.031549, 0.020627, 0.016669, 0.015825, 0.013084, 0.008511, 0.006260, 0.007079, 0.008119, 0.006066, 0.004087, 0.003685, 0.003310, 0.002991, 0.003056, 0.002761, 0.002788],
                  [0.814140, 0.814872, 0.816761, 0.820052, 0.824078, 0.829802, 0.837526, 0.846615, 0.858062, 0.872000, 0.887171, 0.904567, 0.923773, 0.942704, 0.961645, 0.979062, 0.992720, 1.000000, 0.997601, 0.982492, 0.951704, 0.903890, 0.839093, 0.756490, 0.663943, 0.568238, 0.473243, 0.386137, 0.312327, 0.255888, 0.215065, 0.188170, 0.175036, 0.173206, 0.180502, 0.192896, 0.201907, 0.196781, 0.171804, 0.133294, 0.095473, 0.069530, 0.056973, 0.055004, 0.058731, 0.058858, 0.047788, 0.031549, 0.020627, 0.016669, 0.015825, 0.013084, 0.008511, 0.006260, 0.007079, 0.008119, 0.006066, 0.004087, 0.003685, 0.003310, 0.002991, 0.003056, 0.002761, 0.002788],
                  [0.814140, 0.814872, 0.816761, 0.820052, 0.824078, 0.829802, 0.837526, 0.846615, 0.858062, 0.872000, 0.887171, 0.904567, 0.923773, 0.942704, 0.961645, 0.979062, 0.992720, 1.000000, 0.997601, 0.982492, 0.951704, 0.903890, 0.839093, 0.756490, 0.663943, 0.568238, 0.473243, 0.386137, 0.312327, 0.255888, 0.215065, 0.188170, 0.175036, 0.173206, 0.180502, 0.192896, 0.201907, 0.196781, 0.171804, 0.133294, 0.095473, 0.069530, 0.056973, 0.055004, 0.058731, 0.058858, 0.047788, 0.031549, 0.020627, 0.016669, 0.015825, 0.013084, 0.008511, 0.006260, 0.007079, 0.008119, 0.006066, 0.004087, 0.003685, 0.003310, 0.002991, 0.003056, 0.002761, 0.002788],
                  [0.814140, 0.814872, 0.816761, 0.820052, 0.824078, 0.829802, 0.837526, 0.846615, 0.858062, 0.872000, 0.887171, 0.904567, 0.923773, 0.942704, 0.961645, 0.979062, 0.992720, 1.000000, 0.997601, 0.982492, 0.951704, 0.903890, 0.839093, 0.756490, 0.663943, 0.568238, 0.473243, 0.386137, 0.312327, 0.255888, 0.215065, 0.188170, 0.175036, 0.173206, 0.180502, 0.192896, 0.201907, 0.196781, 0.171804, 0.133294, 0.095473, 0.069530, 0.056973, 0.055004, 0.058731, 0.058858, 0.047788, 0.031549, 0.020627, 0.016669, 0.015825, 0.013084, 0.008511, 0.006260, 0.007079, 0.008119, 0.006066, 0.004087, 0.003685, 0.003310, 0.002991, 0.003056, 0.002761, 0.002788],
                  [0.976687, 0.976748, 0.976906, 0.977181, 0.977518, 0.977998, 0.978648, 0.979418, 0.980396, 0.981601, 0.982936, 0.984507, 0.986308, 0.988185, 0.990230, 0.992385, 0.994556, 0.996615, 0.998375, 0.999608, 1.000000, 0.999196, 0.996730, 0.991856, 0.984036, 0.972571, 0.956173, 0.933744, 0.904250, 0.868057, 0.823834, 0.770319, 0.711024, 0.646093, 0.576145, 0.503987, 0.434128, 0.368189, 0.308567, 0.257165, 0.213982, 0.179896, 0.153997, 0.134877, 0.121110, 0.111444, 0.104018, 0.096647, 0.087278, 0.074573, 0.059123, 0.043485, 0.030144, 0.020646, 0.014778, 0.011553, 0.009952, 0.008985, 0.007812, 0.006108, 0.004305, 0.002982, 0.002263, 0.001981],
                  [0.976687, 0.976748, 0.976906, 0.977181, 0.977518, 0.977998, 0.978648, 0.979418, 0.980396, 0.981601, 0.982936, 0.984507, 0.986308, 0.988185, 0.990230, 0.992385, 0.994556, 0.996615, 0.998375, 0.999608, 1.000000, 0.999196, 0.996730, 0.991856, 0.984036, 0.972571, 0.956173, 0.933744, 0.904250, 0.868057, 0.823834, 0.770319, 0.711024, 0.646093, 0.576145, 0.503987, 0.434128, 0.368189, 0.308567, 0.257165, 0.213982, 0.179896, 0.153997, 0.134877, 0.121110, 0.111444, 0.104018, 0.096647, 0.087278, 0.074573, 0.059123, 0.043485, 0.030144, 0.020646, 0.014778, 0.011553, 0.009952, 0.008985, 0.007812, 0.006108, 0.004305, 0.002982, 0.002263, 0.001981],
                  [0.976687, 0.976748, 0.976906, 0.977181, 0.977518, 0.977998, 0.978648, 0.979418, 0.980396, 0.981601, 0.982936, 0.984507, 0.986308, 0.988185, 0.990230, 0.992385, 0.994556, 0.996615, 0.998375, 0.999608, 1.000000, 0.999196, 0.996730, 0.991856, 0.984036, 0.972571, 0.956173, 0.933744, 0.904250, 0.868057, 0.823834, 0.770319, 0.711024, 0.646093, 0.576145, 0.503987, 0.434128, 0.368189, 0.308567, 0.257165, 0.213982, 0.179896, 0.153997, 0.134877, 0.121110, 0.111444, 0.104018, 0.096647, 0.087278, 0.074573, 0.059123, 0.043485, 0.030144, 0.020646, 0.014778, 0.011553, 0.009952, 0.008985, 0.007812, 0.006108, 0.004305, 0.002982, 0.002263, 0.001981],
                  [0.976687, 0.976748, 0.976906, 0.977181, 0.977518, 0.977998, 0.978648, 0.979418, 0.980396, 0.981601, 0.982936, 0.984507, 0.986308, 0.988185, 0.990230, 0.992385, 0.994556, 0.996615, 0.998375, 0.999608, 1.000000, 0.999196, 0.996730, 0.991856, 0.984036, 0.972571, 0.956173, 0.933744, 0.904250, 0.868057, 0.823834, 0.770319, 0.711024, 0.646093, 0.576145, 0.503987, 0.434128, 0.368189, 0.308567, 0.257165, 0.213982, 0.179896, 0.153997, 0.134877, 0.121110, 0.111444, 0.104018, 0.096647, 0.087278, 0.074573, 0.059123, 0.043485, 0.030144, 0.020646, 0.014778, 0.011553, 0.009952, 0.008985, 0.007812, 0.006108, 0.004305, 0.002982, 0.002263, 0.001981],
                  [0.976687, 0.976748, 0.976906, 0.977181, 0.977518, 0.977998, 0.978648, 0.979418, 0.980396, 0.981601, 0.982936, 0.984507, 0.986308, 0.988185, 0.990230, 0.992385, 0.994556, 0.996615, 0.998375, 0.999608, 1.000000, 0.999196, 0.996730, 0.991856, 0.984036, 0.972571, 0.956173, 0.933744, 0.904250, 0.868057, 0.823834, 0.770319, 0.711024, 0.646093, 0.576145, 0.503987, 0.434128, 0.368189, 0.308567, 0.257165, 0.213982, 0.179896, 0.153997, 0.134877, 0.121110, 0.111444, 0.104018, 0.096647, 0.087278, 0.074573, 0.059123, 0.043485, 0.030144, 0.020646, 0.014778, 0.011553, 0.009952, 0.008985, 0.007812, 0.006108, 0.004305, 0.002982, 0.002263, 0.001981],
                  [0.976687, 0.976748, 0.976906, 0.977181, 0.977518, 0.977998, 0.978648, 0.979418, 0.980396, 0.981601, 0.982936, 0.984507, 0.986308, 0.988185, 0.990230, 0.992385, 0.994556, 0.996615, 0.998375, 0.999608, 1.000000, 0.999196, 0.996730, 0.991856, 0.984036, 0.972571, 0.956173, 0.933744, 0.904250, 0.868057, 0.823834, 0.770319, 0.711024, 0.646093, 0.576145, 0.503987, 0.434128, 0.368189, 0.308567, 0.257165, 0.213982, 0.179896, 0.153997, 0.134877, 0.121110, 0.111444, 0.104018, 0.096647, 0.087278, 0.074573, 0.059123, 0.043485, 0.030144, 0.020646, 0.014778, 0.011553, 0.009952, 0.008985, 0.007812, 0.006108, 0.004305, 0.002982, 0.002263, 0.001981]])

    N = 4096
    number_of_band = 64

    H = np.zeros(N)
    for band in range(number_of_band):
        fL = f[band]
        fH = f[band + 1]
        for k in range(fL, fH):
            H[k] = a[note_number - 23][band]

    for k in range(1, int(N / 2)):
        H[N - k] = H[k]

    h = np.real(np.fft.ifft(H, N))

    for m in range(int(N / 2)):
        tmp = h[m]
        h[m] = h[int(N / 2) + m]
        h[int(N / 2) + m] = tmp

    J = 128

    b = np.zeros(J + 1)
    w = Hanning_window(J + 1)
    offset = int(N / 2) - int(J / 2)
    for m in range(J + 1):
        b[m] = h[offset + m] * w[m]

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    # decay
    p1 = 75
    p2 = 0
    p3 = 120 / 12
    p4 = 20
    decay = (1 - 1 / (1 + np.exp(-(note_number - p1) / p3))) * p4 + p2

    # d
    p1 = 70
    p2 = 0
    p3 = 120 / 12
    p4 = 0.5
    d = (1 - 1 / (1 + np.exp(-(note_number - p1) / p3))) * p4 + p2

    num = np.power(10, -3 * T / decay)
    den = np.sqrt((1 - d) * (1 - d) + 2 * d * (1 - d) * np.cos((2 * np.pi * f0) / fs) + d * d)

    c = num / den
    if c > 1:
        c = 1

    D = int(T * fs - d)
    e = T * fs - d - int(T * fs - d)
    g = (1 - e) / (1 + e)

    s0 = np.zeros(length_of_s)
    s1 = np.zeros(length_of_s)
    s2 = np.zeros(length_of_s)

    np.random.seed(0)
    number_of_partial = int(20000 / f0)
    for i in range(number_of_partial):
        theta = (np.random.rand() * 2 - 1) * np.pi
        for n in range(D + 1 + J):
            s0[n] += np.sin(2 * np.pi * f0 * (i + 1) * n / fs + theta)

    mean_of_s0 = 0
    for n in range(J, D + 1 + J):
        mean_of_s0 += s0[n]

    mean_of_s0 /= D + 1
    for n in range(D + 1 + J):
        s0[n] -= mean_of_s0

    for n in range(D + 1 + J, length_of_s):
        # fractional delay
        s1[n] = -g * s1[n - 1] + g * s0[n - D] + s0[n - D - 1]

        # filter
        s2[n] = c * ((1 - d) * s1[n] + d * s1[n - 1])

        # feedback
        s0[n] += s2[n]

    # comb filter
    p = 0.5
    s3 = np.zeros(length_of_s)
    for n in range(length_of_s):
        t = n - T * fs * p
        m = int(t)
        delta = t - m
        if m < 0:
            s3[n] = s0[n] - (delta * s0[m + 1 + D + 1] + (1 - delta) * s0[m + D + 1])
        else:
            s3[n] = s0[n] - (delta * s0[m + 1] + (1 - delta) * s0[m])

    s4 = np.zeros(length_of_s)
    for n in range(length_of_s):
        for m in range(J + 1):
            if n - m >= 0:
                s4[n] += b[m] * s3[n - m]

    s5 = np.zeros(length_of_s)
    for n in range(length_of_s - J):
        s5[n] = s4[J + n]

    # DC cancel
    fc = 5
    Q = 1 / np.sqrt(2)
    a, b = HPF(fs, fc, Q)
    s6 = filter(a, b, s5)
    s6 /= np.max(np.abs(s6))

    # A part
    VCF_A = np.array([0])
    VCF_D = np.array([T * 5])
    VCF_S = np.array([0])
    VCF_R = np.array([T * 5])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([f0 * 32])
    VCF_depth = np.array([f0 * 512])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]
        if vcf[n] > 20000:
            vcf[n] = 20000

    sa = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                sa[n] += b[m] * s6[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                sa[n] += -a[m] * sa[n - m]

    VCA_A = np.array([T * 4])
    VCA_D = np.array([T * 20])
    VCA_S = np.array([0])
    VCA_R = np.array([T * 20])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        sa[n] *= vca[n]

    # B part
    VCF_A = np.array([0])
    VCF_D = np.array([decay * 0.2])
    VCF_S = np.array([0])
    VCF_R = np.array([decay * 0.2])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([f0 * 8])
    VCF_depth = np.array([f0 * 128])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]
        if vcf[n] > 20000:
            vcf[n] = 20000

    sb = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                sb[n] += b[m] * s6[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                sb[n] += -a[m] * sb[n - m]

    VCA_A = np.array([T * 4])
    VCA_D = np.array([0])
    VCA_S = np.array([1])
    VCA_R = np.array([0])
    VCA_gate = np.array([duration])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        sb[n] *= vca[n]

    s7 = sa * 0.5 + sb * 0.5

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
        s7[n] *= vca[n]

    s7 *= velocity / 127 / np.max(np.abs(s7))

    return s7

def acoustic_guitar(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    s0 = np.zeros(length_of_s)

    f = np.array([0, 4, 7, 12, 16, 20, 25, 30, 35, 41, 47, 53, 60, 67, 74, 82, 90, 99, 108, 118, 128, 139, 150, 162, 175, 188, 202, 217, 233, 250, 267, 286, 306, 326, 348, 371, 396, 421, 449, 477, 508, 540, 574, 609, 647, 687, 729, 773, 820, 869, 922, 977, 1035, 1097, 1161, 1230, 1302, 1379, 1460, 1545, 1635, 1730, 1830, 1936, 2048])
    a = np.array([[0.060858, 0.188249, 0.632011, 1.000000, 0.850353, 0.347937, 0.157694, 0.140827, 0.192228, 0.357717, 0.079275, 0.074854, 0.036400, 0.065630, 0.064848, 0.044823, 0.079304, 0.057037, 0.030086, 0.030348, 0.071365, 0.028947, 0.014450, 0.023485, 0.021666, 0.020902, 0.022189, 0.024348, 0.016225, 0.019122, 0.035220, 0.006662, 0.013261, 0.018244, 0.005799, 0.010833, 0.011369, 0.008764, 0.007319, 0.005925, 0.009274, 0.005687, 0.003852, 0.003488, 0.004024, 0.003406, 0.003140, 0.002878, 0.001997, 0.002213, 0.002079, 0.001446, 0.001384, 0.000878, 0.000447, 0.000510, 0.000274, 0.000247, 0.000164, 0.000228, 0.000244, 0.000227, 0.000281, 0.000261],
                  [0.060858, 0.188249, 0.632011, 1.000000, 0.850353, 0.347937, 0.157694, 0.140827, 0.192228, 0.357717, 0.079275, 0.074854, 0.036400, 0.065630, 0.064848, 0.044823, 0.079304, 0.057037, 0.030086, 0.030348, 0.071365, 0.028947, 0.014450, 0.023485, 0.021666, 0.020902, 0.022189, 0.024348, 0.016225, 0.019122, 0.035220, 0.006662, 0.013261, 0.018244, 0.005799, 0.010833, 0.011369, 0.008764, 0.007319, 0.005925, 0.009274, 0.005687, 0.003852, 0.003488, 0.004024, 0.003406, 0.003140, 0.002878, 0.001997, 0.002213, 0.002079, 0.001446, 0.001384, 0.000878, 0.000447, 0.000510, 0.000274, 0.000247, 0.000164, 0.000228, 0.000244, 0.000227, 0.000281, 0.000261],
                  [0.060858, 0.188249, 0.632011, 1.000000, 0.850353, 0.347937, 0.157694, 0.140827, 0.192228, 0.357717, 0.079275, 0.074854, 0.036400, 0.065630, 0.064848, 0.044823, 0.079304, 0.057037, 0.030086, 0.030348, 0.071365, 0.028947, 0.014450, 0.023485, 0.021666, 0.020902, 0.022189, 0.024348, 0.016225, 0.019122, 0.035220, 0.006662, 0.013261, 0.018244, 0.005799, 0.010833, 0.011369, 0.008764, 0.007319, 0.005925, 0.009274, 0.005687, 0.003852, 0.003488, 0.004024, 0.003406, 0.003140, 0.002878, 0.001997, 0.002213, 0.002079, 0.001446, 0.001384, 0.000878, 0.000447, 0.000510, 0.000274, 0.000247, 0.000164, 0.000228, 0.000244, 0.000227, 0.000281, 0.000261],
                  [0.060858, 0.188249, 0.632011, 1.000000, 0.850353, 0.347937, 0.157694, 0.140827, 0.192228, 0.357717, 0.079275, 0.074854, 0.036400, 0.065630, 0.064848, 0.044823, 0.079304, 0.057037, 0.030086, 0.030348, 0.071365, 0.028947, 0.014450, 0.023485, 0.021666, 0.020902, 0.022189, 0.024348, 0.016225, 0.019122, 0.035220, 0.006662, 0.013261, 0.018244, 0.005799, 0.010833, 0.011369, 0.008764, 0.007319, 0.005925, 0.009274, 0.005687, 0.003852, 0.003488, 0.004024, 0.003406, 0.003140, 0.002878, 0.001997, 0.002213, 0.002079, 0.001446, 0.001384, 0.000878, 0.000447, 0.000510, 0.000274, 0.000247, 0.000164, 0.000228, 0.000244, 0.000227, 0.000281, 0.000261],
                  [0.060858, 0.188249, 0.632011, 1.000000, 0.850353, 0.347937, 0.157694, 0.140827, 0.192228, 0.357717, 0.079275, 0.074854, 0.036400, 0.065630, 0.064848, 0.044823, 0.079304, 0.057037, 0.030086, 0.030348, 0.071365, 0.028947, 0.014450, 0.023485, 0.021666, 0.020902, 0.022189, 0.024348, 0.016225, 0.019122, 0.035220, 0.006662, 0.013261, 0.018244, 0.005799, 0.010833, 0.011369, 0.008764, 0.007319, 0.005925, 0.009274, 0.005687, 0.003852, 0.003488, 0.004024, 0.003406, 0.003140, 0.002878, 0.001997, 0.002213, 0.002079, 0.001446, 0.001384, 0.000878, 0.000447, 0.000510, 0.000274, 0.000247, 0.000164, 0.000228, 0.000244, 0.000227, 0.000281, 0.000261],
                  [0.060858, 0.188249, 0.632011, 1.000000, 0.850353, 0.347937, 0.157694, 0.140827, 0.192228, 0.357717, 0.079275, 0.074854, 0.036400, 0.065630, 0.064848, 0.044823, 0.079304, 0.057037, 0.030086, 0.030348, 0.071365, 0.028947, 0.014450, 0.023485, 0.021666, 0.020902, 0.022189, 0.024348, 0.016225, 0.019122, 0.035220, 0.006662, 0.013261, 0.018244, 0.005799, 0.010833, 0.011369, 0.008764, 0.007319, 0.005925, 0.009274, 0.005687, 0.003852, 0.003488, 0.004024, 0.003406, 0.003140, 0.002878, 0.001997, 0.002213, 0.002079, 0.001446, 0.001384, 0.000878, 0.000447, 0.000510, 0.000274, 0.000247, 0.000164, 0.000228, 0.000244, 0.000227, 0.000281, 0.000261],
                  [0.060858, 0.188249, 0.632011, 1.000000, 0.850353, 0.347937, 0.157694, 0.140827, 0.192228, 0.357717, 0.079275, 0.074854, 0.036400, 0.065630, 0.064848, 0.044823, 0.079304, 0.057037, 0.030086, 0.030348, 0.071365, 0.028947, 0.014450, 0.023485, 0.021666, 0.020902, 0.022189, 0.024348, 0.016225, 0.019122, 0.035220, 0.006662, 0.013261, 0.018244, 0.005799, 0.010833, 0.011369, 0.008764, 0.007319, 0.005925, 0.009274, 0.005687, 0.003852, 0.003488, 0.004024, 0.003406, 0.003140, 0.002878, 0.001997, 0.002213, 0.002079, 0.001446, 0.001384, 0.000878, 0.000447, 0.000510, 0.000274, 0.000247, 0.000164, 0.000228, 0.000244, 0.000227, 0.000281, 0.000261],
                  [0.060858, 0.188249, 0.632011, 1.000000, 0.850353, 0.347937, 0.157694, 0.140827, 0.192228, 0.357717, 0.079275, 0.074854, 0.036400, 0.065630, 0.064848, 0.044823, 0.079304, 0.057037, 0.030086, 0.030348, 0.071365, 0.028947, 0.014450, 0.023485, 0.021666, 0.020902, 0.022189, 0.024348, 0.016225, 0.019122, 0.035220, 0.006662, 0.013261, 0.018244, 0.005799, 0.010833, 0.011369, 0.008764, 0.007319, 0.005925, 0.009274, 0.005687, 0.003852, 0.003488, 0.004024, 0.003406, 0.003140, 0.002878, 0.001997, 0.002213, 0.002079, 0.001446, 0.001384, 0.000878, 0.000447, 0.000510, 0.000274, 0.000247, 0.000164, 0.000228, 0.000244, 0.000227, 0.000281, 0.000261],
                  [0.046745, 0.094280, 0.352658, 0.938142, 1.000000, 0.490902, 0.193114, 0.153733, 0.205178, 0.168552, 0.066142, 0.033446, 0.041979, 0.041991, 0.029496, 0.050632, 0.062910, 0.018927, 0.016269, 0.020079, 0.009328, 0.015164, 0.015082, 0.020074, 0.013432, 0.007060, 0.005356, 0.004610, 0.005968, 0.009002, 0.006982, 0.008441, 0.008451, 0.006159, 0.005910, 0.004807, 0.004307, 0.004175, 0.004417, 0.002765, 0.003162, 0.002115, 0.001149, 0.001410, 0.000699, 0.001263, 0.000538, 0.001154, 0.000633, 0.000817, 0.000345, 0.000479, 0.000500, 0.000272, 0.000229, 0.000180, 0.000124, 0.000082, 0.000069, 0.000066, 0.000041, 0.000039, 0.000038, 0.000032],
                  [0.046745, 0.094280, 0.352658, 0.938142, 1.000000, 0.490902, 0.193114, 0.153733, 0.205178, 0.168552, 0.066142, 0.033446, 0.041979, 0.041991, 0.029496, 0.050632, 0.062910, 0.018927, 0.016269, 0.020079, 0.009328, 0.015164, 0.015082, 0.020074, 0.013432, 0.007060, 0.005356, 0.004610, 0.005968, 0.009002, 0.006982, 0.008441, 0.008451, 0.006159, 0.005910, 0.004807, 0.004307, 0.004175, 0.004417, 0.002765, 0.003162, 0.002115, 0.001149, 0.001410, 0.000699, 0.001263, 0.000538, 0.001154, 0.000633, 0.000817, 0.000345, 0.000479, 0.000500, 0.000272, 0.000229, 0.000180, 0.000124, 0.000082, 0.000069, 0.000066, 0.000041, 0.000039, 0.000038, 0.000032],
                  [0.046745, 0.094280, 0.352658, 0.938142, 1.000000, 0.490902, 0.193114, 0.153733, 0.205178, 0.168552, 0.066142, 0.033446, 0.041979, 0.041991, 0.029496, 0.050632, 0.062910, 0.018927, 0.016269, 0.020079, 0.009328, 0.015164, 0.015082, 0.020074, 0.013432, 0.007060, 0.005356, 0.004610, 0.005968, 0.009002, 0.006982, 0.008441, 0.008451, 0.006159, 0.005910, 0.004807, 0.004307, 0.004175, 0.004417, 0.002765, 0.003162, 0.002115, 0.001149, 0.001410, 0.000699, 0.001263, 0.000538, 0.001154, 0.000633, 0.000817, 0.000345, 0.000479, 0.000500, 0.000272, 0.000229, 0.000180, 0.000124, 0.000082, 0.000069, 0.000066, 0.000041, 0.000039, 0.000038, 0.000032],
                  [0.046745, 0.094280, 0.352658, 0.938142, 1.000000, 0.490902, 0.193114, 0.153733, 0.205178, 0.168552, 0.066142, 0.033446, 0.041979, 0.041991, 0.029496, 0.050632, 0.062910, 0.018927, 0.016269, 0.020079, 0.009328, 0.015164, 0.015082, 0.020074, 0.013432, 0.007060, 0.005356, 0.004610, 0.005968, 0.009002, 0.006982, 0.008441, 0.008451, 0.006159, 0.005910, 0.004807, 0.004307, 0.004175, 0.004417, 0.002765, 0.003162, 0.002115, 0.001149, 0.001410, 0.000699, 0.001263, 0.000538, 0.001154, 0.000633, 0.000817, 0.000345, 0.000479, 0.000500, 0.000272, 0.000229, 0.000180, 0.000124, 0.000082, 0.000069, 0.000066, 0.000041, 0.000039, 0.000038, 0.000032],
                  [0.046745, 0.094280, 0.352658, 0.938142, 1.000000, 0.490902, 0.193114, 0.153733, 0.205178, 0.168552, 0.066142, 0.033446, 0.041979, 0.041991, 0.029496, 0.050632, 0.062910, 0.018927, 0.016269, 0.020079, 0.009328, 0.015164, 0.015082, 0.020074, 0.013432, 0.007060, 0.005356, 0.004610, 0.005968, 0.009002, 0.006982, 0.008441, 0.008451, 0.006159, 0.005910, 0.004807, 0.004307, 0.004175, 0.004417, 0.002765, 0.003162, 0.002115, 0.001149, 0.001410, 0.000699, 0.001263, 0.000538, 0.001154, 0.000633, 0.000817, 0.000345, 0.000479, 0.000500, 0.000272, 0.000229, 0.000180, 0.000124, 0.000082, 0.000069, 0.000066, 0.000041, 0.000039, 0.000038, 0.000032],
                  [0.046745, 0.094280, 0.352658, 0.938142, 1.000000, 0.490902, 0.193114, 0.153733, 0.205178, 0.168552, 0.066142, 0.033446, 0.041979, 0.041991, 0.029496, 0.050632, 0.062910, 0.018927, 0.016269, 0.020079, 0.009328, 0.015164, 0.015082, 0.020074, 0.013432, 0.007060, 0.005356, 0.004610, 0.005968, 0.009002, 0.006982, 0.008441, 0.008451, 0.006159, 0.005910, 0.004807, 0.004307, 0.004175, 0.004417, 0.002765, 0.003162, 0.002115, 0.001149, 0.001410, 0.000699, 0.001263, 0.000538, 0.001154, 0.000633, 0.000817, 0.000345, 0.000479, 0.000500, 0.000272, 0.000229, 0.000180, 0.000124, 0.000082, 0.000069, 0.000066, 0.000041, 0.000039, 0.000038, 0.000032],
                  [0.046745, 0.094280, 0.352658, 0.938142, 1.000000, 0.490902, 0.193114, 0.153733, 0.205178, 0.168552, 0.066142, 0.033446, 0.041979, 0.041991, 0.029496, 0.050632, 0.062910, 0.018927, 0.016269, 0.020079, 0.009328, 0.015164, 0.015082, 0.020074, 0.013432, 0.007060, 0.005356, 0.004610, 0.005968, 0.009002, 0.006982, 0.008441, 0.008451, 0.006159, 0.005910, 0.004807, 0.004307, 0.004175, 0.004417, 0.002765, 0.003162, 0.002115, 0.001149, 0.001410, 0.000699, 0.001263, 0.000538, 0.001154, 0.000633, 0.000817, 0.000345, 0.000479, 0.000500, 0.000272, 0.000229, 0.000180, 0.000124, 0.000082, 0.000069, 0.000066, 0.000041, 0.000039, 0.000038, 0.000032],
                  [0.046745, 0.094280, 0.352658, 0.938142, 1.000000, 0.490902, 0.193114, 0.153733, 0.205178, 0.168552, 0.066142, 0.033446, 0.041979, 0.041991, 0.029496, 0.050632, 0.062910, 0.018927, 0.016269, 0.020079, 0.009328, 0.015164, 0.015082, 0.020074, 0.013432, 0.007060, 0.005356, 0.004610, 0.005968, 0.009002, 0.006982, 0.008441, 0.008451, 0.006159, 0.005910, 0.004807, 0.004307, 0.004175, 0.004417, 0.002765, 0.003162, 0.002115, 0.001149, 0.001410, 0.000699, 0.001263, 0.000538, 0.001154, 0.000633, 0.000817, 0.000345, 0.000479, 0.000500, 0.000272, 0.000229, 0.000180, 0.000124, 0.000082, 0.000069, 0.000066, 0.000041, 0.000039, 0.000038, 0.000032],
                  [0.046745, 0.094280, 0.352658, 0.938142, 1.000000, 0.490902, 0.193114, 0.153733, 0.205178, 0.168552, 0.066142, 0.033446, 0.041979, 0.041991, 0.029496, 0.050632, 0.062910, 0.018927, 0.016269, 0.020079, 0.009328, 0.015164, 0.015082, 0.020074, 0.013432, 0.007060, 0.005356, 0.004610, 0.005968, 0.009002, 0.006982, 0.008441, 0.008451, 0.006159, 0.005910, 0.004807, 0.004307, 0.004175, 0.004417, 0.002765, 0.003162, 0.002115, 0.001149, 0.001410, 0.000699, 0.001263, 0.000538, 0.001154, 0.000633, 0.000817, 0.000345, 0.000479, 0.000500, 0.000272, 0.000229, 0.000180, 0.000124, 0.000082, 0.000069, 0.000066, 0.000041, 0.000039, 0.000038, 0.000032],
                  [0.046745, 0.094280, 0.352658, 0.938142, 1.000000, 0.490902, 0.193114, 0.153733, 0.205178, 0.168552, 0.066142, 0.033446, 0.041979, 0.041991, 0.029496, 0.050632, 0.062910, 0.018927, 0.016269, 0.020079, 0.009328, 0.015164, 0.015082, 0.020074, 0.013432, 0.007060, 0.005356, 0.004610, 0.005968, 0.009002, 0.006982, 0.008441, 0.008451, 0.006159, 0.005910, 0.004807, 0.004307, 0.004175, 0.004417, 0.002765, 0.003162, 0.002115, 0.001149, 0.001410, 0.000699, 0.001263, 0.000538, 0.001154, 0.000633, 0.000817, 0.000345, 0.000479, 0.000500, 0.000272, 0.000229, 0.000180, 0.000124, 0.000082, 0.000069, 0.000066, 0.000041, 0.000039, 0.000038, 0.000032],
                  [0.046745, 0.094280, 0.352658, 0.938142, 1.000000, 0.490902, 0.193114, 0.153733, 0.205178, 0.168552, 0.066142, 0.033446, 0.041979, 0.041991, 0.029496, 0.050632, 0.062910, 0.018927, 0.016269, 0.020079, 0.009328, 0.015164, 0.015082, 0.020074, 0.013432, 0.007060, 0.005356, 0.004610, 0.005968, 0.009002, 0.006982, 0.008441, 0.008451, 0.006159, 0.005910, 0.004807, 0.004307, 0.004175, 0.004417, 0.002765, 0.003162, 0.002115, 0.001149, 0.001410, 0.000699, 0.001263, 0.000538, 0.001154, 0.000633, 0.000817, 0.000345, 0.000479, 0.000500, 0.000272, 0.000229, 0.000180, 0.000124, 0.000082, 0.000069, 0.000066, 0.000041, 0.000039, 0.000038, 0.000032],
                  [0.046745, 0.094280, 0.352658, 0.938142, 1.000000, 0.490902, 0.193114, 0.153733, 0.205178, 0.168552, 0.066142, 0.033446, 0.041979, 0.041991, 0.029496, 0.050632, 0.062910, 0.018927, 0.016269, 0.020079, 0.009328, 0.015164, 0.015082, 0.020074, 0.013432, 0.007060, 0.005356, 0.004610, 0.005968, 0.009002, 0.006982, 0.008441, 0.008451, 0.006159, 0.005910, 0.004807, 0.004307, 0.004175, 0.004417, 0.002765, 0.003162, 0.002115, 0.001149, 0.001410, 0.000699, 0.001263, 0.000538, 0.001154, 0.000633, 0.000817, 0.000345, 0.000479, 0.000500, 0.000272, 0.000229, 0.000180, 0.000124, 0.000082, 0.000069, 0.000066, 0.000041, 0.000039, 0.000038, 0.000032],
                  [0.403562, 0.425674, 0.484043, 0.587428, 0.709932, 0.857886, 0.981276, 1.000000, 0.879080, 0.656411, 0.442764, 0.282952, 0.186616, 0.140537, 0.120961, 0.116027, 0.116218, 0.111837, 0.097189, 0.076798, 0.060819, 0.054040, 0.051387, 0.041287, 0.026989, 0.022066, 0.028222, 0.030414, 0.023646, 0.026973, 0.023235, 0.012258, 0.020054, 0.019121, 0.019388, 0.023615, 0.008311, 0.023909, 0.011396, 0.005976, 0.004115, 0.004502, 0.004226, 0.004772, 0.003377, 0.001722, 0.001412, 0.001079, 0.001299, 0.002012, 0.002120, 0.001042, 0.001323, 0.001474, 0.000733, 0.000376, 0.000353, 0.000283, 0.000262, 0.000250, 0.000183, 0.000209, 0.000165, 0.000161],
                  [0.403562, 0.425674, 0.484043, 0.587428, 0.709932, 0.857886, 0.981276, 1.000000, 0.879080, 0.656411, 0.442764, 0.282952, 0.186616, 0.140537, 0.120961, 0.116027, 0.116218, 0.111837, 0.097189, 0.076798, 0.060819, 0.054040, 0.051387, 0.041287, 0.026989, 0.022066, 0.028222, 0.030414, 0.023646, 0.026973, 0.023235, 0.012258, 0.020054, 0.019121, 0.019388, 0.023615, 0.008311, 0.023909, 0.011396, 0.005976, 0.004115, 0.004502, 0.004226, 0.004772, 0.003377, 0.001722, 0.001412, 0.001079, 0.001299, 0.002012, 0.002120, 0.001042, 0.001323, 0.001474, 0.000733, 0.000376, 0.000353, 0.000283, 0.000262, 0.000250, 0.000183, 0.000209, 0.000165, 0.000161],
                  [0.403562, 0.425674, 0.484043, 0.587428, 0.709932, 0.857886, 0.981276, 1.000000, 0.879080, 0.656411, 0.442764, 0.282952, 0.186616, 0.140537, 0.120961, 0.116027, 0.116218, 0.111837, 0.097189, 0.076798, 0.060819, 0.054040, 0.051387, 0.041287, 0.026989, 0.022066, 0.028222, 0.030414, 0.023646, 0.026973, 0.023235, 0.012258, 0.020054, 0.019121, 0.019388, 0.023615, 0.008311, 0.023909, 0.011396, 0.005976, 0.004115, 0.004502, 0.004226, 0.004772, 0.003377, 0.001722, 0.001412, 0.001079, 0.001299, 0.002012, 0.002120, 0.001042, 0.001323, 0.001474, 0.000733, 0.000376, 0.000353, 0.000283, 0.000262, 0.000250, 0.000183, 0.000209, 0.000165, 0.000161],
                  [0.403562, 0.425674, 0.484043, 0.587428, 0.709932, 0.857886, 0.981276, 1.000000, 0.879080, 0.656411, 0.442764, 0.282952, 0.186616, 0.140537, 0.120961, 0.116027, 0.116218, 0.111837, 0.097189, 0.076798, 0.060819, 0.054040, 0.051387, 0.041287, 0.026989, 0.022066, 0.028222, 0.030414, 0.023646, 0.026973, 0.023235, 0.012258, 0.020054, 0.019121, 0.019388, 0.023615, 0.008311, 0.023909, 0.011396, 0.005976, 0.004115, 0.004502, 0.004226, 0.004772, 0.003377, 0.001722, 0.001412, 0.001079, 0.001299, 0.002012, 0.002120, 0.001042, 0.001323, 0.001474, 0.000733, 0.000376, 0.000353, 0.000283, 0.000262, 0.000250, 0.000183, 0.000209, 0.000165, 0.000161],
                  [0.403562, 0.425674, 0.484043, 0.587428, 0.709932, 0.857886, 0.981276, 1.000000, 0.879080, 0.656411, 0.442764, 0.282952, 0.186616, 0.140537, 0.120961, 0.116027, 0.116218, 0.111837, 0.097189, 0.076798, 0.060819, 0.054040, 0.051387, 0.041287, 0.026989, 0.022066, 0.028222, 0.030414, 0.023646, 0.026973, 0.023235, 0.012258, 0.020054, 0.019121, 0.019388, 0.023615, 0.008311, 0.023909, 0.011396, 0.005976, 0.004115, 0.004502, 0.004226, 0.004772, 0.003377, 0.001722, 0.001412, 0.001079, 0.001299, 0.002012, 0.002120, 0.001042, 0.001323, 0.001474, 0.000733, 0.000376, 0.000353, 0.000283, 0.000262, 0.000250, 0.000183, 0.000209, 0.000165, 0.000161],
                  [0.403562, 0.425674, 0.484043, 0.587428, 0.709932, 0.857886, 0.981276, 1.000000, 0.879080, 0.656411, 0.442764, 0.282952, 0.186616, 0.140537, 0.120961, 0.116027, 0.116218, 0.111837, 0.097189, 0.076798, 0.060819, 0.054040, 0.051387, 0.041287, 0.026989, 0.022066, 0.028222, 0.030414, 0.023646, 0.026973, 0.023235, 0.012258, 0.020054, 0.019121, 0.019388, 0.023615, 0.008311, 0.023909, 0.011396, 0.005976, 0.004115, 0.004502, 0.004226, 0.004772, 0.003377, 0.001722, 0.001412, 0.001079, 0.001299, 0.002012, 0.002120, 0.001042, 0.001323, 0.001474, 0.000733, 0.000376, 0.000353, 0.000283, 0.000262, 0.000250, 0.000183, 0.000209, 0.000165, 0.000161],
                  [0.403562, 0.425674, 0.484043, 0.587428, 0.709932, 0.857886, 0.981276, 1.000000, 0.879080, 0.656411, 0.442764, 0.282952, 0.186616, 0.140537, 0.120961, 0.116027, 0.116218, 0.111837, 0.097189, 0.076798, 0.060819, 0.054040, 0.051387, 0.041287, 0.026989, 0.022066, 0.028222, 0.030414, 0.023646, 0.026973, 0.023235, 0.012258, 0.020054, 0.019121, 0.019388, 0.023615, 0.008311, 0.023909, 0.011396, 0.005976, 0.004115, 0.004502, 0.004226, 0.004772, 0.003377, 0.001722, 0.001412, 0.001079, 0.001299, 0.002012, 0.002120, 0.001042, 0.001323, 0.001474, 0.000733, 0.000376, 0.000353, 0.000283, 0.000262, 0.000250, 0.000183, 0.000209, 0.000165, 0.000161],
                  [0.403562, 0.425674, 0.484043, 0.587428, 0.709932, 0.857886, 0.981276, 1.000000, 0.879080, 0.656411, 0.442764, 0.282952, 0.186616, 0.140537, 0.120961, 0.116027, 0.116218, 0.111837, 0.097189, 0.076798, 0.060819, 0.054040, 0.051387, 0.041287, 0.026989, 0.022066, 0.028222, 0.030414, 0.023646, 0.026973, 0.023235, 0.012258, 0.020054, 0.019121, 0.019388, 0.023615, 0.008311, 0.023909, 0.011396, 0.005976, 0.004115, 0.004502, 0.004226, 0.004772, 0.003377, 0.001722, 0.001412, 0.001079, 0.001299, 0.002012, 0.002120, 0.001042, 0.001323, 0.001474, 0.000733, 0.000376, 0.000353, 0.000283, 0.000262, 0.000250, 0.000183, 0.000209, 0.000165, 0.000161],
                  [0.403562, 0.425674, 0.484043, 0.587428, 0.709932, 0.857886, 0.981276, 1.000000, 0.879080, 0.656411, 0.442764, 0.282952, 0.186616, 0.140537, 0.120961, 0.116027, 0.116218, 0.111837, 0.097189, 0.076798, 0.060819, 0.054040, 0.051387, 0.041287, 0.026989, 0.022066, 0.028222, 0.030414, 0.023646, 0.026973, 0.023235, 0.012258, 0.020054, 0.019121, 0.019388, 0.023615, 0.008311, 0.023909, 0.011396, 0.005976, 0.004115, 0.004502, 0.004226, 0.004772, 0.003377, 0.001722, 0.001412, 0.001079, 0.001299, 0.002012, 0.002120, 0.001042, 0.001323, 0.001474, 0.000733, 0.000376, 0.000353, 0.000283, 0.000262, 0.000250, 0.000183, 0.000209, 0.000165, 0.000161],
                  [0.403562, 0.425674, 0.484043, 0.587428, 0.709932, 0.857886, 0.981276, 1.000000, 0.879080, 0.656411, 0.442764, 0.282952, 0.186616, 0.140537, 0.120961, 0.116027, 0.116218, 0.111837, 0.097189, 0.076798, 0.060819, 0.054040, 0.051387, 0.041287, 0.026989, 0.022066, 0.028222, 0.030414, 0.023646, 0.026973, 0.023235, 0.012258, 0.020054, 0.019121, 0.019388, 0.023615, 0.008311, 0.023909, 0.011396, 0.005976, 0.004115, 0.004502, 0.004226, 0.004772, 0.003377, 0.001722, 0.001412, 0.001079, 0.001299, 0.002012, 0.002120, 0.001042, 0.001323, 0.001474, 0.000733, 0.000376, 0.000353, 0.000283, 0.000262, 0.000250, 0.000183, 0.000209, 0.000165, 0.000161],
                  [0.403562, 0.425674, 0.484043, 0.587428, 0.709932, 0.857886, 0.981276, 1.000000, 0.879080, 0.656411, 0.442764, 0.282952, 0.186616, 0.140537, 0.120961, 0.116027, 0.116218, 0.111837, 0.097189, 0.076798, 0.060819, 0.054040, 0.051387, 0.041287, 0.026989, 0.022066, 0.028222, 0.030414, 0.023646, 0.026973, 0.023235, 0.012258, 0.020054, 0.019121, 0.019388, 0.023615, 0.008311, 0.023909, 0.011396, 0.005976, 0.004115, 0.004502, 0.004226, 0.004772, 0.003377, 0.001722, 0.001412, 0.001079, 0.001299, 0.002012, 0.002120, 0.001042, 0.001323, 0.001474, 0.000733, 0.000376, 0.000353, 0.000283, 0.000262, 0.000250, 0.000183, 0.000209, 0.000165, 0.000161],
                  [0.403562, 0.425674, 0.484043, 0.587428, 0.709932, 0.857886, 0.981276, 1.000000, 0.879080, 0.656411, 0.442764, 0.282952, 0.186616, 0.140537, 0.120961, 0.116027, 0.116218, 0.111837, 0.097189, 0.076798, 0.060819, 0.054040, 0.051387, 0.041287, 0.026989, 0.022066, 0.028222, 0.030414, 0.023646, 0.026973, 0.023235, 0.012258, 0.020054, 0.019121, 0.019388, 0.023615, 0.008311, 0.023909, 0.011396, 0.005976, 0.004115, 0.004502, 0.004226, 0.004772, 0.003377, 0.001722, 0.001412, 0.001079, 0.001299, 0.002012, 0.002120, 0.001042, 0.001323, 0.001474, 0.000733, 0.000376, 0.000353, 0.000283, 0.000262, 0.000250, 0.000183, 0.000209, 0.000165, 0.000161],
                  [1.000000, 0.996959, 0.989121, 0.975469, 0.958787, 0.935097, 0.903193, 0.865749, 0.818770, 0.761849, 0.700278, 0.630164, 0.553219, 0.477443, 0.400557, 0.326152, 0.258156, 0.199401, 0.151907, 0.115702, 0.090016, 0.072997, 0.062960, 0.058075, 0.057100, 0.057947, 0.057888, 0.054259, 0.046800, 0.038650, 0.032793, 0.030329, 0.030185, 0.029453, 0.026393, 0.022703, 0.020332, 0.017698, 0.013279, 0.009885, 0.009485, 0.009819, 0.008647, 0.008129, 0.007524, 0.006741, 0.008184, 0.005889, 0.004463, 0.006105, 0.005055, 0.003975, 0.005074, 0.004375, 0.004186, 0.003510, 0.002665, 0.002176, 0.001816, 0.000990, 0.000793, 0.000510, 0.000471, 0.000382],
                  [1.000000, 0.996959, 0.989121, 0.975469, 0.958787, 0.935097, 0.903193, 0.865749, 0.818770, 0.761849, 0.700278, 0.630164, 0.553219, 0.477443, 0.400557, 0.326152, 0.258156, 0.199401, 0.151907, 0.115702, 0.090016, 0.072997, 0.062960, 0.058075, 0.057100, 0.057947, 0.057888, 0.054259, 0.046800, 0.038650, 0.032793, 0.030329, 0.030185, 0.029453, 0.026393, 0.022703, 0.020332, 0.017698, 0.013279, 0.009885, 0.009485, 0.009819, 0.008647, 0.008129, 0.007524, 0.006741, 0.008184, 0.005889, 0.004463, 0.006105, 0.005055, 0.003975, 0.005074, 0.004375, 0.004186, 0.003510, 0.002665, 0.002176, 0.001816, 0.000990, 0.000793, 0.000510, 0.000471, 0.000382],
                  [1.000000, 0.996959, 0.989121, 0.975469, 0.958787, 0.935097, 0.903193, 0.865749, 0.818770, 0.761849, 0.700278, 0.630164, 0.553219, 0.477443, 0.400557, 0.326152, 0.258156, 0.199401, 0.151907, 0.115702, 0.090016, 0.072997, 0.062960, 0.058075, 0.057100, 0.057947, 0.057888, 0.054259, 0.046800, 0.038650, 0.032793, 0.030329, 0.030185, 0.029453, 0.026393, 0.022703, 0.020332, 0.017698, 0.013279, 0.009885, 0.009485, 0.009819, 0.008647, 0.008129, 0.007524, 0.006741, 0.008184, 0.005889, 0.004463, 0.006105, 0.005055, 0.003975, 0.005074, 0.004375, 0.004186, 0.003510, 0.002665, 0.002176, 0.001816, 0.000990, 0.000793, 0.000510, 0.000471, 0.000382],
                  [1.000000, 0.996959, 0.989121, 0.975469, 0.958787, 0.935097, 0.903193, 0.865749, 0.818770, 0.761849, 0.700278, 0.630164, 0.553219, 0.477443, 0.400557, 0.326152, 0.258156, 0.199401, 0.151907, 0.115702, 0.090016, 0.072997, 0.062960, 0.058075, 0.057100, 0.057947, 0.057888, 0.054259, 0.046800, 0.038650, 0.032793, 0.030329, 0.030185, 0.029453, 0.026393, 0.022703, 0.020332, 0.017698, 0.013279, 0.009885, 0.009485, 0.009819, 0.008647, 0.008129, 0.007524, 0.006741, 0.008184, 0.005889, 0.004463, 0.006105, 0.005055, 0.003975, 0.005074, 0.004375, 0.004186, 0.003510, 0.002665, 0.002176, 0.001816, 0.000990, 0.000793, 0.000510, 0.000471, 0.000382],
                  [1.000000, 0.996959, 0.989121, 0.975469, 0.958787, 0.935097, 0.903193, 0.865749, 0.818770, 0.761849, 0.700278, 0.630164, 0.553219, 0.477443, 0.400557, 0.326152, 0.258156, 0.199401, 0.151907, 0.115702, 0.090016, 0.072997, 0.062960, 0.058075, 0.057100, 0.057947, 0.057888, 0.054259, 0.046800, 0.038650, 0.032793, 0.030329, 0.030185, 0.029453, 0.026393, 0.022703, 0.020332, 0.017698, 0.013279, 0.009885, 0.009485, 0.009819, 0.008647, 0.008129, 0.007524, 0.006741, 0.008184, 0.005889, 0.004463, 0.006105, 0.005055, 0.003975, 0.005074, 0.004375, 0.004186, 0.003510, 0.002665, 0.002176, 0.001816, 0.000990, 0.000793, 0.000510, 0.000471, 0.000382],
                  [1.000000, 0.996959, 0.989121, 0.975469, 0.958787, 0.935097, 0.903193, 0.865749, 0.818770, 0.761849, 0.700278, 0.630164, 0.553219, 0.477443, 0.400557, 0.326152, 0.258156, 0.199401, 0.151907, 0.115702, 0.090016, 0.072997, 0.062960, 0.058075, 0.057100, 0.057947, 0.057888, 0.054259, 0.046800, 0.038650, 0.032793, 0.030329, 0.030185, 0.029453, 0.026393, 0.022703, 0.020332, 0.017698, 0.013279, 0.009885, 0.009485, 0.009819, 0.008647, 0.008129, 0.007524, 0.006741, 0.008184, 0.005889, 0.004463, 0.006105, 0.005055, 0.003975, 0.005074, 0.004375, 0.004186, 0.003510, 0.002665, 0.002176, 0.001816, 0.000990, 0.000793, 0.000510, 0.000471, 0.000382],
                  [1.000000, 0.996959, 0.989121, 0.975469, 0.958787, 0.935097, 0.903193, 0.865749, 0.818770, 0.761849, 0.700278, 0.630164, 0.553219, 0.477443, 0.400557, 0.326152, 0.258156, 0.199401, 0.151907, 0.115702, 0.090016, 0.072997, 0.062960, 0.058075, 0.057100, 0.057947, 0.057888, 0.054259, 0.046800, 0.038650, 0.032793, 0.030329, 0.030185, 0.029453, 0.026393, 0.022703, 0.020332, 0.017698, 0.013279, 0.009885, 0.009485, 0.009819, 0.008647, 0.008129, 0.007524, 0.006741, 0.008184, 0.005889, 0.004463, 0.006105, 0.005055, 0.003975, 0.005074, 0.004375, 0.004186, 0.003510, 0.002665, 0.002176, 0.001816, 0.000990, 0.000793, 0.000510, 0.000471, 0.000382],
                  [1.000000, 0.996959, 0.989121, 0.975469, 0.958787, 0.935097, 0.903193, 0.865749, 0.818770, 0.761849, 0.700278, 0.630164, 0.553219, 0.477443, 0.400557, 0.326152, 0.258156, 0.199401, 0.151907, 0.115702, 0.090016, 0.072997, 0.062960, 0.058075, 0.057100, 0.057947, 0.057888, 0.054259, 0.046800, 0.038650, 0.032793, 0.030329, 0.030185, 0.029453, 0.026393, 0.022703, 0.020332, 0.017698, 0.013279, 0.009885, 0.009485, 0.009819, 0.008647, 0.008129, 0.007524, 0.006741, 0.008184, 0.005889, 0.004463, 0.006105, 0.005055, 0.003975, 0.005074, 0.004375, 0.004186, 0.003510, 0.002665, 0.002176, 0.001816, 0.000990, 0.000793, 0.000510, 0.000471, 0.000382],
                  [1.000000, 0.996959, 0.989121, 0.975469, 0.958787, 0.935097, 0.903193, 0.865749, 0.818770, 0.761849, 0.700278, 0.630164, 0.553219, 0.477443, 0.400557, 0.326152, 0.258156, 0.199401, 0.151907, 0.115702, 0.090016, 0.072997, 0.062960, 0.058075, 0.057100, 0.057947, 0.057888, 0.054259, 0.046800, 0.038650, 0.032793, 0.030329, 0.030185, 0.029453, 0.026393, 0.022703, 0.020332, 0.017698, 0.013279, 0.009885, 0.009485, 0.009819, 0.008647, 0.008129, 0.007524, 0.006741, 0.008184, 0.005889, 0.004463, 0.006105, 0.005055, 0.003975, 0.005074, 0.004375, 0.004186, 0.003510, 0.002665, 0.002176, 0.001816, 0.000990, 0.000793, 0.000510, 0.000471, 0.000382],
                  [1.000000, 0.996959, 0.989121, 0.975469, 0.958787, 0.935097, 0.903193, 0.865749, 0.818770, 0.761849, 0.700278, 0.630164, 0.553219, 0.477443, 0.400557, 0.326152, 0.258156, 0.199401, 0.151907, 0.115702, 0.090016, 0.072997, 0.062960, 0.058075, 0.057100, 0.057947, 0.057888, 0.054259, 0.046800, 0.038650, 0.032793, 0.030329, 0.030185, 0.029453, 0.026393, 0.022703, 0.020332, 0.017698, 0.013279, 0.009885, 0.009485, 0.009819, 0.008647, 0.008129, 0.007524, 0.006741, 0.008184, 0.005889, 0.004463, 0.006105, 0.005055, 0.003975, 0.005074, 0.004375, 0.004186, 0.003510, 0.002665, 0.002176, 0.001816, 0.000990, 0.000793, 0.000510, 0.000471, 0.000382],
                  [1.000000, 0.996959, 0.989121, 0.975469, 0.958787, 0.935097, 0.903193, 0.865749, 0.818770, 0.761849, 0.700278, 0.630164, 0.553219, 0.477443, 0.400557, 0.326152, 0.258156, 0.199401, 0.151907, 0.115702, 0.090016, 0.072997, 0.062960, 0.058075, 0.057100, 0.057947, 0.057888, 0.054259, 0.046800, 0.038650, 0.032793, 0.030329, 0.030185, 0.029453, 0.026393, 0.022703, 0.020332, 0.017698, 0.013279, 0.009885, 0.009485, 0.009819, 0.008647, 0.008129, 0.007524, 0.006741, 0.008184, 0.005889, 0.004463, 0.006105, 0.005055, 0.003975, 0.005074, 0.004375, 0.004186, 0.003510, 0.002665, 0.002176, 0.001816, 0.000990, 0.000793, 0.000510, 0.000471, 0.000382],
                  [1.000000, 0.996959, 0.989121, 0.975469, 0.958787, 0.935097, 0.903193, 0.865749, 0.818770, 0.761849, 0.700278, 0.630164, 0.553219, 0.477443, 0.400557, 0.326152, 0.258156, 0.199401, 0.151907, 0.115702, 0.090016, 0.072997, 0.062960, 0.058075, 0.057100, 0.057947, 0.057888, 0.054259, 0.046800, 0.038650, 0.032793, 0.030329, 0.030185, 0.029453, 0.026393, 0.022703, 0.020332, 0.017698, 0.013279, 0.009885, 0.009485, 0.009819, 0.008647, 0.008129, 0.007524, 0.006741, 0.008184, 0.005889, 0.004463, 0.006105, 0.005055, 0.003975, 0.005074, 0.004375, 0.004186, 0.003510, 0.002665, 0.002176, 0.001816, 0.000990, 0.000793, 0.000510, 0.000471, 0.000382],
                  [1.000000, 0.995030, 0.982342, 0.960597, 0.934663, 0.899010, 0.852986, 0.801740, 0.741415, 0.673532, 0.605917, 0.535459, 0.464877, 0.401301, 0.341824, 0.288019, 0.241219, 0.201690, 0.169479, 0.143868, 0.124166, 0.109343, 0.098544, 0.090561, 0.084938, 0.080723, 0.076826, 0.072366, 0.066664, 0.059786, 0.052042, 0.044155, 0.037440, 0.032226, 0.028451, 0.025802, 0.023649, 0.021300, 0.018598, 0.016085, 0.014572, 0.014648, 0.016036, 0.017065, 0.015370, 0.011483, 0.008393, 0.007253, 0.007162, 0.006734, 0.006337, 0.007576, 0.009671, 0.008863, 0.006913, 0.006223, 0.005968, 0.008224, 0.008089, 0.003574, 0.003039, 0.001618, 0.001672, 0.001290]])

    N = 4096
    number_of_band = 64

    H = np.zeros(N)
    for band in range(number_of_band):
        fL = f[band]
        fH = f[band + 1]
        for k in range(fL, fH):
            H[k] = a[note_number - 40][band]

    for k in range(1, int(N / 2)):
        H[N - k] = H[k]

    h = np.real(np.fft.ifft(H, N))

    for m in range(int(N / 2)):
        tmp = h[m]
        h[m] = h[int(N / 2) + m]
        h[int(N / 2) + m] = tmp

    J = 128

    b = np.zeros(J + 1)
    w = Hanning_window(J + 1)
    offset = int(N / 2) - int(J / 2)
    for m in range(J + 1):
        b[m] = h[offset + m] * w[m]

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    # decay
    p1 = 60
    p2 = 0
    p3 = 120 / 12
    p4 = 20
    decay = (1 - 1 / (1 + np.exp(-(note_number - p1) / p3))) * p4 + p2

    # d
    p1 = 70
    p2 = 0
    p3 = 120 / 12
    p4 = 0.5
    d = (1 - 1 / (1 + np.exp(-(note_number - p1) / p3))) * p4 + p2

    num = np.power(10, -3 * T / decay)
    den = np.sqrt((1 - d) * (1 - d) + 2 * d * (1 - d) * np.cos((2 * np.pi * f0) / fs) + d * d)

    c = num / den

    if c > 1:
        c = 1

    D = int(T * fs - d)
    e = T * fs - d - int(T * fs - d)
    g = (1 - e) / (1 + e)

    s0 = np.zeros(length_of_s)
    s1 = np.zeros(length_of_s)
    s2 = np.zeros(length_of_s)

    np.random.seed(0)
    number_of_partial = int(20000 / f0)
    for i in range(number_of_partial):
        theta = (np.random.rand() * 2 - 1) * np.pi
        for n in range(D + 1 + J):
            s0[n] += np.sin(2 * np.pi * f0 * (i + 1) * n / fs + theta)

    mean_of_s0 = 0
    for n in range(J, D + 1 + J):
        mean_of_s0 += s0[n]

    mean_of_s0 /= D + 1
    for n in range(D + 1 + J):
        s0[n] -= mean_of_s0

    for n in range(D + 1 + J, length_of_s):
        # fractional delay
        s1[n] = -g * s1[n - 1] + g * s0[n - D] + s0[n - D - 1]

        # filter
        s2[n] = c * ((1 - d) * s1[n] + d * s1[n - 1])

        # feedback
        s0[n] += s2[n]

    # comb filter
    excitation = np.array([0.25, 0.25, 0.25, 0.3, 0.3, 0.34, 0.34, 0.34, 0.3, 0.3, 0.34, 0.34, 0.38, 0.38, 0.38, 0.34, 0.34, 0.38, 0.38, 0.38, 0.34, 0.34, 0.38, 0.38, 0.43, 0.43, 0.43, 0.4, 0.4, 0.45, 0.45, 0.45, 0.4, 0.4, 0.45, 0.45, 0.51, 0.51, 0.51, 0.6, 0.6, 0.68, 0.68, 0.68, 0.81])
    p = excitation[note_number - 40]
    s3 = np.zeros(length_of_s)
    for n in range(length_of_s):
        t = n - T * fs * p
        m = int(t)
        delta = t - m
        if m < 0:
            s3[n] = s0[n] - (delta * s0[m + 1 + D + 1] + (1 - delta) * s0[m + D + 1])
        else:
            s3[n] = s0[n] - (delta * s0[m + 1] + (1 - delta) * s0[m])

    s4 = np.zeros(length_of_s)
    for n in range(length_of_s):
        for m in range(J + 1):
            if n - m >= 0:
                s4[n] += b[m] * s3[n - m]

    s5 = np.zeros(length_of_s)
    for n in range(length_of_s - J):
        s5[n] = s4[J + n]

    # DC cancel
    fc = 5
    Q = 1 / np.sqrt(2)
    a, b = HPF(fs, fc, Q)
    s6 = filter(a, b, s5)
    s6 /= np.max(np.abs(s6))

    # A part
    VCF_A = np.array([0])
    VCF_D = np.array([T * 5])
    VCF_S = np.array([0])
    VCF_R = np.array([T * 5])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([f0 * 32])
    VCF_depth = np.array([f0 * 512])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]
        if vcf[n] > 20000:
            vcf[n] = 20000

    sa = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                sa[n] += b[m] * s6[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                sa[n] += -a[m] * sa[n - m]

    VCA_A = np.array([T * 1])
    VCA_D = np.array([T * 20])
    VCA_S = np.array([0])
    VCA_R = np.array([T * 20])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        sa[n] *= vca[n]

    # B part
    VCF_A = np.array([0])
    VCF_D = np.array([decay * 0.2])
    VCF_S = np.array([0])
    VCF_R = np.array([decay * 0.2])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([f0 * 8])
    VCF_depth = np.array([f0 * 128])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]
        if vcf[n] > 20000:
            vcf[n] = 20000

    sb = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                sb[n] += b[m] * s6[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                sb[n] += -a[m] * sb[n - m]

    VCA_A = np.array([T * 1])
    VCA_D = np.array([0])
    VCA_S = np.array([1])
    VCA_R = np.array([0])
    VCA_gate = np.array([duration])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        sb[n] *= vca[n]

    s7 = sa * 0.5 + sb * 0.5

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
        s7[n] *= vca[n]

    s7 *= velocity / 127 / np.max(np.abs(s7))

    return s7

def electric_guitar(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    s0 = np.zeros(length_of_s)

    f = np.array([0, 4, 7, 12, 16, 20, 25, 30, 35, 41, 47, 53, 60, 67, 74, 82, 90, 99, 108, 118, 128, 139, 150, 162, 175, 188, 202, 217, 233, 250, 267, 286, 306, 326, 348, 371, 396, 421, 449, 477, 508, 540, 574, 609, 647, 687, 729, 773, 820, 869, 922, 977, 1035, 1097, 1161, 1230, 1302, 1379, 1460, 1545, 1635, 1730, 1830, 1936, 2048])
    a = np.array([[0.042870, 0.101982, 0.383088, 0.721357, 0.619579, 0.517834, 0.682021, 0.911216, 1.000000, 0.753317, 0.226584, 0.083716, 0.186115, 0.329012, 0.365911, 0.262354, 0.142569, 0.070802, 0.031780, 0.070772, 0.096276, 0.173273, 0.103871, 0.034668, 0.044158, 0.048280, 0.031350, 0.009431, 0.003140, 0.007009, 0.008527, 0.007561, 0.003104, 0.002762, 0.002116, 0.000914, 0.000418, 0.000672, 0.000656, 0.000603, 0.000430, 0.000327, 0.000258, 0.000233, 0.000316, 0.000300, 0.000318, 0.000387, 0.000273, 0.000382, 0.000367, 0.000315, 0.000293, 0.000299, 0.000233, 0.000237, 0.000252, 0.000317, 0.000270, 0.000246, 0.000313, 0.000249, 0.000324, 0.000240],
                  [0.042870, 0.101982, 0.383088, 0.721357, 0.619579, 0.517834, 0.682021, 0.911216, 1.000000, 0.753317, 0.226584, 0.083716, 0.186115, 0.329012, 0.365911, 0.262354, 0.142569, 0.070802, 0.031780, 0.070772, 0.096276, 0.173273, 0.103871, 0.034668, 0.044158, 0.048280, 0.031350, 0.009431, 0.003140, 0.007009, 0.008527, 0.007561, 0.003104, 0.002762, 0.002116, 0.000914, 0.000418, 0.000672, 0.000656, 0.000603, 0.000430, 0.000327, 0.000258, 0.000233, 0.000316, 0.000300, 0.000318, 0.000387, 0.000273, 0.000382, 0.000367, 0.000315, 0.000293, 0.000299, 0.000233, 0.000237, 0.000252, 0.000317, 0.000270, 0.000246, 0.000313, 0.000249, 0.000324, 0.000240],
                  [0.042870, 0.101982, 0.383088, 0.721357, 0.619579, 0.517834, 0.682021, 0.911216, 1.000000, 0.753317, 0.226584, 0.083716, 0.186115, 0.329012, 0.365911, 0.262354, 0.142569, 0.070802, 0.031780, 0.070772, 0.096276, 0.173273, 0.103871, 0.034668, 0.044158, 0.048280, 0.031350, 0.009431, 0.003140, 0.007009, 0.008527, 0.007561, 0.003104, 0.002762, 0.002116, 0.000914, 0.000418, 0.000672, 0.000656, 0.000603, 0.000430, 0.000327, 0.000258, 0.000233, 0.000316, 0.000300, 0.000318, 0.000387, 0.000273, 0.000382, 0.000367, 0.000315, 0.000293, 0.000299, 0.000233, 0.000237, 0.000252, 0.000317, 0.000270, 0.000246, 0.000313, 0.000249, 0.000324, 0.000240],
                  [0.054300, 0.105990, 0.324088, 0.677615, 0.789725, 0.830218, 1.000000, 0.962970, 0.615219, 0.402590, 0.269681, 0.129646, 0.128123, 0.259391, 0.272515, 0.227278, 0.162283, 0.073320, 0.045457, 0.050562, 0.081528, 0.104925, 0.104746, 0.058594, 0.027504, 0.030928, 0.030534, 0.011119, 0.005028, 0.004784, 0.006618, 0.006413, 0.004054, 0.002720, 0.001708, 0.000815, 0.000513, 0.000637, 0.000776, 0.000639, 0.000453, 0.000385, 0.000270, 0.000252, 0.000253, 0.000263, 0.000235, 0.000232, 0.000199, 0.000245, 0.000259, 0.000239, 0.000258, 0.000233, 0.000217, 0.000232, 0.000225, 0.000222, 0.000233, 0.000256, 0.000201, 0.000223, 0.000220, 0.000199],
                  [0.054300, 0.105990, 0.324088, 0.677615, 0.789725, 0.830218, 1.000000, 0.962970, 0.615219, 0.402590, 0.269681, 0.129646, 0.128123, 0.259391, 0.272515, 0.227278, 0.162283, 0.073320, 0.045457, 0.050562, 0.081528, 0.104925, 0.104746, 0.058594, 0.027504, 0.030928, 0.030534, 0.011119, 0.005028, 0.004784, 0.006618, 0.006413, 0.004054, 0.002720, 0.001708, 0.000815, 0.000513, 0.000637, 0.000776, 0.000639, 0.000453, 0.000385, 0.000270, 0.000252, 0.000253, 0.000263, 0.000235, 0.000232, 0.000199, 0.000245, 0.000259, 0.000239, 0.000258, 0.000233, 0.000217, 0.000232, 0.000225, 0.000222, 0.000233, 0.000256, 0.000201, 0.000223, 0.000220, 0.000199],
                  [0.069323, 0.127784, 0.388837, 0.886538, 1.000000, 0.721029, 0.533096, 0.576768, 0.730254, 0.766992, 0.554418, 0.250845, 0.119502, 0.150757, 0.223468, 0.190630, 0.148462, 0.071899, 0.042428, 0.047509, 0.056164, 0.094310, 0.098907, 0.106412, 0.044771, 0.029575, 0.025546, 0.015627, 0.004453, 0.008552, 0.006758, 0.007442, 0.008901, 0.004418, 0.002437, 0.001126, 0.000777, 0.001060, 0.000994, 0.000896, 0.000625, 0.000357, 0.000265, 0.000256, 0.000308, 0.000323, 0.000329, 0.000300, 0.000288, 0.000394, 0.000405, 0.000261, 0.000297, 0.000300, 0.000262, 0.000328, 0.000291, 0.000346, 0.000299, 0.000278, 0.000288, 0.000334, 0.000302, 0.000274],
                  [0.069323, 0.127784, 0.388837, 0.886538, 1.000000, 0.721029, 0.533096, 0.576768, 0.730254, 0.766992, 0.554418, 0.250845, 0.119502, 0.150757, 0.223468, 0.190630, 0.148462, 0.071899, 0.042428, 0.047509, 0.056164, 0.094310, 0.098907, 0.106412, 0.044771, 0.029575, 0.025546, 0.015627, 0.004453, 0.008552, 0.006758, 0.007442, 0.008901, 0.004418, 0.002437, 0.001126, 0.000777, 0.001060, 0.000994, 0.000896, 0.000625, 0.000357, 0.000265, 0.000256, 0.000308, 0.000323, 0.000329, 0.000300, 0.000288, 0.000394, 0.000405, 0.000261, 0.000297, 0.000300, 0.000262, 0.000328, 0.000291, 0.000346, 0.000299, 0.000278, 0.000288, 0.000334, 0.000302, 0.000274],
                  [0.069323, 0.127784, 0.388837, 0.886538, 1.000000, 0.721029, 0.533096, 0.576768, 0.730254, 0.766992, 0.554418, 0.250845, 0.119502, 0.150757, 0.223468, 0.190630, 0.148462, 0.071899, 0.042428, 0.047509, 0.056164, 0.094310, 0.098907, 0.106412, 0.044771, 0.029575, 0.025546, 0.015627, 0.004453, 0.008552, 0.006758, 0.007442, 0.008901, 0.004418, 0.002437, 0.001126, 0.000777, 0.001060, 0.000994, 0.000896, 0.000625, 0.000357, 0.000265, 0.000256, 0.000308, 0.000323, 0.000329, 0.000300, 0.000288, 0.000394, 0.000405, 0.000261, 0.000297, 0.000300, 0.000262, 0.000328, 0.000291, 0.000346, 0.000299, 0.000278, 0.000288, 0.000334, 0.000302, 0.000274],
                  [0.061897, 0.098793, 0.253066, 0.620405, 0.945687, 1.000000, 0.902474, 0.876699, 0.794960, 0.530156, 0.359007, 0.403074, 0.436705, 0.207971, 0.095017, 0.198415, 0.442760, 0.321012, 0.239937, 0.237999, 0.143680, 0.046788, 0.026143, 0.057228, 0.106323, 0.146292, 0.182233, 0.093525, 0.090692, 0.087720, 0.054326, 0.023223, 0.008342, 0.006383, 0.010050, 0.008080, 0.003715, 0.004971, 0.003783, 0.002316, 0.001410, 0.001233, 0.001214, 0.001246, 0.001337, 0.001178, 0.001119, 0.000887, 0.001153, 0.001074, 0.000984, 0.001087, 0.001026, 0.000980, 0.000972, 0.000942, 0.000971, 0.001156, 0.001341, 0.000957, 0.001031, 0.001023, 0.001065, 0.001043],
                  [0.061897, 0.098793, 0.253066, 0.620405, 0.945687, 1.000000, 0.902474, 0.876699, 0.794960, 0.530156, 0.359007, 0.403074, 0.436705, 0.207971, 0.095017, 0.198415, 0.442760, 0.321012, 0.239937, 0.237999, 0.143680, 0.046788, 0.026143, 0.057228, 0.106323, 0.146292, 0.182233, 0.093525, 0.090692, 0.087720, 0.054326, 0.023223, 0.008342, 0.006383, 0.010050, 0.008080, 0.003715, 0.004971, 0.003783, 0.002316, 0.001410, 0.001233, 0.001214, 0.001246, 0.001337, 0.001178, 0.001119, 0.000887, 0.001153, 0.001074, 0.000984, 0.001087, 0.001026, 0.000980, 0.000972, 0.000942, 0.000971, 0.001156, 0.001341, 0.000957, 0.001031, 0.001023, 0.001065, 0.001043],
                  [0.070831, 0.102528, 0.224161, 0.517823, 0.844636, 1.000000, 0.924423, 0.839657, 0.780392, 0.578856, 0.313924, 0.188739, 0.235846, 0.424528, 0.538595, 0.522242, 0.489496, 0.241871, 0.097732, 0.151970, 0.170055, 0.070471, 0.040419, 0.043663, 0.116473, 0.121562, 0.145093, 0.055084, 0.120236, 0.120333, 0.056978, 0.025179, 0.011232, 0.008419, 0.009386, 0.013468, 0.008698, 0.006600, 0.004598, 0.002327, 0.002047, 0.001546, 0.001409, 0.001076, 0.001014, 0.000915, 0.000777, 0.000632, 0.000801, 0.000720, 0.000703, 0.000715, 0.000714, 0.000711, 0.000738, 0.000840, 0.000734, 0.000904, 0.000688, 0.000695, 0.000791, 0.000755, 0.000748, 0.000582],
                  [0.070831, 0.102528, 0.224161, 0.517823, 0.844636, 1.000000, 0.924423, 0.839657, 0.780392, 0.578856, 0.313924, 0.188739, 0.235846, 0.424528, 0.538595, 0.522242, 0.489496, 0.241871, 0.097732, 0.151970, 0.170055, 0.070471, 0.040419, 0.043663, 0.116473, 0.121562, 0.145093, 0.055084, 0.120236, 0.120333, 0.056978, 0.025179, 0.011232, 0.008419, 0.009386, 0.013468, 0.008698, 0.006600, 0.004598, 0.002327, 0.002047, 0.001546, 0.001409, 0.001076, 0.001014, 0.000915, 0.000777, 0.000632, 0.000801, 0.000720, 0.000703, 0.000715, 0.000714, 0.000711, 0.000738, 0.000840, 0.000734, 0.000904, 0.000688, 0.000695, 0.000791, 0.000755, 0.000748, 0.000582],
                  [0.057182, 0.085669, 0.204396, 0.522172, 0.895461, 1.000000, 0.750502, 0.547408, 0.532535, 0.600159, 0.457677, 0.195968, 0.098102, 0.175152, 0.570444, 0.694612, 0.263477, 0.133772, 0.130286, 0.078095, 0.063962, 0.069637, 0.029247, 0.036537, 0.063628, 0.061631, 0.145079, 0.100616, 0.064500, 0.077850, 0.082886, 0.044382, 0.015918, 0.010283, 0.010186, 0.005648, 0.008424, 0.005655, 0.002271, 0.002565, 0.002019, 0.001571, 0.001926, 0.001891, 0.001633, 0.001741, 0.001422, 0.001330, 0.001070, 0.001426, 0.001113, 0.001254, 0.001234, 0.001651, 0.001267, 0.001218, 0.001446, 0.001416, 0.001321, 0.001296, 0.001326, 0.001262, 0.001094, 0.001230],
                  [0.057182, 0.085669, 0.204396, 0.522172, 0.895461, 1.000000, 0.750502, 0.547408, 0.532535, 0.600159, 0.457677, 0.195968, 0.098102, 0.175152, 0.570444, 0.694612, 0.263477, 0.133772, 0.130286, 0.078095, 0.063962, 0.069637, 0.029247, 0.036537, 0.063628, 0.061631, 0.145079, 0.100616, 0.064500, 0.077850, 0.082886, 0.044382, 0.015918, 0.010283, 0.010186, 0.005648, 0.008424, 0.005655, 0.002271, 0.002565, 0.002019, 0.001571, 0.001926, 0.001891, 0.001633, 0.001741, 0.001422, 0.001330, 0.001070, 0.001426, 0.001113, 0.001254, 0.001234, 0.001651, 0.001267, 0.001218, 0.001446, 0.001416, 0.001321, 0.001296, 0.001326, 0.001262, 0.001094, 0.001230],
                  [0.057182, 0.085669, 0.204396, 0.522172, 0.895461, 1.000000, 0.750502, 0.547408, 0.532535, 0.600159, 0.457677, 0.195968, 0.098102, 0.175152, 0.570444, 0.694612, 0.263477, 0.133772, 0.130286, 0.078095, 0.063962, 0.069637, 0.029247, 0.036537, 0.063628, 0.061631, 0.145079, 0.100616, 0.064500, 0.077850, 0.082886, 0.044382, 0.015918, 0.010283, 0.010186, 0.005648, 0.008424, 0.005655, 0.002271, 0.002565, 0.002019, 0.001571, 0.001926, 0.001891, 0.001633, 0.001741, 0.001422, 0.001330, 0.001070, 0.001426, 0.001113, 0.001254, 0.001234, 0.001651, 0.001267, 0.001218, 0.001446, 0.001416, 0.001321, 0.001296, 0.001326, 0.001262, 0.001094, 0.001230],
                  [0.057760, 0.070143, 0.111049, 0.209733, 0.371515, 0.623368, 0.879793, 1.000000, 0.998544, 0.961451, 0.921342, 0.807541, 0.598622, 0.428438, 0.387101, 0.491013, 0.576704, 0.378317, 0.168737, 0.139987, 0.272250, 0.383591, 0.264232, 0.202518, 0.211703, 0.133452, 0.056328, 0.037195, 0.056088, 0.114385, 0.145356, 0.200631, 0.152631, 0.066458, 0.029678, 0.042297, 0.031065, 0.015469, 0.008685, 0.008824, 0.012647, 0.012428, 0.010995, 0.007404, 0.005732, 0.004321, 0.004465, 0.003983, 0.004078, 0.003748, 0.003588, 0.003321, 0.003364, 0.003658, 0.003670, 0.003028, 0.003594, 0.003621, 0.003372, 0.002999, 0.003943, 0.003615, 0.003587, 0.003402],
                  [0.057760, 0.070143, 0.111049, 0.209733, 0.371515, 0.623368, 0.879793, 1.000000, 0.998544, 0.961451, 0.921342, 0.807541, 0.598622, 0.428438, 0.387101, 0.491013, 0.576704, 0.378317, 0.168737, 0.139987, 0.272250, 0.383591, 0.264232, 0.202518, 0.211703, 0.133452, 0.056328, 0.037195, 0.056088, 0.114385, 0.145356, 0.200631, 0.152631, 0.066458, 0.029678, 0.042297, 0.031065, 0.015469, 0.008685, 0.008824, 0.012647, 0.012428, 0.010995, 0.007404, 0.005732, 0.004321, 0.004465, 0.003983, 0.004078, 0.003748, 0.003588, 0.003321, 0.003364, 0.003658, 0.003670, 0.003028, 0.003594, 0.003621, 0.003372, 0.002999, 0.003943, 0.003615, 0.003587, 0.003402],
                  [0.059197, 0.071537, 0.112732, 0.214944, 0.391146, 0.680514, 0.962465, 1.000000, 0.793555, 0.551947, 0.430430, 0.424499, 0.491749, 0.507813, 0.375036, 0.206363, 0.125736, 0.129625, 0.205971, 0.302933, 0.344559, 0.369817, 0.320044, 0.137185, 0.043112, 0.032458, 0.037595, 0.036753, 0.054186, 0.053944, 0.047344, 0.082054, 0.106412, 0.094283, 0.036698, 0.034391, 0.030588, 0.015015, 0.008264, 0.011692, 0.011046, 0.006501, 0.005299, 0.006061, 0.004098, 0.004697, 0.003511, 0.003600, 0.003502, 0.004092, 0.003736, 0.004019, 0.003610, 0.003327, 0.003442, 0.003962, 0.004447, 0.003280, 0.002958, 0.003495, 0.003471, 0.003125, 0.003488, 0.003091],
                  [0.059197, 0.071537, 0.112732, 0.214944, 0.391146, 0.680514, 0.962465, 1.000000, 0.793555, 0.551947, 0.430430, 0.424499, 0.491749, 0.507813, 0.375036, 0.206363, 0.125736, 0.129625, 0.205971, 0.302933, 0.344559, 0.369817, 0.320044, 0.137185, 0.043112, 0.032458, 0.037595, 0.036753, 0.054186, 0.053944, 0.047344, 0.082054, 0.106412, 0.094283, 0.036698, 0.034391, 0.030588, 0.015015, 0.008264, 0.011692, 0.011046, 0.006501, 0.005299, 0.006061, 0.004098, 0.004697, 0.003511, 0.003600, 0.003502, 0.004092, 0.003736, 0.004019, 0.003610, 0.003327, 0.003442, 0.003962, 0.004447, 0.003280, 0.002958, 0.003495, 0.003471, 0.003125, 0.003488, 0.003091],
                  [0.059197, 0.071537, 0.112732, 0.214944, 0.391146, 0.680514, 0.962465, 1.000000, 0.793555, 0.551947, 0.430430, 0.424499, 0.491749, 0.507813, 0.375036, 0.206363, 0.125736, 0.129625, 0.205971, 0.302933, 0.344559, 0.369817, 0.320044, 0.137185, 0.043112, 0.032458, 0.037595, 0.036753, 0.054186, 0.053944, 0.047344, 0.082054, 0.106412, 0.094283, 0.036698, 0.034391, 0.030588, 0.015015, 0.008264, 0.011692, 0.011046, 0.006501, 0.005299, 0.006061, 0.004098, 0.004697, 0.003511, 0.003600, 0.003502, 0.004092, 0.003736, 0.004019, 0.003610, 0.003327, 0.003442, 0.003962, 0.004447, 0.003280, 0.002958, 0.003495, 0.003471, 0.003125, 0.003488, 0.003091],
                  [0.107165, 0.117871, 0.149029, 0.213990, 0.311562, 0.474700, 0.700842, 0.905507, 1.000000, 0.926329, 0.764523, 0.618510, 0.554742, 0.594682, 0.734247, 0.902240, 0.901621, 0.642164, 0.337862, 0.168343, 0.116938, 0.142597, 0.262530, 0.462666, 0.522250, 0.387296, 0.243134, 0.149902, 0.101952, 0.088117, 0.070375, 0.049861, 0.055398, 0.084576, 0.135593, 0.092877, 0.090524, 0.118337, 0.149442, 0.143672, 0.057895, 0.044576, 0.030462, 0.019298, 0.017099, 0.015661, 0.019531, 0.014257, 0.011695, 0.011132, 0.014223, 0.006935, 0.007160, 0.006591, 0.006550, 0.005547, 0.005084, 0.005552, 0.006231, 0.005231, 0.006157, 0.005406, 0.005711, 0.004962],
                  [0.107165, 0.117871, 0.149029, 0.213990, 0.311562, 0.474700, 0.700842, 0.905507, 1.000000, 0.926329, 0.764523, 0.618510, 0.554742, 0.594682, 0.734247, 0.902240, 0.901621, 0.642164, 0.337862, 0.168343, 0.116938, 0.142597, 0.262530, 0.462666, 0.522250, 0.387296, 0.243134, 0.149902, 0.101952, 0.088117, 0.070375, 0.049861, 0.055398, 0.084576, 0.135593, 0.092877, 0.090524, 0.118337, 0.149442, 0.143672, 0.057895, 0.044576, 0.030462, 0.019298, 0.017099, 0.015661, 0.019531, 0.014257, 0.011695, 0.011132, 0.014223, 0.006935, 0.007160, 0.006591, 0.006550, 0.005547, 0.005084, 0.005552, 0.006231, 0.005231, 0.006157, 0.005406, 0.005711, 0.004962],
                  [0.097337, 0.104709, 0.125695, 0.168380, 0.231881, 0.342011, 0.514766, 0.721180, 0.917327, 1.000000, 0.918976, 0.734908, 0.551756, 0.441248, 0.401799, 0.418599, 0.447529, 0.409916, 0.286098, 0.167938, 0.117346, 0.137084, 0.246175, 0.395938, 0.360769, 0.219762, 0.152669, 0.131123, 0.089799, 0.056519, 0.049266, 0.042405, 0.043656, 0.081837, 0.095769, 0.082894, 0.083747, 0.075948, 0.100552, 0.086299, 0.104807, 0.056198, 0.029868, 0.028314, 0.020871, 0.015804, 0.016798, 0.016886, 0.012802, 0.009391, 0.009460, 0.007213, 0.008009, 0.007657, 0.005898, 0.006226, 0.005826, 0.006369, 0.007085, 0.006459, 0.006764, 0.006064, 0.006155, 0.005278],
                  [0.097337, 0.104709, 0.125695, 0.168380, 0.231881, 0.342011, 0.514766, 0.721180, 0.917327, 1.000000, 0.918976, 0.734908, 0.551756, 0.441248, 0.401799, 0.418599, 0.447529, 0.409916, 0.286098, 0.167938, 0.117346, 0.137084, 0.246175, 0.395938, 0.360769, 0.219762, 0.152669, 0.131123, 0.089799, 0.056519, 0.049266, 0.042405, 0.043656, 0.081837, 0.095769, 0.082894, 0.083747, 0.075948, 0.100552, 0.086299, 0.104807, 0.056198, 0.029868, 0.028314, 0.020871, 0.015804, 0.016798, 0.016886, 0.012802, 0.009391, 0.009460, 0.007213, 0.008009, 0.007657, 0.005898, 0.006226, 0.005826, 0.006369, 0.007085, 0.006459, 0.006764, 0.006064, 0.006155, 0.005278],
                  [0.209435, 0.217905, 0.240686, 0.283076, 0.339349, 0.425988, 0.549184, 0.691706, 0.844606, 0.964640, 1.000000, 0.938701, 0.801142, 0.643841, 0.495007, 0.372623, 0.280682, 0.215834, 0.176825, 0.164764, 0.187625, 0.260204, 0.387997, 0.501643, 0.462691, 0.309734, 0.183810, 0.123261, 0.097794, 0.082043, 0.070219, 0.066339, 0.070211, 0.076339, 0.081026, 0.080633, 0.077326, 0.091396, 0.095661, 0.076436, 0.080341, 0.078296, 0.105777, 0.056286, 0.029505, 0.023837, 0.029991, 0.024184, 0.017082, 0.018352, 0.011740, 0.008721, 0.011384, 0.012205, 0.009651, 0.007363, 0.007668, 0.010044, 0.006911, 0.007900, 0.009247, 0.009470, 0.009371, 0.007104],
                  [0.209435, 0.217905, 0.240686, 0.283076, 0.339349, 0.425988, 0.549184, 0.691706, 0.844606, 0.964640, 1.000000, 0.938701, 0.801142, 0.643841, 0.495007, 0.372623, 0.280682, 0.215834, 0.176825, 0.164764, 0.187625, 0.260204, 0.387997, 0.501643, 0.462691, 0.309734, 0.183810, 0.123261, 0.097794, 0.082043, 0.070219, 0.066339, 0.070211, 0.076339, 0.081026, 0.080633, 0.077326, 0.091396, 0.095661, 0.076436, 0.080341, 0.078296, 0.105777, 0.056286, 0.029505, 0.023837, 0.029991, 0.024184, 0.017082, 0.018352, 0.011740, 0.008721, 0.011384, 0.012205, 0.009651, 0.007363, 0.007668, 0.010044, 0.006911, 0.007900, 0.009247, 0.009470, 0.009371, 0.007104],
                  [0.209435, 0.217905, 0.240686, 0.283076, 0.339349, 0.425988, 0.549184, 0.691706, 0.844606, 0.964640, 1.000000, 0.938701, 0.801142, 0.643841, 0.495007, 0.372623, 0.280682, 0.215834, 0.176825, 0.164764, 0.187625, 0.260204, 0.387997, 0.501643, 0.462691, 0.309734, 0.183810, 0.123261, 0.097794, 0.082043, 0.070219, 0.066339, 0.070211, 0.076339, 0.081026, 0.080633, 0.077326, 0.091396, 0.095661, 0.076436, 0.080341, 0.078296, 0.105777, 0.056286, 0.029505, 0.023837, 0.029991, 0.024184, 0.017082, 0.018352, 0.011740, 0.008721, 0.011384, 0.012205, 0.009651, 0.007363, 0.007668, 0.010044, 0.006911, 0.007900, 0.009247, 0.009470, 0.009371, 0.007104],
                  [0.280785, 0.289138, 0.311258, 0.351427, 0.403098, 0.480026, 0.586194, 0.707018, 0.838328, 0.949759, 1.000000, 0.976617, 0.889606, 0.782941, 0.689491, 0.635642, 0.634549, 0.680976, 0.743772, 0.757646, 0.668175, 0.509041, 0.367562, 0.292687, 0.294983, 0.360275, 0.441312, 0.451760, 0.388996, 0.333635, 0.298473, 0.236678, 0.158655, 0.108723, 0.080936, 0.057721, 0.052670, 0.077687, 0.088938, 0.072472, 0.103957, 0.138159, 0.090182, 0.058159, 0.049390, 0.046506, 0.026745, 0.027575, 0.030699, 0.027862, 0.031985, 0.032755, 0.030893, 0.033279, 0.015752, 0.015758, 0.014369, 0.017606, 0.015592, 0.013182, 0.013938, 0.014689, 0.012504, 0.010471],
                  [0.280785, 0.289138, 0.311258, 0.351427, 0.403098, 0.480026, 0.586194, 0.707018, 0.838328, 0.949759, 1.000000, 0.976617, 0.889606, 0.782941, 0.689491, 0.635642, 0.634549, 0.680976, 0.743772, 0.757646, 0.668175, 0.509041, 0.367562, 0.292687, 0.294983, 0.360275, 0.441312, 0.451760, 0.388996, 0.333635, 0.298473, 0.236678, 0.158655, 0.108723, 0.080936, 0.057721, 0.052670, 0.077687, 0.088938, 0.072472, 0.103957, 0.138159, 0.090182, 0.058159, 0.049390, 0.046506, 0.026745, 0.027575, 0.030699, 0.027862, 0.031985, 0.032755, 0.030893, 0.033279, 0.015752, 0.015758, 0.014369, 0.017606, 0.015592, 0.013182, 0.013938, 0.014689, 0.012504, 0.010471],
                  [0.190408, 0.194117, 0.203890, 0.221548, 0.244263, 0.278610, 0.328311, 0.391053, 0.474877, 0.580152, 0.692741, 0.810040, 0.914265, 0.980483, 1.000000, 0.964893, 0.875157, 0.742944, 0.588280, 0.437447, 0.314585, 0.231686, 0.190204, 0.187099, 0.222733, 0.286717, 0.335447, 0.312640, 0.245349, 0.203833, 0.199775, 0.187427, 0.133544, 0.087395, 0.081800, 0.098174, 0.093606, 0.084582, 0.088943, 0.075093, 0.081738, 0.118011, 0.111936, 0.147112, 0.097303, 0.073920, 0.076525, 0.058572, 0.055340, 0.069298, 0.051016, 0.054013, 0.059091, 0.046555, 0.043331, 0.043703, 0.033695, 0.045599, 0.041310, 0.038505, 0.033817, 0.032142, 0.036487, 0.031773],
                  [0.190408, 0.194117, 0.203890, 0.221548, 0.244263, 0.278610, 0.328311, 0.391053, 0.474877, 0.580152, 0.692741, 0.810040, 0.914265, 0.980483, 1.000000, 0.964893, 0.875157, 0.742944, 0.588280, 0.437447, 0.314585, 0.231686, 0.190204, 0.187099, 0.222733, 0.286717, 0.335447, 0.312640, 0.245349, 0.203833, 0.199775, 0.187427, 0.133544, 0.087395, 0.081800, 0.098174, 0.093606, 0.084582, 0.088943, 0.075093, 0.081738, 0.118011, 0.111936, 0.147112, 0.097303, 0.073920, 0.076525, 0.058572, 0.055340, 0.069298, 0.051016, 0.054013, 0.059091, 0.046555, 0.043331, 0.043703, 0.033695, 0.045599, 0.041310, 0.038505, 0.033817, 0.032142, 0.036487, 0.031773],
                  [0.190408, 0.194117, 0.203890, 0.221548, 0.244263, 0.278610, 0.328311, 0.391053, 0.474877, 0.580152, 0.692741, 0.810040, 0.914265, 0.980483, 1.000000, 0.964893, 0.875157, 0.742944, 0.588280, 0.437447, 0.314585, 0.231686, 0.190204, 0.187099, 0.222733, 0.286717, 0.335447, 0.312640, 0.245349, 0.203833, 0.199775, 0.187427, 0.133544, 0.087395, 0.081800, 0.098174, 0.093606, 0.084582, 0.088943, 0.075093, 0.081738, 0.118011, 0.111936, 0.147112, 0.097303, 0.073920, 0.076525, 0.058572, 0.055340, 0.069298, 0.051016, 0.054013, 0.059091, 0.046555, 0.043331, 0.043703, 0.033695, 0.045599, 0.041310, 0.038505, 0.033817, 0.032142, 0.036487, 0.031773],
                  [0.287891, 0.292246, 0.303650, 0.324048, 0.349912, 0.388316, 0.442667, 0.509533, 0.596187, 0.700965, 0.807132, 0.907757, 0.980197, 1.000000, 0.959687, 0.864012, 0.733123, 0.595576, 0.474194, 0.379092, 0.310821, 0.263460, 0.230596, 0.206008, 0.188840, 0.180353, 0.184220, 0.204525, 0.236869, 0.256060, 0.231111, 0.169286, 0.117119, 0.094122, 0.093429, 0.096690, 0.087442, 0.072870, 0.062604, 0.051581, 0.042135, 0.044299, 0.049808, 0.047626, 0.054433, 0.056883, 0.042195, 0.035833, 0.034924, 0.042144, 0.033701, 0.034705, 0.025853, 0.026877, 0.034696, 0.030721, 0.032378, 0.033019, 0.031438, 0.024779, 0.025422, 0.024324, 0.027785, 0.021256],
                  [0.287891, 0.292246, 0.303650, 0.324048, 0.349912, 0.388316, 0.442667, 0.509533, 0.596187, 0.700965, 0.807132, 0.907757, 0.980197, 1.000000, 0.959687, 0.864012, 0.733123, 0.595576, 0.474194, 0.379092, 0.310821, 0.263460, 0.230596, 0.206008, 0.188840, 0.180353, 0.184220, 0.204525, 0.236869, 0.256060, 0.231111, 0.169286, 0.117119, 0.094122, 0.093429, 0.096690, 0.087442, 0.072870, 0.062604, 0.051581, 0.042135, 0.044299, 0.049808, 0.047626, 0.054433, 0.056883, 0.042195, 0.035833, 0.034924, 0.042144, 0.033701, 0.034705, 0.025853, 0.026877, 0.034696, 0.030721, 0.032378, 0.033019, 0.031438, 0.024779, 0.025422, 0.024324, 0.027785, 0.021256],
                  [0.141805, 0.144554, 0.151835, 0.165110, 0.182441, 0.209223, 0.249228, 0.301997, 0.376836, 0.478271, 0.597139, 0.734438, 0.870189, 0.966066, 1.000000, 0.955461, 0.840386, 0.689497, 0.541524, 0.421830, 0.339004, 0.288806, 0.263454, 0.252722, 0.246335, 0.233769, 0.210058, 0.179876, 0.153045, 0.136014, 0.126923, 0.119456, 0.108932, 0.097211, 0.090317, 0.089977, 0.088542, 0.077779, 0.062062, 0.048504, 0.037328, 0.031054, 0.033191, 0.039037, 0.043400, 0.054518, 0.050869, 0.035956, 0.041982, 0.037922, 0.039041, 0.034539, 0.025884, 0.024935, 0.032784, 0.030923, 0.039042, 0.029845, 0.026351, 0.024657, 0.025990, 0.025014, 0.024700, 0.026812],
                  [0.141805, 0.144554, 0.151835, 0.165110, 0.182441, 0.209223, 0.249228, 0.301997, 0.376836, 0.478271, 0.597139, 0.734438, 0.870189, 0.966066, 1.000000, 0.955461, 0.840386, 0.689497, 0.541524, 0.421830, 0.339004, 0.288806, 0.263454, 0.252722, 0.246335, 0.233769, 0.210058, 0.179876, 0.153045, 0.136014, 0.126923, 0.119456, 0.108932, 0.097211, 0.090317, 0.089977, 0.088542, 0.077779, 0.062062, 0.048504, 0.037328, 0.031054, 0.033191, 0.039037, 0.043400, 0.054518, 0.050869, 0.035956, 0.041982, 0.037922, 0.039041, 0.034539, 0.025884, 0.024935, 0.032784, 0.030923, 0.039042, 0.029845, 0.026351, 0.024657, 0.025990, 0.025014, 0.024700, 0.026812],
                  [0.280702, 0.283787, 0.291842, 0.306195, 0.324315, 0.351144, 0.389169, 0.436442, 0.499406, 0.579845, 0.669810, 0.771481, 0.874163, 0.954757, 1.000000, 0.992240, 0.923002, 0.803817, 0.659582, 0.519129, 0.403130, 0.319054, 0.266309, 0.239106, 0.234200, 0.245729, 0.266253, 0.283694, 0.284705, 0.265925, 0.235721, 0.205416, 0.182161, 0.163164, 0.144267, 0.126425, 0.114211, 0.106217, 0.091406, 0.064566, 0.041191, 0.034349, 0.042946, 0.055473, 0.056741, 0.057340, 0.060279, 0.053866, 0.054286, 0.056738, 0.042019, 0.037926, 0.035377, 0.033231, 0.041131, 0.040841, 0.043766, 0.042353, 0.035648, 0.032239, 0.023517, 0.028068, 0.027707, 0.026060],
                  [0.280702, 0.283787, 0.291842, 0.306195, 0.324315, 0.351144, 0.389169, 0.436442, 0.499406, 0.579845, 0.669810, 0.771481, 0.874163, 0.954757, 1.000000, 0.992240, 0.923002, 0.803817, 0.659582, 0.519129, 0.403130, 0.319054, 0.266309, 0.239106, 0.234200, 0.245729, 0.266253, 0.283694, 0.284705, 0.265925, 0.235721, 0.205416, 0.182161, 0.163164, 0.144267, 0.126425, 0.114211, 0.106217, 0.091406, 0.064566, 0.041191, 0.034349, 0.042946, 0.055473, 0.056741, 0.057340, 0.060279, 0.053866, 0.054286, 0.056738, 0.042019, 0.037926, 0.035377, 0.033231, 0.041131, 0.040841, 0.043766, 0.042353, 0.035648, 0.032239, 0.023517, 0.028068, 0.027707, 0.026060],
                  [0.280702, 0.283787, 0.291842, 0.306195, 0.324315, 0.351144, 0.389169, 0.436442, 0.499406, 0.579845, 0.669810, 0.771481, 0.874163, 0.954757, 1.000000, 0.992240, 0.923002, 0.803817, 0.659582, 0.519129, 0.403130, 0.319054, 0.266309, 0.239106, 0.234200, 0.245729, 0.266253, 0.283694, 0.284705, 0.265925, 0.235721, 0.205416, 0.182161, 0.163164, 0.144267, 0.126425, 0.114211, 0.106217, 0.091406, 0.064566, 0.041191, 0.034349, 0.042946, 0.055473, 0.056741, 0.057340, 0.060279, 0.053866, 0.054286, 0.056738, 0.042019, 0.037926, 0.035377, 0.033231, 0.041131, 0.040841, 0.043766, 0.042353, 0.035648, 0.032239, 0.023517, 0.028068, 0.027707, 0.026060],
                  [0.389360, 0.391347, 0.396499, 0.405570, 0.416834, 0.433172, 0.455787, 0.483232, 0.519059, 0.564435, 0.615859, 0.677117, 0.746861, 0.816781, 0.886224, 0.947053, 0.988342, 1.000000, 0.974229, 0.911304, 0.818545, 0.710148, 0.601451, 0.501256, 0.421863, 0.365311, 0.328469, 0.308981, 0.301933, 0.299721, 0.292979, 0.273712, 0.242229, 0.205245, 0.171535, 0.147517, 0.133977, 0.126696, 0.119573, 0.108087, 0.092919, 0.079578, 0.073049, 0.076037, 0.088297, 0.101104, 0.099547, 0.085707, 0.078408, 0.083550, 0.083399, 0.066396, 0.054535, 0.060942, 0.069585, 0.065694, 0.066095, 0.079239, 0.070036, 0.053468, 0.057254, 0.053840, 0.047265, 0.045223],
                  [0.389360, 0.391347, 0.396499, 0.405570, 0.416834, 0.433172, 0.455787, 0.483232, 0.519059, 0.564435, 0.615859, 0.677117, 0.746861, 0.816781, 0.886224, 0.947053, 0.988342, 1.000000, 0.974229, 0.911304, 0.818545, 0.710148, 0.601451, 0.501256, 0.421863, 0.365311, 0.328469, 0.308981, 0.301933, 0.299721, 0.292979, 0.273712, 0.242229, 0.205245, 0.171535, 0.147517, 0.133977, 0.126696, 0.119573, 0.108087, 0.092919, 0.079578, 0.073049, 0.076037, 0.088297, 0.101104, 0.099547, 0.085707, 0.078408, 0.083550, 0.083399, 0.066396, 0.054535, 0.060942, 0.069585, 0.065694, 0.066095, 0.079239, 0.070036, 0.053468, 0.057254, 0.053840, 0.047265, 0.045223],
                  [0.664047, 0.665659, 0.669818, 0.677069, 0.685942, 0.698563, 0.715587, 0.735589, 0.760675, 0.790970, 0.823457, 0.859777, 0.898247, 0.933717, 0.965464, 0.989268, 1.000000, 0.993528, 0.966774, 0.920166, 0.856908, 0.783478, 0.707515, 0.633596, 0.570894, 0.522520, 0.487411, 0.464149, 0.447833, 0.431122, 0.405485, 0.365978, 0.318526, 0.272278, 0.236239, 0.215775, 0.208219, 0.202246, 0.183494, 0.149229, 0.114735, 0.096795, 0.098782, 0.111938, 0.114982, 0.100923, 0.093113, 0.105897, 0.119649, 0.108534, 0.099707, 0.109973, 0.103283, 0.082448, 0.083378, 0.091965, 0.097188, 0.106396, 0.093108, 0.075247, 0.072277, 0.067091, 0.066776, 0.057793],
                  [0.664047, 0.665659, 0.669818, 0.677069, 0.685942, 0.698563, 0.715587, 0.735589, 0.760675, 0.790970, 0.823457, 0.859777, 0.898247, 0.933717, 0.965464, 0.989268, 1.000000, 0.993528, 0.966774, 0.920166, 0.856908, 0.783478, 0.707515, 0.633596, 0.570894, 0.522520, 0.487411, 0.464149, 0.447833, 0.431122, 0.405485, 0.365978, 0.318526, 0.272278, 0.236239, 0.215775, 0.208219, 0.202246, 0.183494, 0.149229, 0.114735, 0.096795, 0.098782, 0.111938, 0.114982, 0.100923, 0.093113, 0.105897, 0.119649, 0.108534, 0.099707, 0.109973, 0.103283, 0.082448, 0.083378, 0.091965, 0.097188, 0.106396, 0.093108, 0.075247, 0.072277, 0.067091, 0.066776, 0.057793],
                  [0.664047, 0.665659, 0.669818, 0.677069, 0.685942, 0.698563, 0.715587, 0.735589, 0.760675, 0.790970, 0.823457, 0.859777, 0.898247, 0.933717, 0.965464, 0.989268, 1.000000, 0.993528, 0.966774, 0.920166, 0.856908, 0.783478, 0.707515, 0.633596, 0.570894, 0.522520, 0.487411, 0.464149, 0.447833, 0.431122, 0.405485, 0.365978, 0.318526, 0.272278, 0.236239, 0.215775, 0.208219, 0.202246, 0.183494, 0.149229, 0.114735, 0.096795, 0.098782, 0.111938, 0.114982, 0.100923, 0.093113, 0.105897, 0.119649, 0.108534, 0.099707, 0.109973, 0.103283, 0.082448, 0.083378, 0.091965, 0.097188, 0.106396, 0.093108, 0.075247, 0.072277, 0.067091, 0.066776, 0.057793],
                  [1.000000, 0.999630, 0.998668, 0.996962, 0.994824, 0.991682, 0.987253, 0.981748, 0.974329, 0.964528, 0.952785, 0.937695, 0.918633, 0.896550, 0.869531, 0.837155, 0.799094, 0.755520, 0.706956, 0.654541, 0.599871, 0.544897, 0.491794, 0.440655, 0.395649, 0.358103, 0.327654, 0.305226, 0.290985, 0.284294, 0.283094, 0.284547, 0.284676, 0.279492, 0.266484, 0.246341, 0.223708, 0.203209, 0.187883, 0.177597, 0.169076, 0.159248, 0.148516, 0.141416, 0.142616, 0.150605, 0.154895, 0.147519, 0.137738, 0.138848, 0.143414, 0.132176, 0.118455, 0.125928, 0.132650, 0.117452, 0.122549, 0.135724, 0.113482, 0.099218, 0.080548, 0.079162, 0.087331, 0.078021],
                  [1.000000, 0.999630, 0.998668, 0.996962, 0.994824, 0.991682, 0.987253, 0.981748, 0.974329, 0.964528, 0.952785, 0.937695, 0.918633, 0.896550, 0.869531, 0.837155, 0.799094, 0.755520, 0.706956, 0.654541, 0.599871, 0.544897, 0.491794, 0.440655, 0.395649, 0.358103, 0.327654, 0.305226, 0.290985, 0.284294, 0.283094, 0.284547, 0.284676, 0.279492, 0.266484, 0.246341, 0.223708, 0.203209, 0.187883, 0.177597, 0.169076, 0.159248, 0.148516, 0.141416, 0.142616, 0.150605, 0.154895, 0.147519, 0.137738, 0.138848, 0.143414, 0.132176, 0.118455, 0.125928, 0.132650, 0.117452, 0.122549, 0.135724, 0.113482, 0.099218, 0.080548, 0.079162, 0.087331, 0.078021],
                  [1.000000, 0.998926, 0.996160, 0.991343, 0.985460, 0.977107, 0.965853, 0.952620, 0.935943, 0.915564, 0.893189, 0.867083, 0.837345, 0.806426, 0.772570, 0.736270, 0.698157, 0.658953, 0.619482, 0.580503, 0.542806, 0.506968, 0.473527, 0.441564, 0.412687, 0.386836, 0.362924, 0.340812, 0.320108, 0.300773, 0.281564, 0.261384, 0.240871, 0.219396, 0.196700, 0.173437, 0.151004, 0.130077, 0.111741, 0.097030, 0.086388, 0.080374, 0.078737, 0.080756, 0.085115, 0.090159, 0.095319, 0.101424, 0.107357, 0.105834, 0.090004, 0.069151, 0.059130, 0.064928, 0.076309, 0.077627, 0.077538, 0.083563, 0.075772, 0.059313, 0.050661, 0.051918, 0.060802, 0.045720]])

    N = 4096
    number_of_band = 64

    H = np.zeros(N)
    for band in range(number_of_band):
        fL = f[band]
        fH = f[band + 1]
        for k in range(fL, fH):
            H[k] = a[note_number - 40][band]

    for k in range(1, int(N / 2)):
        H[N - k] = H[k]

    h = np.real(np.fft.ifft(H, N))

    for m in range(int(N / 2)):
        tmp = h[m]
        h[m] = h[int(N / 2) + m]
        h[int(N / 2) + m] = tmp

    J = 128

    b = np.zeros(J + 1)
    w = Hanning_window(J + 1)
    offset = int(N / 2) - int(J / 2)
    for m in range(J + 1):
        b[m] = h[offset + m] * w[m]

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    # decay
    p1 = 75
    p2 = 0
    p3 = 120 / 12
    p4 = 20
    decay = (1 - 1 / (1 + np.exp(-(note_number - p1) / p3))) * p4 + p2

    # d
    p1 = 50
    p2 = 0
    p3 = 120 / 12
    p4 = 0.5
    d = (1 - 1 / (1 + np.exp(-(note_number - p1) / p3))) * p4 + p2

    num = np.power(10, -3 * T / decay)
    den = np.sqrt((1 - d) * (1 - d) + 2 * d * (1 - d) * np.cos((2 * np.pi * f0) / fs) + d * d)

    c = num / den
    if c > 1:
        c = 1

    D = int(T * fs - d)
    e = T * fs - d - int(T * fs - d)
    g = (1 - e) / (1 + e)

    s0 = np.zeros(length_of_s)
    s1 = np.zeros(length_of_s)
    s2 = np.zeros(length_of_s)

    np.random.seed(0)
    number_of_partial = int(20000 / f0)
    for i in range(number_of_partial):
        theta = (np.random.rand() * 2 - 1) * np.pi
        for n in range(D + 1 + J):
            s0[n] += np.sin(2 * np.pi * f0 * (i + 1) * n / fs + theta)

    mean_of_s0 = 0
    for n in range(J, D + 1 + J):
        mean_of_s0 += s0[n]

    mean_of_s0 /= D + 1
    for n in range(D + 1 + J):
        s0[n] -= mean_of_s0

    for n in range(D + 1 + J, length_of_s):
        # fractional delay
        s1[n] = -g * s1[n - 1] + g * s0[n - D] + s0[n - D - 1]

        # filter
        s2[n] = c * ((1 - d) * s1[n] + d * s1[n - 1])

        # feedback
        s0[n] += s2[n]

    # comb filter
    excitation = np.array([0.15, 0.15, 0.15, 0.18, 0.18, 0.21, 0.21, 0.21, 0.18, 0.18, 0.21, 0.21, 0.23, 0.23, 0.23, 0.21, 0.21, 0.23, 0.23, 0.23, 0.21, 0.21, 0.23, 0.23, 0.26, 0.26, 0.26, 0.25, 0.25, 0.28, 0.28, 0.28, 0.25, 0.25, 0.28, 0.28, 0.31, 0.31, 0.31, 0.37, 0.37, 0.41, 0.41, 0.41, 0.49, 0.49, 0.55])
    p = excitation[note_number - 40]
    s3 = np.zeros(length_of_s)
    for n in range(length_of_s):
        t = n - T * fs * p
        m = int(t)
        delta = t - m
        if m < 0:
            s3[n] = s0[n] - (delta * s0[m + 1 + D + 1] + (1 - delta) * s0[m + D + 1])
        else:
            s3[n] = s0[n] - (delta * s0[m + 1] + (1 - delta) * s0[m])

    # comb filter
    pickup = np.array([0.24, 0.24, 0.24, 0.29, 0.29, 0.33, 0.33, 0.33, 0.29, 0.29, 0.32, 0.32, 0.36, 0.36, 0.36, 0.32, 0.32, 0.36, 0.36, 0.36, 0.32, 0.32, 0.36, 0.36, 0.41, 0.41, 0.41, 0.38, 0.38, 0.43, 0.43, 0.43, 0.38, 0.38, 0.43, 0.43, 0.48, 0.48, 0.48, 0.57, 0.57, 0.64, 0.64, 0.64, 0.76, 0.76, 0.85])
    p = pickup[note_number - 40]
    s4 = np.zeros(length_of_s)
    for n in range(length_of_s):
        t = n - T * fs * p
        m = int(t)
        delta = t - m
        if m < 0:
            s4[n] = s3[n] - (delta * s3[m + 1 + D + 1] + (1 - delta) * s3[m + D + 1])
        else:
            s4[n] = s3[n] - (delta * s3[m + 1] + (1 - delta) * s3[m])

    s5 = np.zeros(length_of_s)
    for n in range(length_of_s):
        for m in range(J + 1):
            if n - m >= 0:
                s5[n] += b[m] * s4[n - m]

    s6 = np.zeros(length_of_s)
    for n in range(length_of_s - J):
        s6[n] = s5[J + n]

    # DC cancel
    fc = 5
    Q = 1 / np.sqrt(2)
    a, b = HPF(fs, fc, Q)
    s7 = filter(a, b, s6)
    s7 /= np.max(np.abs(s7))

    # A part
    VCF_A = np.array([0])
    VCF_D = np.array([T * 5])
    VCF_S = np.array([0])
    VCF_R = np.array([T * 5])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([f0 * 32])
    VCF_depth = np.array([f0 * 512])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]
        if vcf[n] > 20000:
            vcf[n] = 20000

    sa = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                sa[n] += b[m] * s7[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                sa[n] += -a[m] * sa[n - m]

    VCA_A = np.array([T * 1])
    VCA_D = np.array([T * 20])
    VCA_S = np.array([0])
    VCA_R = np.array([T * 20])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        sa[n] *= vca[n]

    # B part
    VCF_A = np.array([0])
    VCF_D = np.array([decay * 0.2])
    VCF_S = np.array([0])
    VCF_R = np.array([decay * 0.2])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([f0 * 8])
    VCF_depth = np.array([f0 * 128])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]
        if vcf[n] > 20000:
            vcf[n] = 20000

    sb = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                sb[n] += b[m] * s7[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                sb[n] += -a[m] * sb[n - m]

    VCA_A = np.array([T * 1])
    VCA_D = np.array([0])
    VCA_S = np.array([1])
    VCA_R = np.array([0])
    VCA_gate = np.array([duration])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        sb[n] *= vca[n]

    s8 = sa * 0.5 + sb * 0.5

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
        s8[n] *= vca[n]

    s8 *= velocity / 127 / np.max(np.abs(s8))

    return s8

def electric_bass(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    s0 = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    VCO_A = np.array([0])
    VCO_D = np.array([0])
    VCO_S = np.array([1])
    VCO_R = np.array([0])
    VCO_gate = np.array([duration])
    VCO_duration = np.array([duration])
    VCO_offset = np.array([f0])
    VCO_depth = np.array([0])

    vco = ADSR(fs, VCO_A[0], VCO_D[0], VCO_S[0], VCO_R[0], VCO_gate[0], VCO_duration[0])
    for n in range(length_of_s):
        vco[n] = VCO_offset[0] + vco[n] * VCO_depth[0]

    x = 0
    for n in range(length_of_s):
        if x < 0.125:
            s0[n] = 1
        else:
            s0[n] = -1

        delta = vco[n] / fs
        if 0 <= x and x < delta:
            t = x / delta
            d = -t * t + 2 * t - 1
            s0[n] += d
        elif 1 - delta < x and x <= 1:
            t = (x - 1) / delta
            d = t * t + 2 * t + 1
            s0[n] += d

        if 0.125 <= x and x < 0.125 + delta:
            t = (x - 0.125) / delta
            d = -t * t + 2 * t - 1
            s0[n] -= d
        elif 0.125 - delta < x and x <= 0.125:
            t = (x - 0.125) / delta
            d = t * t + 2 * t + 1
            s0[n] -= d

        x += delta
        if x >= 1:
            x -= 1

    # DC cancel
    fc = 5
    Q = 1 / np.sqrt(2)
    a, b = HPF(fs, fc, Q)
    s1 = filter(a, b, s0)

    VCF_A = np.array([0])
    VCF_D = np.array([0.2])
    VCF_S = np.array([0])
    VCF_R = np.array([0.2])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([200])
    VCF_depth = np.array([800])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]

    s2 = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                s2[n] += b[m] * s1[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                s2[n] += -a[m] * s2[n - m]

    VCA_A = np.array([0.1])
    VCA_D = np.array([8])
    VCA_S = np.array([0])
    VCA_R = np.array([0.1])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        s2[n] *= vca[n]

    # compressor
    s2 = s2 / np.max(np.abs(s2))
    threshold = 0.5
    width = 0.4
    ratio = 8
    s2 = compressor(threshold, width, ratio, s2)

    s2 *= velocity / 127 / np.max(np.abs(s2))

    return s2

def slap_bass(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    s0 = np.zeros(length_of_s)

    f = np.array([0, 4, 7, 12, 16, 20, 25, 30, 35, 41, 47, 53, 60, 67, 74, 82, 90, 99, 108, 118, 128, 139, 150, 162, 175, 188, 202, 217, 233, 250, 267, 286, 306, 326, 348, 371, 396, 421, 449, 477, 508, 540, 574, 609, 647, 687, 729, 773, 820, 869, 922, 977, 1035, 1097, 1161, 1230, 1302, 1379, 1460, 1545, 1635, 1730, 1830, 1936, 2048])
    a = np.array([[0.216749, 0.527732, 0.481251, 1.000000, 0.362505, 0.286246, 0.158858, 0.068548, 0.030996, 0.032048, 0.066160, 0.027113, 0.015777, 0.043906, 0.026962, 0.025070, 0.008796, 0.027376, 0.067058, 0.055036, 0.068607, 0.064697, 0.017552, 0.003479, 0.003391, 0.003602, 0.016313, 0.052859, 0.032530, 0.022682, 0.015054, 0.011142, 0.034442, 0.013446, 0.009187, 0.005844, 0.009808, 0.010207, 0.004706, 0.005734, 0.001149, 0.001347, 0.001229, 0.001721, 0.000350, 0.000174, 0.000566, 0.000233, 0.000157, 0.000148, 0.000155, 0.000102, 0.000072, 0.000053, 0.000048, 0.000045, 0.000032, 0.000035, 0.000031, 0.000048, 0.000047, 0.000062, 0.000070, 0.000075],
                  [0.216749, 0.527732, 0.481251, 1.000000, 0.362505, 0.286246, 0.158858, 0.068548, 0.030996, 0.032048, 0.066160, 0.027113, 0.015777, 0.043906, 0.026962, 0.025070, 0.008796, 0.027376, 0.067058, 0.055036, 0.068607, 0.064697, 0.017552, 0.003479, 0.003391, 0.003602, 0.016313, 0.052859, 0.032530, 0.022682, 0.015054, 0.011142, 0.034442, 0.013446, 0.009187, 0.005844, 0.009808, 0.010207, 0.004706, 0.005734, 0.001149, 0.001347, 0.001229, 0.001721, 0.000350, 0.000174, 0.000566, 0.000233, 0.000157, 0.000148, 0.000155, 0.000102, 0.000072, 0.000053, 0.000048, 0.000045, 0.000032, 0.000035, 0.000031, 0.000048, 0.000047, 0.000062, 0.000070, 0.000075],
                  [0.216749, 0.527732, 0.481251, 1.000000, 0.362505, 0.286246, 0.158858, 0.068548, 0.030996, 0.032048, 0.066160, 0.027113, 0.015777, 0.043906, 0.026962, 0.025070, 0.008796, 0.027376, 0.067058, 0.055036, 0.068607, 0.064697, 0.017552, 0.003479, 0.003391, 0.003602, 0.016313, 0.052859, 0.032530, 0.022682, 0.015054, 0.011142, 0.034442, 0.013446, 0.009187, 0.005844, 0.009808, 0.010207, 0.004706, 0.005734, 0.001149, 0.001347, 0.001229, 0.001721, 0.000350, 0.000174, 0.000566, 0.000233, 0.000157, 0.000148, 0.000155, 0.000102, 0.000072, 0.000053, 0.000048, 0.000045, 0.000032, 0.000035, 0.000031, 0.000048, 0.000047, 0.000062, 0.000070, 0.000075],
                  [0.167128, 0.713017, 1.000000, 0.563959, 0.660722, 0.378607, 0.111434, 0.049276, 0.033340, 0.067471, 0.052205, 0.048170, 0.010926, 0.064852, 0.030572, 0.020883, 0.007620, 0.013092, 0.021944, 0.025286, 0.104344, 0.166854, 0.101496, 0.020612, 0.007233, 0.004343, 0.008497, 0.012514, 0.004724, 0.007894, 0.009868, 0.016830, 0.031454, 0.011797, 0.005178, 0.003650, 0.003741, 0.006266, 0.003026, 0.005221, 0.001905, 0.001083, 0.000869, 0.001154, 0.000269, 0.000211, 0.000579, 0.000165, 0.000116, 0.000105, 0.000110, 0.000078, 0.000056, 0.000046, 0.000048, 0.000040, 0.000037, 0.000044, 0.000046, 0.000058, 0.000062, 0.000074, 0.000091, 0.000097],
                  [0.167128, 0.713017, 1.000000, 0.563959, 0.660722, 0.378607, 0.111434, 0.049276, 0.033340, 0.067471, 0.052205, 0.048170, 0.010926, 0.064852, 0.030572, 0.020883, 0.007620, 0.013092, 0.021944, 0.025286, 0.104344, 0.166854, 0.101496, 0.020612, 0.007233, 0.004343, 0.008497, 0.012514, 0.004724, 0.007894, 0.009868, 0.016830, 0.031454, 0.011797, 0.005178, 0.003650, 0.003741, 0.006266, 0.003026, 0.005221, 0.001905, 0.001083, 0.000869, 0.001154, 0.000269, 0.000211, 0.000579, 0.000165, 0.000116, 0.000105, 0.000110, 0.000078, 0.000056, 0.000046, 0.000048, 0.000040, 0.000037, 0.000044, 0.000046, 0.000058, 0.000062, 0.000074, 0.000091, 0.000097],
                  [0.167128, 0.713017, 1.000000, 0.563959, 0.660722, 0.378607, 0.111434, 0.049276, 0.033340, 0.067471, 0.052205, 0.048170, 0.010926, 0.064852, 0.030572, 0.020883, 0.007620, 0.013092, 0.021944, 0.025286, 0.104344, 0.166854, 0.101496, 0.020612, 0.007233, 0.004343, 0.008497, 0.012514, 0.004724, 0.007894, 0.009868, 0.016830, 0.031454, 0.011797, 0.005178, 0.003650, 0.003741, 0.006266, 0.003026, 0.005221, 0.001905, 0.001083, 0.000869, 0.001154, 0.000269, 0.000211, 0.000579, 0.000165, 0.000116, 0.000105, 0.000110, 0.000078, 0.000056, 0.000046, 0.000048, 0.000040, 0.000037, 0.000044, 0.000046, 0.000058, 0.000062, 0.000074, 0.000091, 0.000097],
                  [0.167128, 0.713017, 1.000000, 0.563959, 0.660722, 0.378607, 0.111434, 0.049276, 0.033340, 0.067471, 0.052205, 0.048170, 0.010926, 0.064852, 0.030572, 0.020883, 0.007620, 0.013092, 0.021944, 0.025286, 0.104344, 0.166854, 0.101496, 0.020612, 0.007233, 0.004343, 0.008497, 0.012514, 0.004724, 0.007894, 0.009868, 0.016830, 0.031454, 0.011797, 0.005178, 0.003650, 0.003741, 0.006266, 0.003026, 0.005221, 0.001905, 0.001083, 0.000869, 0.001154, 0.000269, 0.000211, 0.000579, 0.000165, 0.000116, 0.000105, 0.000110, 0.000078, 0.000056, 0.000046, 0.000048, 0.000040, 0.000037, 0.000044, 0.000046, 0.000058, 0.000062, 0.000074, 0.000091, 0.000097],
                  [0.167128, 0.713017, 1.000000, 0.563959, 0.660722, 0.378607, 0.111434, 0.049276, 0.033340, 0.067471, 0.052205, 0.048170, 0.010926, 0.064852, 0.030572, 0.020883, 0.007620, 0.013092, 0.021944, 0.025286, 0.104344, 0.166854, 0.101496, 0.020612, 0.007233, 0.004343, 0.008497, 0.012514, 0.004724, 0.007894, 0.009868, 0.016830, 0.031454, 0.011797, 0.005178, 0.003650, 0.003741, 0.006266, 0.003026, 0.005221, 0.001905, 0.001083, 0.000869, 0.001154, 0.000269, 0.000211, 0.000579, 0.000165, 0.000116, 0.000105, 0.000110, 0.000078, 0.000056, 0.000046, 0.000048, 0.000040, 0.000037, 0.000044, 0.000046, 0.000058, 0.000062, 0.000074, 0.000091, 0.000097],
                  [0.332502, 0.553390, 0.782657, 0.649905, 0.717403, 1.000000, 0.710641, 0.302835, 0.078330, 0.073053, 0.111022, 0.060015, 0.112122, 0.083925, 0.050413, 0.031002, 0.115716, 0.090607, 0.080204, 0.019413, 0.018352, 0.039550, 0.056375, 0.070711, 0.289227, 0.250648, 0.121603, 0.031917, 0.016564, 0.014221, 0.038728, 0.049885, 0.025767, 0.012300, 0.018764, 0.039352, 0.045976, 0.023917, 0.009403, 0.007502, 0.007245, 0.008309, 0.003832, 0.002287, 0.001369, 0.001443, 0.000728, 0.000434, 0.000408, 0.000322, 0.000293, 0.000339, 0.000121, 0.000182, 0.000301, 0.000145, 0.000126, 0.000106, 0.000155, 0.000147, 0.000188, 0.000203, 0.000208, 0.000242],
                  [0.332502, 0.553390, 0.782657, 0.649905, 0.717403, 1.000000, 0.710641, 0.302835, 0.078330, 0.073053, 0.111022, 0.060015, 0.112122, 0.083925, 0.050413, 0.031002, 0.115716, 0.090607, 0.080204, 0.019413, 0.018352, 0.039550, 0.056375, 0.070711, 0.289227, 0.250648, 0.121603, 0.031917, 0.016564, 0.014221, 0.038728, 0.049885, 0.025767, 0.012300, 0.018764, 0.039352, 0.045976, 0.023917, 0.009403, 0.007502, 0.007245, 0.008309, 0.003832, 0.002287, 0.001369, 0.001443, 0.000728, 0.000434, 0.000408, 0.000322, 0.000293, 0.000339, 0.000121, 0.000182, 0.000301, 0.000145, 0.000126, 0.000106, 0.000155, 0.000147, 0.000188, 0.000203, 0.000208, 0.000242],
                  [0.332502, 0.553390, 0.782657, 0.649905, 0.717403, 1.000000, 0.710641, 0.302835, 0.078330, 0.073053, 0.111022, 0.060015, 0.112122, 0.083925, 0.050413, 0.031002, 0.115716, 0.090607, 0.080204, 0.019413, 0.018352, 0.039550, 0.056375, 0.070711, 0.289227, 0.250648, 0.121603, 0.031917, 0.016564, 0.014221, 0.038728, 0.049885, 0.025767, 0.012300, 0.018764, 0.039352, 0.045976, 0.023917, 0.009403, 0.007502, 0.007245, 0.008309, 0.003832, 0.002287, 0.001369, 0.001443, 0.000728, 0.000434, 0.000408, 0.000322, 0.000293, 0.000339, 0.000121, 0.000182, 0.000301, 0.000145, 0.000126, 0.000106, 0.000155, 0.000147, 0.000188, 0.000203, 0.000208, 0.000242],
                  [0.332502, 0.553390, 0.782657, 0.649905, 0.717403, 1.000000, 0.710641, 0.302835, 0.078330, 0.073053, 0.111022, 0.060015, 0.112122, 0.083925, 0.050413, 0.031002, 0.115716, 0.090607, 0.080204, 0.019413, 0.018352, 0.039550, 0.056375, 0.070711, 0.289227, 0.250648, 0.121603, 0.031917, 0.016564, 0.014221, 0.038728, 0.049885, 0.025767, 0.012300, 0.018764, 0.039352, 0.045976, 0.023917, 0.009403, 0.007502, 0.007245, 0.008309, 0.003832, 0.002287, 0.001369, 0.001443, 0.000728, 0.000434, 0.000408, 0.000322, 0.000293, 0.000339, 0.000121, 0.000182, 0.000301, 0.000145, 0.000126, 0.000106, 0.000155, 0.000147, 0.000188, 0.000203, 0.000208, 0.000242],
                  [0.332502, 0.553390, 0.782657, 0.649905, 0.717403, 1.000000, 0.710641, 0.302835, 0.078330, 0.073053, 0.111022, 0.060015, 0.112122, 0.083925, 0.050413, 0.031002, 0.115716, 0.090607, 0.080204, 0.019413, 0.018352, 0.039550, 0.056375, 0.070711, 0.289227, 0.250648, 0.121603, 0.031917, 0.016564, 0.014221, 0.038728, 0.049885, 0.025767, 0.012300, 0.018764, 0.039352, 0.045976, 0.023917, 0.009403, 0.007502, 0.007245, 0.008309, 0.003832, 0.002287, 0.001369, 0.001443, 0.000728, 0.000434, 0.000408, 0.000322, 0.000293, 0.000339, 0.000121, 0.000182, 0.000301, 0.000145, 0.000126, 0.000106, 0.000155, 0.000147, 0.000188, 0.000203, 0.000208, 0.000242],
                  [0.332502, 0.553390, 0.782657, 0.649905, 0.717403, 1.000000, 0.710641, 0.302835, 0.078330, 0.073053, 0.111022, 0.060015, 0.112122, 0.083925, 0.050413, 0.031002, 0.115716, 0.090607, 0.080204, 0.019413, 0.018352, 0.039550, 0.056375, 0.070711, 0.289227, 0.250648, 0.121603, 0.031917, 0.016564, 0.014221, 0.038728, 0.049885, 0.025767, 0.012300, 0.018764, 0.039352, 0.045976, 0.023917, 0.009403, 0.007502, 0.007245, 0.008309, 0.003832, 0.002287, 0.001369, 0.001443, 0.000728, 0.000434, 0.000408, 0.000322, 0.000293, 0.000339, 0.000121, 0.000182, 0.000301, 0.000145, 0.000126, 0.000106, 0.000155, 0.000147, 0.000188, 0.000203, 0.000208, 0.000242],
                  [0.332502, 0.553390, 0.782657, 0.649905, 0.717403, 1.000000, 0.710641, 0.302835, 0.078330, 0.073053, 0.111022, 0.060015, 0.112122, 0.083925, 0.050413, 0.031002, 0.115716, 0.090607, 0.080204, 0.019413, 0.018352, 0.039550, 0.056375, 0.070711, 0.289227, 0.250648, 0.121603, 0.031917, 0.016564, 0.014221, 0.038728, 0.049885, 0.025767, 0.012300, 0.018764, 0.039352, 0.045976, 0.023917, 0.009403, 0.007502, 0.007245, 0.008309, 0.003832, 0.002287, 0.001369, 0.001443, 0.000728, 0.000434, 0.000408, 0.000322, 0.000293, 0.000339, 0.000121, 0.000182, 0.000301, 0.000145, 0.000126, 0.000106, 0.000155, 0.000147, 0.000188, 0.000203, 0.000208, 0.000242],
                  [0.163850, 0.300053, 0.680352, 0.875371, 0.702252, 0.713800, 1.000000, 0.858163, 0.444088, 0.327635, 0.194284, 0.058759, 0.091168, 0.253788, 0.241265, 0.338328, 0.443466, 0.323739, 0.137998, 0.245859, 0.338991, 0.364212, 0.195182, 0.050442, 0.060808, 0.332741, 0.598163, 0.338551, 0.280761, 0.431791, 0.220589, 0.079874, 0.119520, 0.274049, 0.165020, 0.369775, 0.153168, 0.056825, 0.150436, 0.107181, 0.098430, 0.035936, 0.018287, 0.026828, 0.041121, 0.039838, 0.007296, 0.006483, 0.010059, 0.004678, 0.002086, 0.002242, 0.002257, 0.001810, 0.002369, 0.002393, 0.001447, 0.001483, 0.002548, 0.000692, 0.000728, 0.000397, 0.000305, 0.000262],
                  [0.163850, 0.300053, 0.680352, 0.875371, 0.702252, 0.713800, 1.000000, 0.858163, 0.444088, 0.327635, 0.194284, 0.058759, 0.091168, 0.253788, 0.241265, 0.338328, 0.443466, 0.323739, 0.137998, 0.245859, 0.338991, 0.364212, 0.195182, 0.050442, 0.060808, 0.332741, 0.598163, 0.338551, 0.280761, 0.431791, 0.220589, 0.079874, 0.119520, 0.274049, 0.165020, 0.369775, 0.153168, 0.056825, 0.150436, 0.107181, 0.098430, 0.035936, 0.018287, 0.026828, 0.041121, 0.039838, 0.007296, 0.006483, 0.010059, 0.004678, 0.002086, 0.002242, 0.002257, 0.001810, 0.002369, 0.002393, 0.001447, 0.001483, 0.002548, 0.000692, 0.000728, 0.000397, 0.000305, 0.000262],
                  [0.163850, 0.300053, 0.680352, 0.875371, 0.702252, 0.713800, 1.000000, 0.858163, 0.444088, 0.327635, 0.194284, 0.058759, 0.091168, 0.253788, 0.241265, 0.338328, 0.443466, 0.323739, 0.137998, 0.245859, 0.338991, 0.364212, 0.195182, 0.050442, 0.060808, 0.332741, 0.598163, 0.338551, 0.280761, 0.431791, 0.220589, 0.079874, 0.119520, 0.274049, 0.165020, 0.369775, 0.153168, 0.056825, 0.150436, 0.107181, 0.098430, 0.035936, 0.018287, 0.026828, 0.041121, 0.039838, 0.007296, 0.006483, 0.010059, 0.004678, 0.002086, 0.002242, 0.002257, 0.001810, 0.002369, 0.002393, 0.001447, 0.001483, 0.002548, 0.000692, 0.000728, 0.000397, 0.000305, 0.000262],
                  [0.163850, 0.300053, 0.680352, 0.875371, 0.702252, 0.713800, 1.000000, 0.858163, 0.444088, 0.327635, 0.194284, 0.058759, 0.091168, 0.253788, 0.241265, 0.338328, 0.443466, 0.323739, 0.137998, 0.245859, 0.338991, 0.364212, 0.195182, 0.050442, 0.060808, 0.332741, 0.598163, 0.338551, 0.280761, 0.431791, 0.220589, 0.079874, 0.119520, 0.274049, 0.165020, 0.369775, 0.153168, 0.056825, 0.150436, 0.107181, 0.098430, 0.035936, 0.018287, 0.026828, 0.041121, 0.039838, 0.007296, 0.006483, 0.010059, 0.004678, 0.002086, 0.002242, 0.002257, 0.001810, 0.002369, 0.002393, 0.001447, 0.001483, 0.002548, 0.000692, 0.000728, 0.000397, 0.000305, 0.000262],
                  [0.163850, 0.300053, 0.680352, 0.875371, 0.702252, 0.713800, 1.000000, 0.858163, 0.444088, 0.327635, 0.194284, 0.058759, 0.091168, 0.253788, 0.241265, 0.338328, 0.443466, 0.323739, 0.137998, 0.245859, 0.338991, 0.364212, 0.195182, 0.050442, 0.060808, 0.332741, 0.598163, 0.338551, 0.280761, 0.431791, 0.220589, 0.079874, 0.119520, 0.274049, 0.165020, 0.369775, 0.153168, 0.056825, 0.150436, 0.107181, 0.098430, 0.035936, 0.018287, 0.026828, 0.041121, 0.039838, 0.007296, 0.006483, 0.010059, 0.004678, 0.002086, 0.002242, 0.002257, 0.001810, 0.002369, 0.002393, 0.001447, 0.001483, 0.002548, 0.000692, 0.000728, 0.000397, 0.000305, 0.000262],
                  [0.210100, 0.249915, 0.343372, 0.447735, 0.474780, 0.455342, 0.484928, 0.629478, 0.791594, 0.677484, 0.432989, 0.317359, 0.271415, 0.182453, 0.123853, 0.161507, 0.235425, 0.264377, 0.394282, 0.452391, 0.343762, 0.252003, 0.218224, 0.378872, 0.513294, 0.357329, 0.146624, 0.059900, 0.089342, 0.326059, 0.655117, 0.601106, 0.425685, 1.000000, 0.811598, 0.261500, 0.159016, 0.314296, 0.169880, 0.142314, 0.228116, 0.070548, 0.069056, 0.101868, 0.061678, 0.086747, 0.016611, 0.019249, 0.016456, 0.037542, 0.013490, 0.005286, 0.003444, 0.006690, 0.001834, 0.003003, 0.002195, 0.001591, 0.000863, 0.001437, 0.001593, 0.001077, 0.000859, 0.000857],
                  [0.210100, 0.249915, 0.343372, 0.447735, 0.474780, 0.455342, 0.484928, 0.629478, 0.791594, 0.677484, 0.432989, 0.317359, 0.271415, 0.182453, 0.123853, 0.161507, 0.235425, 0.264377, 0.394282, 0.452391, 0.343762, 0.252003, 0.218224, 0.378872, 0.513294, 0.357329, 0.146624, 0.059900, 0.089342, 0.326059, 0.655117, 0.601106, 0.425685, 1.000000, 0.811598, 0.261500, 0.159016, 0.314296, 0.169880, 0.142314, 0.228116, 0.070548, 0.069056, 0.101868, 0.061678, 0.086747, 0.016611, 0.019249, 0.016456, 0.037542, 0.013490, 0.005286, 0.003444, 0.006690, 0.001834, 0.003003, 0.002195, 0.001591, 0.000863, 0.001437, 0.001593, 0.001077, 0.000859, 0.000857],
                  [0.210100, 0.249915, 0.343372, 0.447735, 0.474780, 0.455342, 0.484928, 0.629478, 0.791594, 0.677484, 0.432989, 0.317359, 0.271415, 0.182453, 0.123853, 0.161507, 0.235425, 0.264377, 0.394282, 0.452391, 0.343762, 0.252003, 0.218224, 0.378872, 0.513294, 0.357329, 0.146624, 0.059900, 0.089342, 0.326059, 0.655117, 0.601106, 0.425685, 1.000000, 0.811598, 0.261500, 0.159016, 0.314296, 0.169880, 0.142314, 0.228116, 0.070548, 0.069056, 0.101868, 0.061678, 0.086747, 0.016611, 0.019249, 0.016456, 0.037542, 0.013490, 0.005286, 0.003444, 0.006690, 0.001834, 0.003003, 0.002195, 0.001591, 0.000863, 0.001437, 0.001593, 0.001077, 0.000859, 0.000857],
                  [0.210100, 0.249915, 0.343372, 0.447735, 0.474780, 0.455342, 0.484928, 0.629478, 0.791594, 0.677484, 0.432989, 0.317359, 0.271415, 0.182453, 0.123853, 0.161507, 0.235425, 0.264377, 0.394282, 0.452391, 0.343762, 0.252003, 0.218224, 0.378872, 0.513294, 0.357329, 0.146624, 0.059900, 0.089342, 0.326059, 0.655117, 0.601106, 0.425685, 1.000000, 0.811598, 0.261500, 0.159016, 0.314296, 0.169880, 0.142314, 0.228116, 0.070548, 0.069056, 0.101868, 0.061678, 0.086747, 0.016611, 0.019249, 0.016456, 0.037542, 0.013490, 0.005286, 0.003444, 0.006690, 0.001834, 0.003003, 0.002195, 0.001591, 0.000863, 0.001437, 0.001593, 0.001077, 0.000859, 0.000857],
                  [0.366670, 0.355505, 0.333573, 0.311790, 0.307966, 0.334937, 0.415253, 0.545689, 0.667658, 0.624734, 0.427093, 0.253682, 0.182229, 0.175845, 0.167145, 0.138552, 0.167758, 0.408113, 0.800934, 0.680389, 0.480586, 0.425365, 0.365095, 0.630317, 1.000000, 0.541301, 0.249082, 0.111379, 0.096617, 0.248926, 0.323736, 0.476434, 0.474263, 0.732650, 0.912361, 0.382800, 0.334573, 0.750182, 0.494614, 0.410149, 0.338613, 0.114981, 0.052968, 0.073716, 0.069490, 0.121999, 0.033201, 0.034570, 0.027467, 0.050323, 0.015531, 0.006893, 0.007587, 0.015499, 0.004683, 0.003717, 0.002956, 0.002174, 0.000920, 0.002010, 0.002385, 0.001256, 0.000899, 0.001231],
                  [0.366670, 0.355505, 0.333573, 0.311790, 0.307966, 0.334937, 0.415253, 0.545689, 0.667658, 0.624734, 0.427093, 0.253682, 0.182229, 0.175845, 0.167145, 0.138552, 0.167758, 0.408113, 0.800934, 0.680389, 0.480586, 0.425365, 0.365095, 0.630317, 1.000000, 0.541301, 0.249082, 0.111379, 0.096617, 0.248926, 0.323736, 0.476434, 0.474263, 0.732650, 0.912361, 0.382800, 0.334573, 0.750182, 0.494614, 0.410149, 0.338613, 0.114981, 0.052968, 0.073716, 0.069490, 0.121999, 0.033201, 0.034570, 0.027467, 0.050323, 0.015531, 0.006893, 0.007587, 0.015499, 0.004683, 0.003717, 0.002956, 0.002174, 0.000920, 0.002010, 0.002385, 0.001256, 0.000899, 0.001231],
                  [0.366670, 0.355505, 0.333573, 0.311790, 0.307966, 0.334937, 0.415253, 0.545689, 0.667658, 0.624734, 0.427093, 0.253682, 0.182229, 0.175845, 0.167145, 0.138552, 0.167758, 0.408113, 0.800934, 0.680389, 0.480586, 0.425365, 0.365095, 0.630317, 1.000000, 0.541301, 0.249082, 0.111379, 0.096617, 0.248926, 0.323736, 0.476434, 0.474263, 0.732650, 0.912361, 0.382800, 0.334573, 0.750182, 0.494614, 0.410149, 0.338613, 0.114981, 0.052968, 0.073716, 0.069490, 0.121999, 0.033201, 0.034570, 0.027467, 0.050323, 0.015531, 0.006893, 0.007587, 0.015499, 0.004683, 0.003717, 0.002956, 0.002174, 0.000920, 0.002010, 0.002385, 0.001256, 0.000899, 0.001231],
                  [0.366670, 0.355505, 0.333573, 0.311790, 0.307966, 0.334937, 0.415253, 0.545689, 0.667658, 0.624734, 0.427093, 0.253682, 0.182229, 0.175845, 0.167145, 0.138552, 0.167758, 0.408113, 0.800934, 0.680389, 0.480586, 0.425365, 0.365095, 0.630317, 1.000000, 0.541301, 0.249082, 0.111379, 0.096617, 0.248926, 0.323736, 0.476434, 0.474263, 0.732650, 0.912361, 0.382800, 0.334573, 0.750182, 0.494614, 0.410149, 0.338613, 0.114981, 0.052968, 0.073716, 0.069490, 0.121999, 0.033201, 0.034570, 0.027467, 0.050323, 0.015531, 0.006893, 0.007587, 0.015499, 0.004683, 0.003717, 0.002956, 0.002174, 0.000920, 0.002010, 0.002385, 0.001256, 0.000899, 0.001231]])

    N = 4096
    number_of_band = 64

    H = np.zeros(N)
    for band in range(number_of_band):
        fL = f[band]
        fH = f[band + 1]
        for k in range(fL, fH):
            H[k] = a[note_number - 28][band]

    for k in range(1, int(N / 2)):
        H[N - k] = H[k]

    h = np.real(np.fft.ifft(H, N))

    for m in range(int(N / 2)):
        tmp = h[m]
        h[m] = h[int(N / 2) + m]
        h[int(N / 2) + m] = tmp

    J = 128

    b = np.zeros(J + 1)
    w = Hanning_window(J + 1)
    offset = int(N / 2) - int(J / 2)
    for m in range(J + 1):
        b[m] = h[offset + m] * w[m]

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    # decay
    p1 = 55
    p2 = 0
    p3 = 120 / 12
    p4 = 20
    decay = (1 - 1 / (1 + np.exp(-(note_number - p1) / p3))) * p4 + p2

    # d
    p1 = 60
    p2 = 0
    p3 = 120 / 12
    p4 = 0.5
    d = (1 - 1 / (1 + np.exp(-(note_number - p1) / p3))) * p4 + p2

    num = np.power(10, -3 * T / decay)
    den = np.sqrt((1 - d) * (1 - d) + 2 * d * (1 - d) * np.cos((2 * np.pi * f0) / fs) + d * d)

    c = num / den
    if c > 1:
        c = 1

    D = int(T * fs - d)
    e = T * fs - d - int(T * fs - d)
    g = (1 - e) / (1 + e)

    s0 = np.zeros(length_of_s)
    s1 = np.zeros(length_of_s)
    s2 = np.zeros(length_of_s)

    np.random.seed(0)
    number_of_partial = int(20000 / f0)
    for i in range(number_of_partial):
        theta = (np.random.rand() * 2 - 1) * np.pi
        for n in range(D + 1 + J):
            s0[n] += np.sin(2 * np.pi * f0 * (i + 1) * n / fs + theta)

    mean_of_s0 = 0
    for n in range(J, D + 1 + J):
        mean_of_s0 += s0[n]

    mean_of_s0 /= D + 1
    for n in range(D + 1 + J):
        s0[n] -= mean_of_s0

    for n in range(D + 1 + J, length_of_s):
        # fractional delay
        s1[n] = -g * s1[n - 1] + g * s0[n - D] + s0[n - D - 1]

        # filter
        s2[n] = c * ((1 - d) * s1[n] + d * s1[n - 1])

        # feedback
        s0[n] += s2[n]

    # comb filter
    excitation = np.array([0.3, 0.3, 0.3, 0.36, 0.36, 0.4, 0.4, 0.4, 0.36, 0.36, 0.4, 0.4, 0.45, 0.45, 0.45, 0.34, 0.34, 0.38, 0.38, 0.38, 0.34, 0.34, 0.38, 0.38, 0.43, 0.43, 0.43, 0.51])
    p = excitation[note_number - 28]
    s3 = np.zeros(length_of_s)
    for n in range(length_of_s):
        t = n - T * fs * p
        m = int(t)
        delta = t - m
        if m < 0:
            s3[n] = s0[n] - (delta * s0[m + 1 + D + 1] + (1 - delta) * s0[m + D + 1])
        else:
            s3[n] = s0[n] - (delta * s0[m + 1] + (1 - delta) * s0[m])

    # comb filter
    pickup = np.array([0.19, 0.19, 0.19, 0.22, 0.22, 0.25, 0.25, 0.25, 0.22, 0.22, 0.25, 0.25, 0.28, 0.28, 0.28, 0.25, 0.25, 0.28, 0.28, 0.28, 0.25, 0.25, 0.28, 0.28, 0.31, 0.31, 0.31, 0.37])
    p = pickup[note_number - 28]
    s4 = np.zeros(length_of_s)
    for n in range(length_of_s):
        t = n - T * fs * p
        m = int(t)
        delta = t - m
        if m < 0:
            s4[n] = s3[n] - (delta * s3[m + 1 + D + 1] + (1 - delta) * s3[m + D + 1])
        else:
            s4[n] = s3[n] - (delta * s3[m + 1] + (1 - delta) * s3[m])

    s5 = np.zeros(length_of_s)
    for n in range(length_of_s):
        for m in range(J + 1):
            if n - m >= 0:
                s5[n] += b[m] * s4[n - m]

    s6 = np.zeros(length_of_s)
    for n in range(length_of_s - J):
        s6[n] = s5[J + n]

    # DC cancel
    fc = 5
    Q = 1 / np.sqrt(2)
    a, b = HPF(fs, fc, Q)
    s7 = filter(a, b, s6)
    s7 /= np.max(np.abs(s7))

    # A part
    VCF_A = np.array([0])
    VCF_D = np.array([T * 5])
    VCF_S = np.array([0])
    VCF_R = np.array([T * 5])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([f0 * 32])
    VCF_depth = np.array([f0 * 512])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]
        if vcf[n] > 20000:
            vcf[n] = 20000

    sa = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                sa[n] += b[m] * s7[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                sa[n] += -a[m] * sa[n - m]

    VCA_A = np.array([T * 0.1])
    VCA_D = np.array([T * 10])
    VCA_S = np.array([0])
    VCA_R = np.array([T * 10])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        sa[n] *= vca[n]

    # B part
    VCF_A = np.array([0])
    VCF_D = np.array([decay * 0.1])
    VCF_S = np.array([0])
    VCF_R = np.array([decay * 0.1])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([f0 * 8])
    VCF_depth = np.array([f0 * 128])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]
        if vcf[n] > 20000:
            vcf[n] = 20000

    sb = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                sb[n] += b[m] * s7[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                sb[n] += -a[m] * sb[n - m]

    VCA_A = np.array([T * 0.1])
    VCA_D = np.array([0])
    VCA_S = np.array([1])
    VCA_R = np.array([0])
    VCA_gate = np.array([duration])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        sb[n] *= vca[n]

    s8 = sa * 0.7 + sb * 0.3

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
        s8[n] *= vca[n]

    s8 *= velocity / 127 / np.max(np.abs(s8))

    return s8

def pipe_organ(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    s = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    number_of_partial = 15

    a = np.array([0.3, 0.6, 0.3, 0.85, 0.25, 0.5, 0.95, 0.25, 0.5, 0.15, 0.95, 0, 0.25, 0, 0.5])

    VCO_A = np.repeat(0, number_of_partial)
    VCO_D = np.repeat(0, number_of_partial)
    VCO_S = np.repeat(1, number_of_partial)
    VCO_R = np.repeat(0, number_of_partial)
    VCO_gate = np.repeat(duration, number_of_partial)
    VCO_duration = np.repeat(duration, number_of_partial)
    VCO_offset = np.array([0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) * f0
    VCO_depth = np.repeat(0, number_of_partial)

    p1 = 0
    p2 = -0.05
    p3 = 8000 / 12
    p4 = 0.1
    VCA_delay = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_delay[i] = 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3)) * p4 + p2

    for i in range(1, number_of_partial):
        VCA_delay[i] = VCA_delay[i] - VCA_delay[0]

    VCA_delay[0] = 0

    p1 = 0
    p2 = 0.05
    p3 = 8000 / 12
    p4 = 0.1
    VCA_A = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_A[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA_S = np.repeat(1, number_of_partial)

    p1 = 0
    p2 = 0.2
    p3 = 8000 / 12
    p4 = 0.4
    VCA_R = np.zeros(number_of_partial)
    for i in range(number_of_partial):
        VCA_R[i] = (1 - 1 / (1 + np.exp(-(VCO_offset[i] - p1) / p3))) * p4 + p2

    VCA_gate = np.repeat(gate, number_of_partial)
    VCA_duration = np.repeat(duration, number_of_partial)
    VCA_offset = np.repeat(0, number_of_partial)
    VCA_depth = a

    np.random.seed(0)

    for i in range(number_of_partial):
        vco = ADSR(fs, VCO_A[i], VCO_D[i], VCO_S[i], VCO_R[i], VCO_gate[i], VCO_duration[i])
        for n in range(length_of_s):
            vco[n] = VCO_offset[i] + vco[n] * VCO_depth[i]

        w = np.zeros(length_of_s)
        for n in range(length_of_s):
            w[n] = np.random.rand() * 2 - 1

        fc = 40
        Q = 1 / np.sqrt(2)
        a, b = LPF(fs, fc, Q)
        jitter = filter(a, b, w)
        jitter /= np.max(np.abs(jitter))

        p1 = 108
        p2 = 0.2
        p3 = 120 / 12
        p4 = 4
        jitter_depth = 1 / (1 + np.exp(-(note_number - p1) / p3)) * p4 + p2

        for n in range(length_of_s):
            vco[n] += jitter[n] * jitter_depth

        if np.max(vco) < 20000:
            p = np.zeros(length_of_s)
            x = 0
            for n in range(length_of_s):
                p[n] = np.sin(2 * np.pi * x)
                delta = vco[n] / fs
                x += delta
                if x >= 1:
                    x -= 1

            vca = cosine_envelope(fs, VCA_delay[i], VCA_A[i], VCA_S[i], VCA_R[i], VCA_gate[i], VCA_duration[i])
            for n in range(length_of_s):
                vca[n] = VCA_offset[i] + vca[n] * VCA_depth[i]

            w = np.zeros(length_of_s)
            for n in range(length_of_s):
                w[n] = np.random.rand() * 2 - 1

            fc = 40
            Q = 1 / np.sqrt(2)
            a, b = LPF(fs, fc, Q)
            shimmer = filter(a, b, w)
            shimmer /= np.max(np.abs(shimmer))

            p1 = 1
            p2 = -0.05
            p3 = 100 / 12
            p4 = 0.2
            shimmer_depth = 1 / (1 + np.exp(-(VCO_offset[i] / f0 - p1) / p3)) * p4 + p2

            for n in range(length_of_s):
                vca[n] *= 1 + shimmer[n] * shimmer_depth
                if vca[n] < 0:
                    vca[n] = 0

            for n in range(length_of_s):
                s[n] += p[n] * vca[n]

    s *= velocity / 127 / np.max(np.abs(s))

    return s

def reed_organ(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    s0 = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    VCO_A = np.array([0])
    VCO_D = np.array([0])
    VCO_S = np.array([1])
    VCO_R = np.array([0])
    VCO_gate = np.array([duration])
    VCO_duration = np.array([duration])
    VCO_offset = np.array([f0])
    VCO_depth = np.array([0])

    vco = ADSR(fs, VCO_A[0], VCO_D[0], VCO_S[0], VCO_R[0], VCO_gate[0], VCO_duration[0])
    for n in range(length_of_s):
        vco[n] = VCO_offset[0] + vco[n] * VCO_depth[0]

    x = 0
    for n in range(length_of_s):
        s0[n] = -2 * x + 1
        delta = vco[n] / fs
        if 0 <= x and x < delta:
            t = x / delta
            d = -t * t + 2 * t - 1
            s0[n] += d
        elif 1 - delta < x and x <= 1:
            t = (x - 1) / delta
            d = t * t + 2 * t + 1
            s0[n] += d

        x += delta
        if x >= 1:
            x -= 1

    VCF_A = np.array([0])
    VCF_D = np.array([0])
    VCF_S = np.array([1])
    VCF_R = np.array([0])
    VCF_gate = np.array([duration])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([500])
    VCF_depth = np.array([0])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]

    s1 = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                s1[n] += b[m] * s0[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                s1[n] += -a[m] * s1[n - m]

    VCA_A = np.array([0.5])
    VCA_D = np.array([0])
    VCA_S = np.array([1])
    VCA_R = np.array([0.2])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        s1[n] *= vca[n]

    s1 *= velocity / 127 / np.max(np.abs(s1))

    return s1

def harpsichord(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    s0 = np.zeros(length_of_s)

    f = np.array([0, 4, 7, 12, 16, 20, 25, 30, 35, 41, 47, 53, 60, 67, 74, 82, 90, 99, 108, 118, 128, 139, 150, 162, 175, 188, 202, 217, 233, 250, 267, 286, 306, 326, 348, 371, 396, 421, 449, 477, 508, 540, 574, 609, 647, 687, 729, 773, 820, 869, 922, 977, 1035, 1097, 1161, 1230, 1302, 1379, 1460, 1545, 1635, 1730, 1830, 1936, 2048])
    a = np.array([[0.041661, 0.042592, 0.072654, 0.204266, 0.568685, 1.000000, 0.419720, 0.220019, 0.495519, 0.199545, 0.070834, 0.164315, 0.147530, 0.154276, 0.116741, 0.121176, 0.091007, 0.067478, 0.042546, 0.073123, 0.068860, 0.053839, 0.029921, 0.029768, 0.029465, 0.061012, 0.045171, 0.090820, 0.070394, 0.054073, 0.063202, 0.033421, 0.028364, 0.036574, 0.045538, 0.051262, 0.038076, 0.033794, 0.020677, 0.070303, 0.063244, 0.060428, 0.038189, 0.038606, 0.042050, 0.038997, 0.035323, 0.029665, 0.033907, 0.038288, 0.017471, 0.031069, 0.030238, 0.025735, 0.020350, 0.014198, 0.012919, 0.012094, 0.010160, 0.010514, 0.005527, 0.006399, 0.004232, 0.002667],
                  [0.041661, 0.042592, 0.072654, 0.204266, 0.568685, 1.000000, 0.419720, 0.220019, 0.495519, 0.199545, 0.070834, 0.164315, 0.147530, 0.154276, 0.116741, 0.121176, 0.091007, 0.067478, 0.042546, 0.073123, 0.068860, 0.053839, 0.029921, 0.029768, 0.029465, 0.061012, 0.045171, 0.090820, 0.070394, 0.054073, 0.063202, 0.033421, 0.028364, 0.036574, 0.045538, 0.051262, 0.038076, 0.033794, 0.020677, 0.070303, 0.063244, 0.060428, 0.038189, 0.038606, 0.042050, 0.038997, 0.035323, 0.029665, 0.033907, 0.038288, 0.017471, 0.031069, 0.030238, 0.025735, 0.020350, 0.014198, 0.012919, 0.012094, 0.010160, 0.010514, 0.005527, 0.006399, 0.004232, 0.002667],
                  [0.041661, 0.042592, 0.072654, 0.204266, 0.568685, 1.000000, 0.419720, 0.220019, 0.495519, 0.199545, 0.070834, 0.164315, 0.147530, 0.154276, 0.116741, 0.121176, 0.091007, 0.067478, 0.042546, 0.073123, 0.068860, 0.053839, 0.029921, 0.029768, 0.029465, 0.061012, 0.045171, 0.090820, 0.070394, 0.054073, 0.063202, 0.033421, 0.028364, 0.036574, 0.045538, 0.051262, 0.038076, 0.033794, 0.020677, 0.070303, 0.063244, 0.060428, 0.038189, 0.038606, 0.042050, 0.038997, 0.035323, 0.029665, 0.033907, 0.038288, 0.017471, 0.031069, 0.030238, 0.025735, 0.020350, 0.014198, 0.012919, 0.012094, 0.010160, 0.010514, 0.005527, 0.006399, 0.004232, 0.002667],
                  [0.041661, 0.042592, 0.072654, 0.204266, 0.568685, 1.000000, 0.419720, 0.220019, 0.495519, 0.199545, 0.070834, 0.164315, 0.147530, 0.154276, 0.116741, 0.121176, 0.091007, 0.067478, 0.042546, 0.073123, 0.068860, 0.053839, 0.029921, 0.029768, 0.029465, 0.061012, 0.045171, 0.090820, 0.070394, 0.054073, 0.063202, 0.033421, 0.028364, 0.036574, 0.045538, 0.051262, 0.038076, 0.033794, 0.020677, 0.070303, 0.063244, 0.060428, 0.038189, 0.038606, 0.042050, 0.038997, 0.035323, 0.029665, 0.033907, 0.038288, 0.017471, 0.031069, 0.030238, 0.025735, 0.020350, 0.014198, 0.012919, 0.012094, 0.010160, 0.010514, 0.005527, 0.006399, 0.004232, 0.002667],
                  [0.041661, 0.042592, 0.072654, 0.204266, 0.568685, 1.000000, 0.419720, 0.220019, 0.495519, 0.199545, 0.070834, 0.164315, 0.147530, 0.154276, 0.116741, 0.121176, 0.091007, 0.067478, 0.042546, 0.073123, 0.068860, 0.053839, 0.029921, 0.029768, 0.029465, 0.061012, 0.045171, 0.090820, 0.070394, 0.054073, 0.063202, 0.033421, 0.028364, 0.036574, 0.045538, 0.051262, 0.038076, 0.033794, 0.020677, 0.070303, 0.063244, 0.060428, 0.038189, 0.038606, 0.042050, 0.038997, 0.035323, 0.029665, 0.033907, 0.038288, 0.017471, 0.031069, 0.030238, 0.025735, 0.020350, 0.014198, 0.012919, 0.012094, 0.010160, 0.010514, 0.005527, 0.006399, 0.004232, 0.002667],
                  [0.041661, 0.042592, 0.072654, 0.204266, 0.568685, 1.000000, 0.419720, 0.220019, 0.495519, 0.199545, 0.070834, 0.164315, 0.147530, 0.154276, 0.116741, 0.121176, 0.091007, 0.067478, 0.042546, 0.073123, 0.068860, 0.053839, 0.029921, 0.029768, 0.029465, 0.061012, 0.045171, 0.090820, 0.070394, 0.054073, 0.063202, 0.033421, 0.028364, 0.036574, 0.045538, 0.051262, 0.038076, 0.033794, 0.020677, 0.070303, 0.063244, 0.060428, 0.038189, 0.038606, 0.042050, 0.038997, 0.035323, 0.029665, 0.033907, 0.038288, 0.017471, 0.031069, 0.030238, 0.025735, 0.020350, 0.014198, 0.012919, 0.012094, 0.010160, 0.010514, 0.005527, 0.006399, 0.004232, 0.002667],
                  [0.041661, 0.042592, 0.072654, 0.204266, 0.568685, 1.000000, 0.419720, 0.220019, 0.495519, 0.199545, 0.070834, 0.164315, 0.147530, 0.154276, 0.116741, 0.121176, 0.091007, 0.067478, 0.042546, 0.073123, 0.068860, 0.053839, 0.029921, 0.029768, 0.029465, 0.061012, 0.045171, 0.090820, 0.070394, 0.054073, 0.063202, 0.033421, 0.028364, 0.036574, 0.045538, 0.051262, 0.038076, 0.033794, 0.020677, 0.070303, 0.063244, 0.060428, 0.038189, 0.038606, 0.042050, 0.038997, 0.035323, 0.029665, 0.033907, 0.038288, 0.017471, 0.031069, 0.030238, 0.025735, 0.020350, 0.014198, 0.012919, 0.012094, 0.010160, 0.010514, 0.005527, 0.006399, 0.004232, 0.002667],
                  [0.200378, 0.282013, 0.542122, 0.937123, 1.000000, 0.669833, 0.359089, 0.272836, 0.357189, 0.526830, 0.503286, 0.312226, 0.179440, 0.109140, 0.068114, 0.073872, 0.125688, 0.109209, 0.063483, 0.078976, 0.074902, 0.048985, 0.061145, 0.044165, 0.046876, 0.039973, 0.057835, 0.029931, 0.040996, 0.045161, 0.050566, 0.042235, 0.032082, 0.020011, 0.037255, 0.077936, 0.047181, 0.037362, 0.035730, 0.035150, 0.045436, 0.050355, 0.044337, 0.078095, 0.041606, 0.055516, 0.046250, 0.031578, 0.053856, 0.034707, 0.036807, 0.031010, 0.027231, 0.035225, 0.046270, 0.019425, 0.022486, 0.021249, 0.011158, 0.009750, 0.012357, 0.011702, 0.008706, 0.004675],
                  [0.200378, 0.282013, 0.542122, 0.937123, 1.000000, 0.669833, 0.359089, 0.272836, 0.357189, 0.526830, 0.503286, 0.312226, 0.179440, 0.109140, 0.068114, 0.073872, 0.125688, 0.109209, 0.063483, 0.078976, 0.074902, 0.048985, 0.061145, 0.044165, 0.046876, 0.039973, 0.057835, 0.029931, 0.040996, 0.045161, 0.050566, 0.042235, 0.032082, 0.020011, 0.037255, 0.077936, 0.047181, 0.037362, 0.035730, 0.035150, 0.045436, 0.050355, 0.044337, 0.078095, 0.041606, 0.055516, 0.046250, 0.031578, 0.053856, 0.034707, 0.036807, 0.031010, 0.027231, 0.035225, 0.046270, 0.019425, 0.022486, 0.021249, 0.011158, 0.009750, 0.012357, 0.011702, 0.008706, 0.004675],
                  [0.200378, 0.282013, 0.542122, 0.937123, 1.000000, 0.669833, 0.359089, 0.272836, 0.357189, 0.526830, 0.503286, 0.312226, 0.179440, 0.109140, 0.068114, 0.073872, 0.125688, 0.109209, 0.063483, 0.078976, 0.074902, 0.048985, 0.061145, 0.044165, 0.046876, 0.039973, 0.057835, 0.029931, 0.040996, 0.045161, 0.050566, 0.042235, 0.032082, 0.020011, 0.037255, 0.077936, 0.047181, 0.037362, 0.035730, 0.035150, 0.045436, 0.050355, 0.044337, 0.078095, 0.041606, 0.055516, 0.046250, 0.031578, 0.053856, 0.034707, 0.036807, 0.031010, 0.027231, 0.035225, 0.046270, 0.019425, 0.022486, 0.021249, 0.011158, 0.009750, 0.012357, 0.011702, 0.008706, 0.004675],
                  [0.200378, 0.282013, 0.542122, 0.937123, 1.000000, 0.669833, 0.359089, 0.272836, 0.357189, 0.526830, 0.503286, 0.312226, 0.179440, 0.109140, 0.068114, 0.073872, 0.125688, 0.109209, 0.063483, 0.078976, 0.074902, 0.048985, 0.061145, 0.044165, 0.046876, 0.039973, 0.057835, 0.029931, 0.040996, 0.045161, 0.050566, 0.042235, 0.032082, 0.020011, 0.037255, 0.077936, 0.047181, 0.037362, 0.035730, 0.035150, 0.045436, 0.050355, 0.044337, 0.078095, 0.041606, 0.055516, 0.046250, 0.031578, 0.053856, 0.034707, 0.036807, 0.031010, 0.027231, 0.035225, 0.046270, 0.019425, 0.022486, 0.021249, 0.011158, 0.009750, 0.012357, 0.011702, 0.008706, 0.004675],
                  [0.200378, 0.282013, 0.542122, 0.937123, 1.000000, 0.669833, 0.359089, 0.272836, 0.357189, 0.526830, 0.503286, 0.312226, 0.179440, 0.109140, 0.068114, 0.073872, 0.125688, 0.109209, 0.063483, 0.078976, 0.074902, 0.048985, 0.061145, 0.044165, 0.046876, 0.039973, 0.057835, 0.029931, 0.040996, 0.045161, 0.050566, 0.042235, 0.032082, 0.020011, 0.037255, 0.077936, 0.047181, 0.037362, 0.035730, 0.035150, 0.045436, 0.050355, 0.044337, 0.078095, 0.041606, 0.055516, 0.046250, 0.031578, 0.053856, 0.034707, 0.036807, 0.031010, 0.027231, 0.035225, 0.046270, 0.019425, 0.022486, 0.021249, 0.011158, 0.009750, 0.012357, 0.011702, 0.008706, 0.004675],
                  [0.200378, 0.282013, 0.542122, 0.937123, 1.000000, 0.669833, 0.359089, 0.272836, 0.357189, 0.526830, 0.503286, 0.312226, 0.179440, 0.109140, 0.068114, 0.073872, 0.125688, 0.109209, 0.063483, 0.078976, 0.074902, 0.048985, 0.061145, 0.044165, 0.046876, 0.039973, 0.057835, 0.029931, 0.040996, 0.045161, 0.050566, 0.042235, 0.032082, 0.020011, 0.037255, 0.077936, 0.047181, 0.037362, 0.035730, 0.035150, 0.045436, 0.050355, 0.044337, 0.078095, 0.041606, 0.055516, 0.046250, 0.031578, 0.053856, 0.034707, 0.036807, 0.031010, 0.027231, 0.035225, 0.046270, 0.019425, 0.022486, 0.021249, 0.011158, 0.009750, 0.012357, 0.011702, 0.008706, 0.004675],
                  [0.200378, 0.282013, 0.542122, 0.937123, 1.000000, 0.669833, 0.359089, 0.272836, 0.357189, 0.526830, 0.503286, 0.312226, 0.179440, 0.109140, 0.068114, 0.073872, 0.125688, 0.109209, 0.063483, 0.078976, 0.074902, 0.048985, 0.061145, 0.044165, 0.046876, 0.039973, 0.057835, 0.029931, 0.040996, 0.045161, 0.050566, 0.042235, 0.032082, 0.020011, 0.037255, 0.077936, 0.047181, 0.037362, 0.035730, 0.035150, 0.045436, 0.050355, 0.044337, 0.078095, 0.041606, 0.055516, 0.046250, 0.031578, 0.053856, 0.034707, 0.036807, 0.031010, 0.027231, 0.035225, 0.046270, 0.019425, 0.022486, 0.021249, 0.011158, 0.009750, 0.012357, 0.011702, 0.008706, 0.004675],
                  [0.200378, 0.282013, 0.542122, 0.937123, 1.000000, 0.669833, 0.359089, 0.272836, 0.357189, 0.526830, 0.503286, 0.312226, 0.179440, 0.109140, 0.068114, 0.073872, 0.125688, 0.109209, 0.063483, 0.078976, 0.074902, 0.048985, 0.061145, 0.044165, 0.046876, 0.039973, 0.057835, 0.029931, 0.040996, 0.045161, 0.050566, 0.042235, 0.032082, 0.020011, 0.037255, 0.077936, 0.047181, 0.037362, 0.035730, 0.035150, 0.045436, 0.050355, 0.044337, 0.078095, 0.041606, 0.055516, 0.046250, 0.031578, 0.053856, 0.034707, 0.036807, 0.031010, 0.027231, 0.035225, 0.046270, 0.019425, 0.022486, 0.021249, 0.011158, 0.009750, 0.012357, 0.011702, 0.008706, 0.004675],
                  [0.200378, 0.282013, 0.542122, 0.937123, 1.000000, 0.669833, 0.359089, 0.272836, 0.357189, 0.526830, 0.503286, 0.312226, 0.179440, 0.109140, 0.068114, 0.073872, 0.125688, 0.109209, 0.063483, 0.078976, 0.074902, 0.048985, 0.061145, 0.044165, 0.046876, 0.039973, 0.057835, 0.029931, 0.040996, 0.045161, 0.050566, 0.042235, 0.032082, 0.020011, 0.037255, 0.077936, 0.047181, 0.037362, 0.035730, 0.035150, 0.045436, 0.050355, 0.044337, 0.078095, 0.041606, 0.055516, 0.046250, 0.031578, 0.053856, 0.034707, 0.036807, 0.031010, 0.027231, 0.035225, 0.046270, 0.019425, 0.022486, 0.021249, 0.011158, 0.009750, 0.012357, 0.011702, 0.008706, 0.004675],
                  [0.200378, 0.282013, 0.542122, 0.937123, 1.000000, 0.669833, 0.359089, 0.272836, 0.357189, 0.526830, 0.503286, 0.312226, 0.179440, 0.109140, 0.068114, 0.073872, 0.125688, 0.109209, 0.063483, 0.078976, 0.074902, 0.048985, 0.061145, 0.044165, 0.046876, 0.039973, 0.057835, 0.029931, 0.040996, 0.045161, 0.050566, 0.042235, 0.032082, 0.020011, 0.037255, 0.077936, 0.047181, 0.037362, 0.035730, 0.035150, 0.045436, 0.050355, 0.044337, 0.078095, 0.041606, 0.055516, 0.046250, 0.031578, 0.053856, 0.034707, 0.036807, 0.031010, 0.027231, 0.035225, 0.046270, 0.019425, 0.022486, 0.021249, 0.011158, 0.009750, 0.012357, 0.011702, 0.008706, 0.004675],
                  [0.200378, 0.282013, 0.542122, 0.937123, 1.000000, 0.669833, 0.359089, 0.272836, 0.357189, 0.526830, 0.503286, 0.312226, 0.179440, 0.109140, 0.068114, 0.073872, 0.125688, 0.109209, 0.063483, 0.078976, 0.074902, 0.048985, 0.061145, 0.044165, 0.046876, 0.039973, 0.057835, 0.029931, 0.040996, 0.045161, 0.050566, 0.042235, 0.032082, 0.020011, 0.037255, 0.077936, 0.047181, 0.037362, 0.035730, 0.035150, 0.045436, 0.050355, 0.044337, 0.078095, 0.041606, 0.055516, 0.046250, 0.031578, 0.053856, 0.034707, 0.036807, 0.031010, 0.027231, 0.035225, 0.046270, 0.019425, 0.022486, 0.021249, 0.011158, 0.009750, 0.012357, 0.011702, 0.008706, 0.004675],
                  [0.200378, 0.282013, 0.542122, 0.937123, 1.000000, 0.669833, 0.359089, 0.272836, 0.357189, 0.526830, 0.503286, 0.312226, 0.179440, 0.109140, 0.068114, 0.073872, 0.125688, 0.109209, 0.063483, 0.078976, 0.074902, 0.048985, 0.061145, 0.044165, 0.046876, 0.039973, 0.057835, 0.029931, 0.040996, 0.045161, 0.050566, 0.042235, 0.032082, 0.020011, 0.037255, 0.077936, 0.047181, 0.037362, 0.035730, 0.035150, 0.045436, 0.050355, 0.044337, 0.078095, 0.041606, 0.055516, 0.046250, 0.031578, 0.053856, 0.034707, 0.036807, 0.031010, 0.027231, 0.035225, 0.046270, 0.019425, 0.022486, 0.021249, 0.011158, 0.009750, 0.012357, 0.011702, 0.008706, 0.004675],
                  [0.307938, 0.322513, 0.361429, 0.432496, 0.522951, 0.650771, 0.804849, 0.934890, 1.000000, 0.957381, 0.826601, 0.656589, 0.500956, 0.392643, 0.320361, 0.273750, 0.242903, 0.220764, 0.203371, 0.184314, 0.155127, 0.117715, 0.087720, 0.076660, 0.083853, 0.092005, 0.086559, 0.085785, 0.101626, 0.099901, 0.084348, 0.089350, 0.064314, 0.033261, 0.058540, 0.088122, 0.082847, 0.101780, 0.035665, 0.054918, 0.083491, 0.075051, 0.040073, 0.057531, 0.071317, 0.090247, 0.053445, 0.065962, 0.068310, 0.112469, 0.046693, 0.063332, 0.078693, 0.078395, 0.050339, 0.039677, 0.034165, 0.043517, 0.034151, 0.027810, 0.017624, 0.021335, 0.017565, 0.012180],
                  [0.307938, 0.322513, 0.361429, 0.432496, 0.522951, 0.650771, 0.804849, 0.934890, 1.000000, 0.957381, 0.826601, 0.656589, 0.500956, 0.392643, 0.320361, 0.273750, 0.242903, 0.220764, 0.203371, 0.184314, 0.155127, 0.117715, 0.087720, 0.076660, 0.083853, 0.092005, 0.086559, 0.085785, 0.101626, 0.099901, 0.084348, 0.089350, 0.064314, 0.033261, 0.058540, 0.088122, 0.082847, 0.101780, 0.035665, 0.054918, 0.083491, 0.075051, 0.040073, 0.057531, 0.071317, 0.090247, 0.053445, 0.065962, 0.068310, 0.112469, 0.046693, 0.063332, 0.078693, 0.078395, 0.050339, 0.039677, 0.034165, 0.043517, 0.034151, 0.027810, 0.017624, 0.021335, 0.017565, 0.012180],
                  [0.307938, 0.322513, 0.361429, 0.432496, 0.522951, 0.650771, 0.804849, 0.934890, 1.000000, 0.957381, 0.826601, 0.656589, 0.500956, 0.392643, 0.320361, 0.273750, 0.242903, 0.220764, 0.203371, 0.184314, 0.155127, 0.117715, 0.087720, 0.076660, 0.083853, 0.092005, 0.086559, 0.085785, 0.101626, 0.099901, 0.084348, 0.089350, 0.064314, 0.033261, 0.058540, 0.088122, 0.082847, 0.101780, 0.035665, 0.054918, 0.083491, 0.075051, 0.040073, 0.057531, 0.071317, 0.090247, 0.053445, 0.065962, 0.068310, 0.112469, 0.046693, 0.063332, 0.078693, 0.078395, 0.050339, 0.039677, 0.034165, 0.043517, 0.034151, 0.027810, 0.017624, 0.021335, 0.017565, 0.012180],
                  [0.307938, 0.322513, 0.361429, 0.432496, 0.522951, 0.650771, 0.804849, 0.934890, 1.000000, 0.957381, 0.826601, 0.656589, 0.500956, 0.392643, 0.320361, 0.273750, 0.242903, 0.220764, 0.203371, 0.184314, 0.155127, 0.117715, 0.087720, 0.076660, 0.083853, 0.092005, 0.086559, 0.085785, 0.101626, 0.099901, 0.084348, 0.089350, 0.064314, 0.033261, 0.058540, 0.088122, 0.082847, 0.101780, 0.035665, 0.054918, 0.083491, 0.075051, 0.040073, 0.057531, 0.071317, 0.090247, 0.053445, 0.065962, 0.068310, 0.112469, 0.046693, 0.063332, 0.078693, 0.078395, 0.050339, 0.039677, 0.034165, 0.043517, 0.034151, 0.027810, 0.017624, 0.021335, 0.017565, 0.012180],
                  [0.307938, 0.322513, 0.361429, 0.432496, 0.522951, 0.650771, 0.804849, 0.934890, 1.000000, 0.957381, 0.826601, 0.656589, 0.500956, 0.392643, 0.320361, 0.273750, 0.242903, 0.220764, 0.203371, 0.184314, 0.155127, 0.117715, 0.087720, 0.076660, 0.083853, 0.092005, 0.086559, 0.085785, 0.101626, 0.099901, 0.084348, 0.089350, 0.064314, 0.033261, 0.058540, 0.088122, 0.082847, 0.101780, 0.035665, 0.054918, 0.083491, 0.075051, 0.040073, 0.057531, 0.071317, 0.090247, 0.053445, 0.065962, 0.068310, 0.112469, 0.046693, 0.063332, 0.078693, 0.078395, 0.050339, 0.039677, 0.034165, 0.043517, 0.034151, 0.027810, 0.017624, 0.021335, 0.017565, 0.012180],
                  [0.307938, 0.322513, 0.361429, 0.432496, 0.522951, 0.650771, 0.804849, 0.934890, 1.000000, 0.957381, 0.826601, 0.656589, 0.500956, 0.392643, 0.320361, 0.273750, 0.242903, 0.220764, 0.203371, 0.184314, 0.155127, 0.117715, 0.087720, 0.076660, 0.083853, 0.092005, 0.086559, 0.085785, 0.101626, 0.099901, 0.084348, 0.089350, 0.064314, 0.033261, 0.058540, 0.088122, 0.082847, 0.101780, 0.035665, 0.054918, 0.083491, 0.075051, 0.040073, 0.057531, 0.071317, 0.090247, 0.053445, 0.065962, 0.068310, 0.112469, 0.046693, 0.063332, 0.078693, 0.078395, 0.050339, 0.039677, 0.034165, 0.043517, 0.034151, 0.027810, 0.017624, 0.021335, 0.017565, 0.012180],
                  [0.307938, 0.322513, 0.361429, 0.432496, 0.522951, 0.650771, 0.804849, 0.934890, 1.000000, 0.957381, 0.826601, 0.656589, 0.500956, 0.392643, 0.320361, 0.273750, 0.242903, 0.220764, 0.203371, 0.184314, 0.155127, 0.117715, 0.087720, 0.076660, 0.083853, 0.092005, 0.086559, 0.085785, 0.101626, 0.099901, 0.084348, 0.089350, 0.064314, 0.033261, 0.058540, 0.088122, 0.082847, 0.101780, 0.035665, 0.054918, 0.083491, 0.075051, 0.040073, 0.057531, 0.071317, 0.090247, 0.053445, 0.065962, 0.068310, 0.112469, 0.046693, 0.063332, 0.078693, 0.078395, 0.050339, 0.039677, 0.034165, 0.043517, 0.034151, 0.027810, 0.017624, 0.021335, 0.017565, 0.012180],
                  [0.307938, 0.322513, 0.361429, 0.432496, 0.522951, 0.650771, 0.804849, 0.934890, 1.000000, 0.957381, 0.826601, 0.656589, 0.500956, 0.392643, 0.320361, 0.273750, 0.242903, 0.220764, 0.203371, 0.184314, 0.155127, 0.117715, 0.087720, 0.076660, 0.083853, 0.092005, 0.086559, 0.085785, 0.101626, 0.099901, 0.084348, 0.089350, 0.064314, 0.033261, 0.058540, 0.088122, 0.082847, 0.101780, 0.035665, 0.054918, 0.083491, 0.075051, 0.040073, 0.057531, 0.071317, 0.090247, 0.053445, 0.065962, 0.068310, 0.112469, 0.046693, 0.063332, 0.078693, 0.078395, 0.050339, 0.039677, 0.034165, 0.043517, 0.034151, 0.027810, 0.017624, 0.021335, 0.017565, 0.012180],
                  [0.307938, 0.322513, 0.361429, 0.432496, 0.522951, 0.650771, 0.804849, 0.934890, 1.000000, 0.957381, 0.826601, 0.656589, 0.500956, 0.392643, 0.320361, 0.273750, 0.242903, 0.220764, 0.203371, 0.184314, 0.155127, 0.117715, 0.087720, 0.076660, 0.083853, 0.092005, 0.086559, 0.085785, 0.101626, 0.099901, 0.084348, 0.089350, 0.064314, 0.033261, 0.058540, 0.088122, 0.082847, 0.101780, 0.035665, 0.054918, 0.083491, 0.075051, 0.040073, 0.057531, 0.071317, 0.090247, 0.053445, 0.065962, 0.068310, 0.112469, 0.046693, 0.063332, 0.078693, 0.078395, 0.050339, 0.039677, 0.034165, 0.043517, 0.034151, 0.027810, 0.017624, 0.021335, 0.017565, 0.012180],
                  [0.307938, 0.322513, 0.361429, 0.432496, 0.522951, 0.650771, 0.804849, 0.934890, 1.000000, 0.957381, 0.826601, 0.656589, 0.500956, 0.392643, 0.320361, 0.273750, 0.242903, 0.220764, 0.203371, 0.184314, 0.155127, 0.117715, 0.087720, 0.076660, 0.083853, 0.092005, 0.086559, 0.085785, 0.101626, 0.099901, 0.084348, 0.089350, 0.064314, 0.033261, 0.058540, 0.088122, 0.082847, 0.101780, 0.035665, 0.054918, 0.083491, 0.075051, 0.040073, 0.057531, 0.071317, 0.090247, 0.053445, 0.065962, 0.068310, 0.112469, 0.046693, 0.063332, 0.078693, 0.078395, 0.050339, 0.039677, 0.034165, 0.043517, 0.034151, 0.027810, 0.017624, 0.021335, 0.017565, 0.012180],
                  [0.307938, 0.322513, 0.361429, 0.432496, 0.522951, 0.650771, 0.804849, 0.934890, 1.000000, 0.957381, 0.826601, 0.656589, 0.500956, 0.392643, 0.320361, 0.273750, 0.242903, 0.220764, 0.203371, 0.184314, 0.155127, 0.117715, 0.087720, 0.076660, 0.083853, 0.092005, 0.086559, 0.085785, 0.101626, 0.099901, 0.084348, 0.089350, 0.064314, 0.033261, 0.058540, 0.088122, 0.082847, 0.101780, 0.035665, 0.054918, 0.083491, 0.075051, 0.040073, 0.057531, 0.071317, 0.090247, 0.053445, 0.065962, 0.068310, 0.112469, 0.046693, 0.063332, 0.078693, 0.078395, 0.050339, 0.039677, 0.034165, 0.043517, 0.034151, 0.027810, 0.017624, 0.021335, 0.017565, 0.012180],
                  [0.307938, 0.322513, 0.361429, 0.432496, 0.522951, 0.650771, 0.804849, 0.934890, 1.000000, 0.957381, 0.826601, 0.656589, 0.500956, 0.392643, 0.320361, 0.273750, 0.242903, 0.220764, 0.203371, 0.184314, 0.155127, 0.117715, 0.087720, 0.076660, 0.083853, 0.092005, 0.086559, 0.085785, 0.101626, 0.099901, 0.084348, 0.089350, 0.064314, 0.033261, 0.058540, 0.088122, 0.082847, 0.101780, 0.035665, 0.054918, 0.083491, 0.075051, 0.040073, 0.057531, 0.071317, 0.090247, 0.053445, 0.065962, 0.068310, 0.112469, 0.046693, 0.063332, 0.078693, 0.078395, 0.050339, 0.039677, 0.034165, 0.043517, 0.034151, 0.027810, 0.017624, 0.021335, 0.017565, 0.012180],
                  [0.950827, 0.952489, 0.956634, 0.963440, 0.970987, 0.980206, 0.989908, 0.997221, 1.000000, 0.994141, 0.976463, 0.942285, 0.888936, 0.821214, 0.738332, 0.646020, 0.551690, 0.462552, 0.384377, 0.319868, 0.270047, 0.234061, 0.211169, 0.200277, 0.201681, 0.213177, 0.228126, 0.231000, 0.206113, 0.161842, 0.124313, 0.114420, 0.142130, 0.194204, 0.194364, 0.118646, 0.068841, 0.076481, 0.131631, 0.156365, 0.125620, 0.115408, 0.103836, 0.101251, 0.132330, 0.094633, 0.092095, 0.123912, 0.095676, 0.178548, 0.114326, 0.066725, 0.075763, 0.121136, 0.096280, 0.050853, 0.079216, 0.034775, 0.039218, 0.035903, 0.032956, 0.029297, 0.018401, 0.014322],
                  [0.950827, 0.952489, 0.956634, 0.963440, 0.970987, 0.980206, 0.989908, 0.997221, 1.000000, 0.994141, 0.976463, 0.942285, 0.888936, 0.821214, 0.738332, 0.646020, 0.551690, 0.462552, 0.384377, 0.319868, 0.270047, 0.234061, 0.211169, 0.200277, 0.201681, 0.213177, 0.228126, 0.231000, 0.206113, 0.161842, 0.124313, 0.114420, 0.142130, 0.194204, 0.194364, 0.118646, 0.068841, 0.076481, 0.131631, 0.156365, 0.125620, 0.115408, 0.103836, 0.101251, 0.132330, 0.094633, 0.092095, 0.123912, 0.095676, 0.178548, 0.114326, 0.066725, 0.075763, 0.121136, 0.096280, 0.050853, 0.079216, 0.034775, 0.039218, 0.035903, 0.032956, 0.029297, 0.018401, 0.014322],
                  [0.950827, 0.952489, 0.956634, 0.963440, 0.970987, 0.980206, 0.989908, 0.997221, 1.000000, 0.994141, 0.976463, 0.942285, 0.888936, 0.821214, 0.738332, 0.646020, 0.551690, 0.462552, 0.384377, 0.319868, 0.270047, 0.234061, 0.211169, 0.200277, 0.201681, 0.213177, 0.228126, 0.231000, 0.206113, 0.161842, 0.124313, 0.114420, 0.142130, 0.194204, 0.194364, 0.118646, 0.068841, 0.076481, 0.131631, 0.156365, 0.125620, 0.115408, 0.103836, 0.101251, 0.132330, 0.094633, 0.092095, 0.123912, 0.095676, 0.178548, 0.114326, 0.066725, 0.075763, 0.121136, 0.096280, 0.050853, 0.079216, 0.034775, 0.039218, 0.035903, 0.032956, 0.029297, 0.018401, 0.014322],
                  [0.950827, 0.952489, 0.956634, 0.963440, 0.970987, 0.980206, 0.989908, 0.997221, 1.000000, 0.994141, 0.976463, 0.942285, 0.888936, 0.821214, 0.738332, 0.646020, 0.551690, 0.462552, 0.384377, 0.319868, 0.270047, 0.234061, 0.211169, 0.200277, 0.201681, 0.213177, 0.228126, 0.231000, 0.206113, 0.161842, 0.124313, 0.114420, 0.142130, 0.194204, 0.194364, 0.118646, 0.068841, 0.076481, 0.131631, 0.156365, 0.125620, 0.115408, 0.103836, 0.101251, 0.132330, 0.094633, 0.092095, 0.123912, 0.095676, 0.178548, 0.114326, 0.066725, 0.075763, 0.121136, 0.096280, 0.050853, 0.079216, 0.034775, 0.039218, 0.035903, 0.032956, 0.029297, 0.018401, 0.014322],
                  [0.950827, 0.952489, 0.956634, 0.963440, 0.970987, 0.980206, 0.989908, 0.997221, 1.000000, 0.994141, 0.976463, 0.942285, 0.888936, 0.821214, 0.738332, 0.646020, 0.551690, 0.462552, 0.384377, 0.319868, 0.270047, 0.234061, 0.211169, 0.200277, 0.201681, 0.213177, 0.228126, 0.231000, 0.206113, 0.161842, 0.124313, 0.114420, 0.142130, 0.194204, 0.194364, 0.118646, 0.068841, 0.076481, 0.131631, 0.156365, 0.125620, 0.115408, 0.103836, 0.101251, 0.132330, 0.094633, 0.092095, 0.123912, 0.095676, 0.178548, 0.114326, 0.066725, 0.075763, 0.121136, 0.096280, 0.050853, 0.079216, 0.034775, 0.039218, 0.035903, 0.032956, 0.029297, 0.018401, 0.014322],
                  [0.950827, 0.952489, 0.956634, 0.963440, 0.970987, 0.980206, 0.989908, 0.997221, 1.000000, 0.994141, 0.976463, 0.942285, 0.888936, 0.821214, 0.738332, 0.646020, 0.551690, 0.462552, 0.384377, 0.319868, 0.270047, 0.234061, 0.211169, 0.200277, 0.201681, 0.213177, 0.228126, 0.231000, 0.206113, 0.161842, 0.124313, 0.114420, 0.142130, 0.194204, 0.194364, 0.118646, 0.068841, 0.076481, 0.131631, 0.156365, 0.125620, 0.115408, 0.103836, 0.101251, 0.132330, 0.094633, 0.092095, 0.123912, 0.095676, 0.178548, 0.114326, 0.066725, 0.075763, 0.121136, 0.096280, 0.050853, 0.079216, 0.034775, 0.039218, 0.035903, 0.032956, 0.029297, 0.018401, 0.014322],
                  [0.950827, 0.952489, 0.956634, 0.963440, 0.970987, 0.980206, 0.989908, 0.997221, 1.000000, 0.994141, 0.976463, 0.942285, 0.888936, 0.821214, 0.738332, 0.646020, 0.551690, 0.462552, 0.384377, 0.319868, 0.270047, 0.234061, 0.211169, 0.200277, 0.201681, 0.213177, 0.228126, 0.231000, 0.206113, 0.161842, 0.124313, 0.114420, 0.142130, 0.194204, 0.194364, 0.118646, 0.068841, 0.076481, 0.131631, 0.156365, 0.125620, 0.115408, 0.103836, 0.101251, 0.132330, 0.094633, 0.092095, 0.123912, 0.095676, 0.178548, 0.114326, 0.066725, 0.075763, 0.121136, 0.096280, 0.050853, 0.079216, 0.034775, 0.039218, 0.035903, 0.032956, 0.029297, 0.018401, 0.014322],
                  [0.950827, 0.952489, 0.956634, 0.963440, 0.970987, 0.980206, 0.989908, 0.997221, 1.000000, 0.994141, 0.976463, 0.942285, 0.888936, 0.821214, 0.738332, 0.646020, 0.551690, 0.462552, 0.384377, 0.319868, 0.270047, 0.234061, 0.211169, 0.200277, 0.201681, 0.213177, 0.228126, 0.231000, 0.206113, 0.161842, 0.124313, 0.114420, 0.142130, 0.194204, 0.194364, 0.118646, 0.068841, 0.076481, 0.131631, 0.156365, 0.125620, 0.115408, 0.103836, 0.101251, 0.132330, 0.094633, 0.092095, 0.123912, 0.095676, 0.178548, 0.114326, 0.066725, 0.075763, 0.121136, 0.096280, 0.050853, 0.079216, 0.034775, 0.039218, 0.035903, 0.032956, 0.029297, 0.018401, 0.014322],
                  [0.950827, 0.952489, 0.956634, 0.963440, 0.970987, 0.980206, 0.989908, 0.997221, 1.000000, 0.994141, 0.976463, 0.942285, 0.888936, 0.821214, 0.738332, 0.646020, 0.551690, 0.462552, 0.384377, 0.319868, 0.270047, 0.234061, 0.211169, 0.200277, 0.201681, 0.213177, 0.228126, 0.231000, 0.206113, 0.161842, 0.124313, 0.114420, 0.142130, 0.194204, 0.194364, 0.118646, 0.068841, 0.076481, 0.131631, 0.156365, 0.125620, 0.115408, 0.103836, 0.101251, 0.132330, 0.094633, 0.092095, 0.123912, 0.095676, 0.178548, 0.114326, 0.066725, 0.075763, 0.121136, 0.096280, 0.050853, 0.079216, 0.034775, 0.039218, 0.035903, 0.032956, 0.029297, 0.018401, 0.014322],
                  [0.950827, 0.952489, 0.956634, 0.963440, 0.970987, 0.980206, 0.989908, 0.997221, 1.000000, 0.994141, 0.976463, 0.942285, 0.888936, 0.821214, 0.738332, 0.646020, 0.551690, 0.462552, 0.384377, 0.319868, 0.270047, 0.234061, 0.211169, 0.200277, 0.201681, 0.213177, 0.228126, 0.231000, 0.206113, 0.161842, 0.124313, 0.114420, 0.142130, 0.194204, 0.194364, 0.118646, 0.068841, 0.076481, 0.131631, 0.156365, 0.125620, 0.115408, 0.103836, 0.101251, 0.132330, 0.094633, 0.092095, 0.123912, 0.095676, 0.178548, 0.114326, 0.066725, 0.075763, 0.121136, 0.096280, 0.050853, 0.079216, 0.034775, 0.039218, 0.035903, 0.032956, 0.029297, 0.018401, 0.014322],
                  [0.950827, 0.952489, 0.956634, 0.963440, 0.970987, 0.980206, 0.989908, 0.997221, 1.000000, 0.994141, 0.976463, 0.942285, 0.888936, 0.821214, 0.738332, 0.646020, 0.551690, 0.462552, 0.384377, 0.319868, 0.270047, 0.234061, 0.211169, 0.200277, 0.201681, 0.213177, 0.228126, 0.231000, 0.206113, 0.161842, 0.124313, 0.114420, 0.142130, 0.194204, 0.194364, 0.118646, 0.068841, 0.076481, 0.131631, 0.156365, 0.125620, 0.115408, 0.103836, 0.101251, 0.132330, 0.094633, 0.092095, 0.123912, 0.095676, 0.178548, 0.114326, 0.066725, 0.075763, 0.121136, 0.096280, 0.050853, 0.079216, 0.034775, 0.039218, 0.035903, 0.032956, 0.029297, 0.018401, 0.014322],
                  [0.950827, 0.952489, 0.956634, 0.963440, 0.970987, 0.980206, 0.989908, 0.997221, 1.000000, 0.994141, 0.976463, 0.942285, 0.888936, 0.821214, 0.738332, 0.646020, 0.551690, 0.462552, 0.384377, 0.319868, 0.270047, 0.234061, 0.211169, 0.200277, 0.201681, 0.213177, 0.228126, 0.231000, 0.206113, 0.161842, 0.124313, 0.114420, 0.142130, 0.194204, 0.194364, 0.118646, 0.068841, 0.076481, 0.131631, 0.156365, 0.125620, 0.115408, 0.103836, 0.101251, 0.132330, 0.094633, 0.092095, 0.123912, 0.095676, 0.178548, 0.114326, 0.066725, 0.075763, 0.121136, 0.096280, 0.050853, 0.079216, 0.034775, 0.039218, 0.035903, 0.032956, 0.029297, 0.018401, 0.014322],
                  [1.000000, 0.998415, 0.994330, 0.987207, 0.978489, 0.966079, 0.949304, 0.929500, 0.904427, 0.873633, 0.839652, 0.799816, 0.754270, 0.706823, 0.654909, 0.599501, 0.541904, 0.483637, 0.426460, 0.371974, 0.321778, 0.276927, 0.238290, 0.204944, 0.178548, 0.158744, 0.144471, 0.135598, 0.131746, 0.132118, 0.135405, 0.139706, 0.142054, 0.139808, 0.131554, 0.118343, 0.103931, 0.091795, 0.084241, 0.081933, 0.083855, 0.087787, 0.090879, 0.090968, 0.087148, 0.080299, 0.074033, 0.074345, 0.085243, 0.096745, 0.085614, 0.065129, 0.070458, 0.101532, 0.089487, 0.053366, 0.050339, 0.042231, 0.031691, 0.039279, 0.026118, 0.030892, 0.025939, 0.018271],
                  [1.000000, 0.998415, 0.994330, 0.987207, 0.978489, 0.966079, 0.949304, 0.929500, 0.904427, 0.873633, 0.839652, 0.799816, 0.754270, 0.706823, 0.654909, 0.599501, 0.541904, 0.483637, 0.426460, 0.371974, 0.321778, 0.276927, 0.238290, 0.204944, 0.178548, 0.158744, 0.144471, 0.135598, 0.131746, 0.132118, 0.135405, 0.139706, 0.142054, 0.139808, 0.131554, 0.118343, 0.103931, 0.091795, 0.084241, 0.081933, 0.083855, 0.087787, 0.090879, 0.090968, 0.087148, 0.080299, 0.074033, 0.074345, 0.085243, 0.096745, 0.085614, 0.065129, 0.070458, 0.101532, 0.089487, 0.053366, 0.050339, 0.042231, 0.031691, 0.039279, 0.026118, 0.030892, 0.025939, 0.018271],
                  [1.000000, 0.998415, 0.994330, 0.987207, 0.978489, 0.966079, 0.949304, 0.929500, 0.904427, 0.873633, 0.839652, 0.799816, 0.754270, 0.706823, 0.654909, 0.599501, 0.541904, 0.483637, 0.426460, 0.371974, 0.321778, 0.276927, 0.238290, 0.204944, 0.178548, 0.158744, 0.144471, 0.135598, 0.131746, 0.132118, 0.135405, 0.139706, 0.142054, 0.139808, 0.131554, 0.118343, 0.103931, 0.091795, 0.084241, 0.081933, 0.083855, 0.087787, 0.090879, 0.090968, 0.087148, 0.080299, 0.074033, 0.074345, 0.085243, 0.096745, 0.085614, 0.065129, 0.070458, 0.101532, 0.089487, 0.053366, 0.050339, 0.042231, 0.031691, 0.039279, 0.026118, 0.030892, 0.025939, 0.018271],
                  [1.000000, 0.998415, 0.994330, 0.987207, 0.978489, 0.966079, 0.949304, 0.929500, 0.904427, 0.873633, 0.839652, 0.799816, 0.754270, 0.706823, 0.654909, 0.599501, 0.541904, 0.483637, 0.426460, 0.371974, 0.321778, 0.276927, 0.238290, 0.204944, 0.178548, 0.158744, 0.144471, 0.135598, 0.131746, 0.132118, 0.135405, 0.139706, 0.142054, 0.139808, 0.131554, 0.118343, 0.103931, 0.091795, 0.084241, 0.081933, 0.083855, 0.087787, 0.090879, 0.090968, 0.087148, 0.080299, 0.074033, 0.074345, 0.085243, 0.096745, 0.085614, 0.065129, 0.070458, 0.101532, 0.089487, 0.053366, 0.050339, 0.042231, 0.031691, 0.039279, 0.026118, 0.030892, 0.025939, 0.018271],
                  [1.000000, 0.998415, 0.994330, 0.987207, 0.978489, 0.966079, 0.949304, 0.929500, 0.904427, 0.873633, 0.839652, 0.799816, 0.754270, 0.706823, 0.654909, 0.599501, 0.541904, 0.483637, 0.426460, 0.371974, 0.321778, 0.276927, 0.238290, 0.204944, 0.178548, 0.158744, 0.144471, 0.135598, 0.131746, 0.132118, 0.135405, 0.139706, 0.142054, 0.139808, 0.131554, 0.118343, 0.103931, 0.091795, 0.084241, 0.081933, 0.083855, 0.087787, 0.090879, 0.090968, 0.087148, 0.080299, 0.074033, 0.074345, 0.085243, 0.096745, 0.085614, 0.065129, 0.070458, 0.101532, 0.089487, 0.053366, 0.050339, 0.042231, 0.031691, 0.039279, 0.026118, 0.030892, 0.025939, 0.018271],
                  [1.000000, 0.998415, 0.994330, 0.987207, 0.978489, 0.966079, 0.949304, 0.929500, 0.904427, 0.873633, 0.839652, 0.799816, 0.754270, 0.706823, 0.654909, 0.599501, 0.541904, 0.483637, 0.426460, 0.371974, 0.321778, 0.276927, 0.238290, 0.204944, 0.178548, 0.158744, 0.144471, 0.135598, 0.131746, 0.132118, 0.135405, 0.139706, 0.142054, 0.139808, 0.131554, 0.118343, 0.103931, 0.091795, 0.084241, 0.081933, 0.083855, 0.087787, 0.090879, 0.090968, 0.087148, 0.080299, 0.074033, 0.074345, 0.085243, 0.096745, 0.085614, 0.065129, 0.070458, 0.101532, 0.089487, 0.053366, 0.050339, 0.042231, 0.031691, 0.039279, 0.026118, 0.030892, 0.025939, 0.018271]])

    N = 4096
    number_of_band = 64

    H = np.zeros(N)
    for band in range(number_of_band):
        fL = f[band]
        fH = f[band + 1]
        for k in range(fL, fH):
            H[k] = a[note_number - 41][band]

    for k in range(1, int(N / 2)):
        H[N - k] = H[k]

    h = np.real(np.fft.ifft(H, N))

    for m in range(int(N / 2)):
        tmp = h[m]
        h[m] = h[int(N / 2) + m]
        h[int(N / 2) + m] = tmp

    J = 128

    b = np.zeros(J + 1)
    w = Hanning_window(J + 1)
    offset = int(N / 2) - int(J / 2)
    for m in range(J + 1):
        b[m] = h[offset + m] * w[m]

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    # decay
    p1 = 75
    p2 = 0
    p3 = 120 / 12
    p4 = 20
    decay = (1 - 1 / (1 + np.exp(-(note_number - p1) / p3))) * p4 + p2

    # d
    p1 = 30
    p2 = 0
    p3 = 120 / 12
    p4 = 0.5
    d = (1 - 1 / (1 + np.exp(-(note_number - p1) / p3))) * p4 + p2

    num = np.power(10, -3 * T / decay)
    den = np.sqrt((1 - d) * (1 - d) + 2 * d * (1 - d) * np.cos((2 * np.pi * f0) / fs) + d * d)

    c = num / den
    if c > 1:
        c = 1

    D = int(T * fs - d)
    e = T * fs - d - int(T * fs - d)
    g = (1 - e) / (1 + e)

    s0 = np.zeros(length_of_s)
    s1 = np.zeros(length_of_s)
    s2 = np.zeros(length_of_s)

    np.random.seed(0)
    number_of_partial = int(20000 / f0)
    for i in range(number_of_partial):
        theta = (np.random.rand() * 2 - 1) * np.pi
        for n in range(D + 1 + J):
            s0[n] += np.sin(2 * np.pi * f0 * (i + 1) * n / fs + theta)

    mean_of_s0 = 0
    for n in range(J, D + 1 + J):
        mean_of_s0 += s0[n]

    mean_of_s0 /= D + 1
    for n in range(D + 1 + J):
        s0[n] -= mean_of_s0

    for n in range(D + 1 + J, length_of_s):
        # fractional delay
        s1[n] = -g * s1[n - 1] + g * s0[n - D] + s0[n - D - 1]

        # filter
        s2[n] = c * ((1 - d) * s1[n] + d * s1[n - 1])

        # feedback
        s0[n] += s2[n]

    # comb filter
    excitation = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.11, 0.11, 0.11, 0.11, 0.11, 0.12, 0.12, 0.12, 0.13, 0.13, 0.13, 0.14, 0.14, 0.15, 0.15, 0.16, 0.16, 0.17, 0.17, 0.18, 0.19, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.3, 0.31, 0.32, 0.34, 0.35, 0.37, 0.39, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5])
    p = excitation[note_number - 41]
    s3 = np.zeros(length_of_s)
    for n in range(length_of_s):
        t = n - T * fs * p
        m = int(t)
        delta = t - m
        if m < 0:
            s3[n] = s0[n] - (delta * s0[m + 1 + D + 1] + (1 - delta) * s0[m + D + 1])
        else:
            s3[n] = s0[n] - (delta * s0[m + 1] + (1 - delta) * s0[m])

    s4 = np.zeros(length_of_s)
    for n in range(length_of_s):
        for m in range(J + 1):
            if n - m >= 0:
                s4[n] += b[m] * s3[n - m]

    s5 = np.zeros(length_of_s)
    for n in range(length_of_s - J):
        s5[n] = s4[J + n]

    # DC cancel
    fc = 5
    Q = 1 / np.sqrt(2)
    a, b = HPF(fs, fc, Q)
    s6 = filter(a, b, s5)
    s6 /= np.max(np.abs(s6))

    # A part
    VCF_A = np.array([0])
    VCF_D = np.array([T * 5])
    VCF_S = np.array([0])
    VCF_R = np.array([T * 5])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([f0 * 32])
    VCF_depth = np.array([f0 * 512])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]
        if vcf[n] > 20000:
            vcf[n] = 20000

    sa = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                sa[n] += b[m] * s6[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                sa[n] += -a[m] * sa[n - m]

    VCA_A = np.array([T * 1])
    VCA_D = np.array([T * 20])
    VCA_S = np.array([0])
    VCA_R = np.array([T * 20])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        sa[n] *= vca[n]

    # B part
    VCF_A = np.array([0])
    VCF_D = np.array([decay * 0.2])
    VCF_S = np.array([0])
    VCF_R = np.array([decay * 0.2])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([f0 * 8])
    VCF_depth = np.array([f0 * 128])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]
        if vcf[n] > 20000:
            vcf[n] = 20000

    sb = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                sb[n] += b[m] * s6[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                sb[n] += -a[m] * sb[n - m]

    VCA_A = np.array([T * 1])
    VCA_D = np.array([0])
    VCA_S = np.array([1])
    VCA_R = np.array([0])
    VCA_gate = np.array([duration])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        sb[n] *= vca[n]

    s7 = sa * 0.5 + sb * 0.5

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
        s7[n] *= vca[n]

    s7 *= velocity / 127 / np.max(np.abs(s7))

    return s7

def acoustic_piano(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    s0 = np.zeros(length_of_s)

    f = np.array([0, 4, 7, 12, 16, 20, 25, 30, 35, 41, 47, 53, 60, 67, 74, 82, 90, 99, 108, 118, 128, 139, 150, 162, 175, 188, 202, 217, 233, 250, 267, 286, 306, 326, 348, 371, 396, 421, 449, 477, 508, 540, 574, 609, 647, 687, 729, 773, 820, 869, 922, 977, 1035, 1097, 1161, 1230, 1302, 1379, 1460, 1545, 1635, 1730, 1830, 1936, 2048])
    a = np.array([[0.097669, 0.203922, 1.000000, 0.964866, 0.459230, 0.102213, 0.135372, 0.253113, 0.314785, 0.184495, 0.075184, 0.111320, 0.197297, 0.424351, 0.036438, 0.046227, 0.281716, 0.172889, 0.040024, 0.073363, 0.045535, 0.051568, 0.129786, 0.031104, 0.022747, 0.025675, 0.054747, 0.013732, 0.003639, 0.031264, 0.013598, 0.007524, 0.013859, 0.005472, 0.009672, 0.009180, 0.004518, 0.004326, 0.011827, 0.003077, 0.001889, 0.000623, 0.001578, 0.000632, 0.000441, 0.000516, 0.000229, 0.000249, 0.000095, 0.000085, 0.000045, 0.000025, 0.000027, 0.000017, 0.000013, 0.000012, 0.000012, 0.000013, 0.000010, 0.000009, 0.000011, 0.000009, 0.000009, 0.000008],
                  [0.097669, 0.203922, 1.000000, 0.964866, 0.459230, 0.102213, 0.135372, 0.253113, 0.314785, 0.184495, 0.075184, 0.111320, 0.197297, 0.424351, 0.036438, 0.046227, 0.281716, 0.172889, 0.040024, 0.073363, 0.045535, 0.051568, 0.129786, 0.031104, 0.022747, 0.025675, 0.054747, 0.013732, 0.003639, 0.031264, 0.013598, 0.007524, 0.013859, 0.005472, 0.009672, 0.009180, 0.004518, 0.004326, 0.011827, 0.003077, 0.001889, 0.000623, 0.001578, 0.000632, 0.000441, 0.000516, 0.000229, 0.000249, 0.000095, 0.000085, 0.000045, 0.000025, 0.000027, 0.000017, 0.000013, 0.000012, 0.000012, 0.000013, 0.000010, 0.000009, 0.000011, 0.000009, 0.000009, 0.000008],
                  [0.097669, 0.203922, 1.000000, 0.964866, 0.459230, 0.102213, 0.135372, 0.253113, 0.314785, 0.184495, 0.075184, 0.111320, 0.197297, 0.424351, 0.036438, 0.046227, 0.281716, 0.172889, 0.040024, 0.073363, 0.045535, 0.051568, 0.129786, 0.031104, 0.022747, 0.025675, 0.054747, 0.013732, 0.003639, 0.031264, 0.013598, 0.007524, 0.013859, 0.005472, 0.009672, 0.009180, 0.004518, 0.004326, 0.011827, 0.003077, 0.001889, 0.000623, 0.001578, 0.000632, 0.000441, 0.000516, 0.000229, 0.000249, 0.000095, 0.000085, 0.000045, 0.000025, 0.000027, 0.000017, 0.000013, 0.000012, 0.000012, 0.000013, 0.000010, 0.000009, 0.000011, 0.000009, 0.000009, 0.000008],
                  [0.097669, 0.203922, 1.000000, 0.964866, 0.459230, 0.102213, 0.135372, 0.253113, 0.314785, 0.184495, 0.075184, 0.111320, 0.197297, 0.424351, 0.036438, 0.046227, 0.281716, 0.172889, 0.040024, 0.073363, 0.045535, 0.051568, 0.129786, 0.031104, 0.022747, 0.025675, 0.054747, 0.013732, 0.003639, 0.031264, 0.013598, 0.007524, 0.013859, 0.005472, 0.009672, 0.009180, 0.004518, 0.004326, 0.011827, 0.003077, 0.001889, 0.000623, 0.001578, 0.000632, 0.000441, 0.000516, 0.000229, 0.000249, 0.000095, 0.000085, 0.000045, 0.000025, 0.000027, 0.000017, 0.000013, 0.000012, 0.000012, 0.000013, 0.000010, 0.000009, 0.000011, 0.000009, 0.000009, 0.000008],
                  [0.097669, 0.203922, 1.000000, 0.964866, 0.459230, 0.102213, 0.135372, 0.253113, 0.314785, 0.184495, 0.075184, 0.111320, 0.197297, 0.424351, 0.036438, 0.046227, 0.281716, 0.172889, 0.040024, 0.073363, 0.045535, 0.051568, 0.129786, 0.031104, 0.022747, 0.025675, 0.054747, 0.013732, 0.003639, 0.031264, 0.013598, 0.007524, 0.013859, 0.005472, 0.009672, 0.009180, 0.004518, 0.004326, 0.011827, 0.003077, 0.001889, 0.000623, 0.001578, 0.000632, 0.000441, 0.000516, 0.000229, 0.000249, 0.000095, 0.000085, 0.000045, 0.000025, 0.000027, 0.000017, 0.000013, 0.000012, 0.000012, 0.000013, 0.000010, 0.000009, 0.000011, 0.000009, 0.000009, 0.000008],
                  [0.097669, 0.203922, 1.000000, 0.964866, 0.459230, 0.102213, 0.135372, 0.253113, 0.314785, 0.184495, 0.075184, 0.111320, 0.197297, 0.424351, 0.036438, 0.046227, 0.281716, 0.172889, 0.040024, 0.073363, 0.045535, 0.051568, 0.129786, 0.031104, 0.022747, 0.025675, 0.054747, 0.013732, 0.003639, 0.031264, 0.013598, 0.007524, 0.013859, 0.005472, 0.009672, 0.009180, 0.004518, 0.004326, 0.011827, 0.003077, 0.001889, 0.000623, 0.001578, 0.000632, 0.000441, 0.000516, 0.000229, 0.000249, 0.000095, 0.000085, 0.000045, 0.000025, 0.000027, 0.000017, 0.000013, 0.000012, 0.000012, 0.000013, 0.000010, 0.000009, 0.000011, 0.000009, 0.000009, 0.000008],
                  [0.097669, 0.203922, 1.000000, 0.964866, 0.459230, 0.102213, 0.135372, 0.253113, 0.314785, 0.184495, 0.075184, 0.111320, 0.197297, 0.424351, 0.036438, 0.046227, 0.281716, 0.172889, 0.040024, 0.073363, 0.045535, 0.051568, 0.129786, 0.031104, 0.022747, 0.025675, 0.054747, 0.013732, 0.003639, 0.031264, 0.013598, 0.007524, 0.013859, 0.005472, 0.009672, 0.009180, 0.004518, 0.004326, 0.011827, 0.003077, 0.001889, 0.000623, 0.001578, 0.000632, 0.000441, 0.000516, 0.000229, 0.000249, 0.000095, 0.000085, 0.000045, 0.000025, 0.000027, 0.000017, 0.000013, 0.000012, 0.000012, 0.000013, 0.000010, 0.000009, 0.000011, 0.000009, 0.000009, 0.000008],
                  [0.097669, 0.203922, 1.000000, 0.964866, 0.459230, 0.102213, 0.135372, 0.253113, 0.314785, 0.184495, 0.075184, 0.111320, 0.197297, 0.424351, 0.036438, 0.046227, 0.281716, 0.172889, 0.040024, 0.073363, 0.045535, 0.051568, 0.129786, 0.031104, 0.022747, 0.025675, 0.054747, 0.013732, 0.003639, 0.031264, 0.013598, 0.007524, 0.013859, 0.005472, 0.009672, 0.009180, 0.004518, 0.004326, 0.011827, 0.003077, 0.001889, 0.000623, 0.001578, 0.000632, 0.000441, 0.000516, 0.000229, 0.000249, 0.000095, 0.000085, 0.000045, 0.000025, 0.000027, 0.000017, 0.000013, 0.000012, 0.000012, 0.000013, 0.000010, 0.000009, 0.000011, 0.000009, 0.000009, 0.000008],
                  [0.097669, 0.203922, 1.000000, 0.964866, 0.459230, 0.102213, 0.135372, 0.253113, 0.314785, 0.184495, 0.075184, 0.111320, 0.197297, 0.424351, 0.036438, 0.046227, 0.281716, 0.172889, 0.040024, 0.073363, 0.045535, 0.051568, 0.129786, 0.031104, 0.022747, 0.025675, 0.054747, 0.013732, 0.003639, 0.031264, 0.013598, 0.007524, 0.013859, 0.005472, 0.009672, 0.009180, 0.004518, 0.004326, 0.011827, 0.003077, 0.001889, 0.000623, 0.001578, 0.000632, 0.000441, 0.000516, 0.000229, 0.000249, 0.000095, 0.000085, 0.000045, 0.000025, 0.000027, 0.000017, 0.000013, 0.000012, 0.000012, 0.000013, 0.000010, 0.000009, 0.000011, 0.000009, 0.000009, 0.000008],
                  [0.097669, 0.203922, 1.000000, 0.964866, 0.459230, 0.102213, 0.135372, 0.253113, 0.314785, 0.184495, 0.075184, 0.111320, 0.197297, 0.424351, 0.036438, 0.046227, 0.281716, 0.172889, 0.040024, 0.073363, 0.045535, 0.051568, 0.129786, 0.031104, 0.022747, 0.025675, 0.054747, 0.013732, 0.003639, 0.031264, 0.013598, 0.007524, 0.013859, 0.005472, 0.009672, 0.009180, 0.004518, 0.004326, 0.011827, 0.003077, 0.001889, 0.000623, 0.001578, 0.000632, 0.000441, 0.000516, 0.000229, 0.000249, 0.000095, 0.000085, 0.000045, 0.000025, 0.000027, 0.000017, 0.000013, 0.000012, 0.000012, 0.000013, 0.000010, 0.000009, 0.000011, 0.000009, 0.000009, 0.000008],
                  [0.097669, 0.203922, 1.000000, 0.964866, 0.459230, 0.102213, 0.135372, 0.253113, 0.314785, 0.184495, 0.075184, 0.111320, 0.197297, 0.424351, 0.036438, 0.046227, 0.281716, 0.172889, 0.040024, 0.073363, 0.045535, 0.051568, 0.129786, 0.031104, 0.022747, 0.025675, 0.054747, 0.013732, 0.003639, 0.031264, 0.013598, 0.007524, 0.013859, 0.005472, 0.009672, 0.009180, 0.004518, 0.004326, 0.011827, 0.003077, 0.001889, 0.000623, 0.001578, 0.000632, 0.000441, 0.000516, 0.000229, 0.000249, 0.000095, 0.000085, 0.000045, 0.000025, 0.000027, 0.000017, 0.000013, 0.000012, 0.000012, 0.000013, 0.000010, 0.000009, 0.000011, 0.000009, 0.000009, 0.000008],
                  [0.097669, 0.203922, 1.000000, 0.964866, 0.459230, 0.102213, 0.135372, 0.253113, 0.314785, 0.184495, 0.075184, 0.111320, 0.197297, 0.424351, 0.036438, 0.046227, 0.281716, 0.172889, 0.040024, 0.073363, 0.045535, 0.051568, 0.129786, 0.031104, 0.022747, 0.025675, 0.054747, 0.013732, 0.003639, 0.031264, 0.013598, 0.007524, 0.013859, 0.005472, 0.009672, 0.009180, 0.004518, 0.004326, 0.011827, 0.003077, 0.001889, 0.000623, 0.001578, 0.000632, 0.000441, 0.000516, 0.000229, 0.000249, 0.000095, 0.000085, 0.000045, 0.000025, 0.000027, 0.000017, 0.000013, 0.000012, 0.000012, 0.000013, 0.000010, 0.000009, 0.000011, 0.000009, 0.000009, 0.000008],
                  [0.097669, 0.203922, 1.000000, 0.964866, 0.459230, 0.102213, 0.135372, 0.253113, 0.314785, 0.184495, 0.075184, 0.111320, 0.197297, 0.424351, 0.036438, 0.046227, 0.281716, 0.172889, 0.040024, 0.073363, 0.045535, 0.051568, 0.129786, 0.031104, 0.022747, 0.025675, 0.054747, 0.013732, 0.003639, 0.031264, 0.013598, 0.007524, 0.013859, 0.005472, 0.009672, 0.009180, 0.004518, 0.004326, 0.011827, 0.003077, 0.001889, 0.000623, 0.001578, 0.000632, 0.000441, 0.000516, 0.000229, 0.000249, 0.000095, 0.000085, 0.000045, 0.000025, 0.000027, 0.000017, 0.000013, 0.000012, 0.000012, 0.000013, 0.000010, 0.000009, 0.000011, 0.000009, 0.000009, 0.000008],
                  [0.097669, 0.203922, 1.000000, 0.964866, 0.459230, 0.102213, 0.135372, 0.253113, 0.314785, 0.184495, 0.075184, 0.111320, 0.197297, 0.424351, 0.036438, 0.046227, 0.281716, 0.172889, 0.040024, 0.073363, 0.045535, 0.051568, 0.129786, 0.031104, 0.022747, 0.025675, 0.054747, 0.013732, 0.003639, 0.031264, 0.013598, 0.007524, 0.013859, 0.005472, 0.009672, 0.009180, 0.004518, 0.004326, 0.011827, 0.003077, 0.001889, 0.000623, 0.001578, 0.000632, 0.000441, 0.000516, 0.000229, 0.000249, 0.000095, 0.000085, 0.000045, 0.000025, 0.000027, 0.000017, 0.000013, 0.000012, 0.000012, 0.000013, 0.000010, 0.000009, 0.000011, 0.000009, 0.000009, 0.000008],
                  [0.097669, 0.203922, 1.000000, 0.964866, 0.459230, 0.102213, 0.135372, 0.253113, 0.314785, 0.184495, 0.075184, 0.111320, 0.197297, 0.424351, 0.036438, 0.046227, 0.281716, 0.172889, 0.040024, 0.073363, 0.045535, 0.051568, 0.129786, 0.031104, 0.022747, 0.025675, 0.054747, 0.013732, 0.003639, 0.031264, 0.013598, 0.007524, 0.013859, 0.005472, 0.009672, 0.009180, 0.004518, 0.004326, 0.011827, 0.003077, 0.001889, 0.000623, 0.001578, 0.000632, 0.000441, 0.000516, 0.000229, 0.000249, 0.000095, 0.000085, 0.000045, 0.000025, 0.000027, 0.000017, 0.000013, 0.000012, 0.000012, 0.000013, 0.000010, 0.000009, 0.000011, 0.000009, 0.000009, 0.000008],
                  [0.026717, 0.235472, 1.000000, 0.728696, 0.515928, 0.260131, 0.145353, 0.312078, 0.353537, 0.132617, 0.042548, 0.136805, 0.120123, 0.065458, 0.084035, 0.158735, 0.055513, 0.032892, 0.036392, 0.041120, 0.061329, 0.099248, 0.048188, 0.024600, 0.022842, 0.019324, 0.008070, 0.022261, 0.015431, 0.014689, 0.010877, 0.017862, 0.004900, 0.005434, 0.004199, 0.009823, 0.004090, 0.010497, 0.006024, 0.001760, 0.004274, 0.002997, 0.001316, 0.000951, 0.001334, 0.000488, 0.000549, 0.000254, 0.000413, 0.000168, 0.000081, 0.000099, 0.000042, 0.000053, 0.000033, 0.000035, 0.000023, 0.000021, 0.000012, 0.000017, 0.000012, 0.000011, 0.000011, 0.000010],
                  [0.026717, 0.235472, 1.000000, 0.728696, 0.515928, 0.260131, 0.145353, 0.312078, 0.353537, 0.132617, 0.042548, 0.136805, 0.120123, 0.065458, 0.084035, 0.158735, 0.055513, 0.032892, 0.036392, 0.041120, 0.061329, 0.099248, 0.048188, 0.024600, 0.022842, 0.019324, 0.008070, 0.022261, 0.015431, 0.014689, 0.010877, 0.017862, 0.004900, 0.005434, 0.004199, 0.009823, 0.004090, 0.010497, 0.006024, 0.001760, 0.004274, 0.002997, 0.001316, 0.000951, 0.001334, 0.000488, 0.000549, 0.000254, 0.000413, 0.000168, 0.000081, 0.000099, 0.000042, 0.000053, 0.000033, 0.000035, 0.000023, 0.000021, 0.000012, 0.000017, 0.000012, 0.000011, 0.000011, 0.000010],
                  [0.026717, 0.235472, 1.000000, 0.728696, 0.515928, 0.260131, 0.145353, 0.312078, 0.353537, 0.132617, 0.042548, 0.136805, 0.120123, 0.065458, 0.084035, 0.158735, 0.055513, 0.032892, 0.036392, 0.041120, 0.061329, 0.099248, 0.048188, 0.024600, 0.022842, 0.019324, 0.008070, 0.022261, 0.015431, 0.014689, 0.010877, 0.017862, 0.004900, 0.005434, 0.004199, 0.009823, 0.004090, 0.010497, 0.006024, 0.001760, 0.004274, 0.002997, 0.001316, 0.000951, 0.001334, 0.000488, 0.000549, 0.000254, 0.000413, 0.000168, 0.000081, 0.000099, 0.000042, 0.000053, 0.000033, 0.000035, 0.000023, 0.000021, 0.000012, 0.000017, 0.000012, 0.000011, 0.000011, 0.000010],
                  [0.026717, 0.235472, 1.000000, 0.728696, 0.515928, 0.260131, 0.145353, 0.312078, 0.353537, 0.132617, 0.042548, 0.136805, 0.120123, 0.065458, 0.084035, 0.158735, 0.055513, 0.032892, 0.036392, 0.041120, 0.061329, 0.099248, 0.048188, 0.024600, 0.022842, 0.019324, 0.008070, 0.022261, 0.015431, 0.014689, 0.010877, 0.017862, 0.004900, 0.005434, 0.004199, 0.009823, 0.004090, 0.010497, 0.006024, 0.001760, 0.004274, 0.002997, 0.001316, 0.000951, 0.001334, 0.000488, 0.000549, 0.000254, 0.000413, 0.000168, 0.000081, 0.000099, 0.000042, 0.000053, 0.000033, 0.000035, 0.000023, 0.000021, 0.000012, 0.000017, 0.000012, 0.000011, 0.000011, 0.000010],
                  [0.026717, 0.235472, 1.000000, 0.728696, 0.515928, 0.260131, 0.145353, 0.312078, 0.353537, 0.132617, 0.042548, 0.136805, 0.120123, 0.065458, 0.084035, 0.158735, 0.055513, 0.032892, 0.036392, 0.041120, 0.061329, 0.099248, 0.048188, 0.024600, 0.022842, 0.019324, 0.008070, 0.022261, 0.015431, 0.014689, 0.010877, 0.017862, 0.004900, 0.005434, 0.004199, 0.009823, 0.004090, 0.010497, 0.006024, 0.001760, 0.004274, 0.002997, 0.001316, 0.000951, 0.001334, 0.000488, 0.000549, 0.000254, 0.000413, 0.000168, 0.000081, 0.000099, 0.000042, 0.000053, 0.000033, 0.000035, 0.000023, 0.000021, 0.000012, 0.000017, 0.000012, 0.000011, 0.000011, 0.000010],
                  [0.026717, 0.235472, 1.000000, 0.728696, 0.515928, 0.260131, 0.145353, 0.312078, 0.353537, 0.132617, 0.042548, 0.136805, 0.120123, 0.065458, 0.084035, 0.158735, 0.055513, 0.032892, 0.036392, 0.041120, 0.061329, 0.099248, 0.048188, 0.024600, 0.022842, 0.019324, 0.008070, 0.022261, 0.015431, 0.014689, 0.010877, 0.017862, 0.004900, 0.005434, 0.004199, 0.009823, 0.004090, 0.010497, 0.006024, 0.001760, 0.004274, 0.002997, 0.001316, 0.000951, 0.001334, 0.000488, 0.000549, 0.000254, 0.000413, 0.000168, 0.000081, 0.000099, 0.000042, 0.000053, 0.000033, 0.000035, 0.000023, 0.000021, 0.000012, 0.000017, 0.000012, 0.000011, 0.000011, 0.000010],
                  [0.026717, 0.235472, 1.000000, 0.728696, 0.515928, 0.260131, 0.145353, 0.312078, 0.353537, 0.132617, 0.042548, 0.136805, 0.120123, 0.065458, 0.084035, 0.158735, 0.055513, 0.032892, 0.036392, 0.041120, 0.061329, 0.099248, 0.048188, 0.024600, 0.022842, 0.019324, 0.008070, 0.022261, 0.015431, 0.014689, 0.010877, 0.017862, 0.004900, 0.005434, 0.004199, 0.009823, 0.004090, 0.010497, 0.006024, 0.001760, 0.004274, 0.002997, 0.001316, 0.000951, 0.001334, 0.000488, 0.000549, 0.000254, 0.000413, 0.000168, 0.000081, 0.000099, 0.000042, 0.000053, 0.000033, 0.000035, 0.000023, 0.000021, 0.000012, 0.000017, 0.000012, 0.000011, 0.000011, 0.000010],
                  [0.026717, 0.235472, 1.000000, 0.728696, 0.515928, 0.260131, 0.145353, 0.312078, 0.353537, 0.132617, 0.042548, 0.136805, 0.120123, 0.065458, 0.084035, 0.158735, 0.055513, 0.032892, 0.036392, 0.041120, 0.061329, 0.099248, 0.048188, 0.024600, 0.022842, 0.019324, 0.008070, 0.022261, 0.015431, 0.014689, 0.010877, 0.017862, 0.004900, 0.005434, 0.004199, 0.009823, 0.004090, 0.010497, 0.006024, 0.001760, 0.004274, 0.002997, 0.001316, 0.000951, 0.001334, 0.000488, 0.000549, 0.000254, 0.000413, 0.000168, 0.000081, 0.000099, 0.000042, 0.000053, 0.000033, 0.000035, 0.000023, 0.000021, 0.000012, 0.000017, 0.000012, 0.000011, 0.000011, 0.000010],
                  [0.026717, 0.235472, 1.000000, 0.728696, 0.515928, 0.260131, 0.145353, 0.312078, 0.353537, 0.132617, 0.042548, 0.136805, 0.120123, 0.065458, 0.084035, 0.158735, 0.055513, 0.032892, 0.036392, 0.041120, 0.061329, 0.099248, 0.048188, 0.024600, 0.022842, 0.019324, 0.008070, 0.022261, 0.015431, 0.014689, 0.010877, 0.017862, 0.004900, 0.005434, 0.004199, 0.009823, 0.004090, 0.010497, 0.006024, 0.001760, 0.004274, 0.002997, 0.001316, 0.000951, 0.001334, 0.000488, 0.000549, 0.000254, 0.000413, 0.000168, 0.000081, 0.000099, 0.000042, 0.000053, 0.000033, 0.000035, 0.000023, 0.000021, 0.000012, 0.000017, 0.000012, 0.000011, 0.000011, 0.000010],
                  [0.026717, 0.235472, 1.000000, 0.728696, 0.515928, 0.260131, 0.145353, 0.312078, 0.353537, 0.132617, 0.042548, 0.136805, 0.120123, 0.065458, 0.084035, 0.158735, 0.055513, 0.032892, 0.036392, 0.041120, 0.061329, 0.099248, 0.048188, 0.024600, 0.022842, 0.019324, 0.008070, 0.022261, 0.015431, 0.014689, 0.010877, 0.017862, 0.004900, 0.005434, 0.004199, 0.009823, 0.004090, 0.010497, 0.006024, 0.001760, 0.004274, 0.002997, 0.001316, 0.000951, 0.001334, 0.000488, 0.000549, 0.000254, 0.000413, 0.000168, 0.000081, 0.000099, 0.000042, 0.000053, 0.000033, 0.000035, 0.000023, 0.000021, 0.000012, 0.000017, 0.000012, 0.000011, 0.000011, 0.000010],
                  [0.026717, 0.235472, 1.000000, 0.728696, 0.515928, 0.260131, 0.145353, 0.312078, 0.353537, 0.132617, 0.042548, 0.136805, 0.120123, 0.065458, 0.084035, 0.158735, 0.055513, 0.032892, 0.036392, 0.041120, 0.061329, 0.099248, 0.048188, 0.024600, 0.022842, 0.019324, 0.008070, 0.022261, 0.015431, 0.014689, 0.010877, 0.017862, 0.004900, 0.005434, 0.004199, 0.009823, 0.004090, 0.010497, 0.006024, 0.001760, 0.004274, 0.002997, 0.001316, 0.000951, 0.001334, 0.000488, 0.000549, 0.000254, 0.000413, 0.000168, 0.000081, 0.000099, 0.000042, 0.000053, 0.000033, 0.000035, 0.000023, 0.000021, 0.000012, 0.000017, 0.000012, 0.000011, 0.000011, 0.000010],
                  [0.026717, 0.235472, 1.000000, 0.728696, 0.515928, 0.260131, 0.145353, 0.312078, 0.353537, 0.132617, 0.042548, 0.136805, 0.120123, 0.065458, 0.084035, 0.158735, 0.055513, 0.032892, 0.036392, 0.041120, 0.061329, 0.099248, 0.048188, 0.024600, 0.022842, 0.019324, 0.008070, 0.022261, 0.015431, 0.014689, 0.010877, 0.017862, 0.004900, 0.005434, 0.004199, 0.009823, 0.004090, 0.010497, 0.006024, 0.001760, 0.004274, 0.002997, 0.001316, 0.000951, 0.001334, 0.000488, 0.000549, 0.000254, 0.000413, 0.000168, 0.000081, 0.000099, 0.000042, 0.000053, 0.000033, 0.000035, 0.000023, 0.000021, 0.000012, 0.000017, 0.000012, 0.000011, 0.000011, 0.000010],
                  [0.474882, 0.559165, 0.753305, 0.962662, 1.000000, 0.903066, 0.803115, 0.748720, 0.623323, 0.415379, 0.287977, 0.246387, 0.182959, 0.105115, 0.096228, 0.139240, 0.107159, 0.079642, 0.130498, 0.112418, 0.067649, 0.071145, 0.082864, 0.035909, 0.093766, 0.100572, 0.030493, 0.026670, 0.034083, 0.059199, 0.019766, 0.049920, 0.019306, 0.015909, 0.024513, 0.032658, 0.016280, 0.006376, 0.004377, 0.005736, 0.009051, 0.015734, 0.004334, 0.004351, 0.003543, 0.001201, 0.003786, 0.001516, 0.000689, 0.000452, 0.000626, 0.000426, 0.000155, 0.000270, 0.000177, 0.000160, 0.000103, 0.000097, 0.000085, 0.000073, 0.000067, 0.000043, 0.000047, 0.000052],
                  [0.474882, 0.559165, 0.753305, 0.962662, 1.000000, 0.903066, 0.803115, 0.748720, 0.623323, 0.415379, 0.287977, 0.246387, 0.182959, 0.105115, 0.096228, 0.139240, 0.107159, 0.079642, 0.130498, 0.112418, 0.067649, 0.071145, 0.082864, 0.035909, 0.093766, 0.100572, 0.030493, 0.026670, 0.034083, 0.059199, 0.019766, 0.049920, 0.019306, 0.015909, 0.024513, 0.032658, 0.016280, 0.006376, 0.004377, 0.005736, 0.009051, 0.015734, 0.004334, 0.004351, 0.003543, 0.001201, 0.003786, 0.001516, 0.000689, 0.000452, 0.000626, 0.000426, 0.000155, 0.000270, 0.000177, 0.000160, 0.000103, 0.000097, 0.000085, 0.000073, 0.000067, 0.000043, 0.000047, 0.000052],
                  [0.474882, 0.559165, 0.753305, 0.962662, 1.000000, 0.903066, 0.803115, 0.748720, 0.623323, 0.415379, 0.287977, 0.246387, 0.182959, 0.105115, 0.096228, 0.139240, 0.107159, 0.079642, 0.130498, 0.112418, 0.067649, 0.071145, 0.082864, 0.035909, 0.093766, 0.100572, 0.030493, 0.026670, 0.034083, 0.059199, 0.019766, 0.049920, 0.019306, 0.015909, 0.024513, 0.032658, 0.016280, 0.006376, 0.004377, 0.005736, 0.009051, 0.015734, 0.004334, 0.004351, 0.003543, 0.001201, 0.003786, 0.001516, 0.000689, 0.000452, 0.000626, 0.000426, 0.000155, 0.000270, 0.000177, 0.000160, 0.000103, 0.000097, 0.000085, 0.000073, 0.000067, 0.000043, 0.000047, 0.000052],
                  [0.474882, 0.559165, 0.753305, 0.962662, 1.000000, 0.903066, 0.803115, 0.748720, 0.623323, 0.415379, 0.287977, 0.246387, 0.182959, 0.105115, 0.096228, 0.139240, 0.107159, 0.079642, 0.130498, 0.112418, 0.067649, 0.071145, 0.082864, 0.035909, 0.093766, 0.100572, 0.030493, 0.026670, 0.034083, 0.059199, 0.019766, 0.049920, 0.019306, 0.015909, 0.024513, 0.032658, 0.016280, 0.006376, 0.004377, 0.005736, 0.009051, 0.015734, 0.004334, 0.004351, 0.003543, 0.001201, 0.003786, 0.001516, 0.000689, 0.000452, 0.000626, 0.000426, 0.000155, 0.000270, 0.000177, 0.000160, 0.000103, 0.000097, 0.000085, 0.000073, 0.000067, 0.000043, 0.000047, 0.000052],
                  [0.474882, 0.559165, 0.753305, 0.962662, 1.000000, 0.903066, 0.803115, 0.748720, 0.623323, 0.415379, 0.287977, 0.246387, 0.182959, 0.105115, 0.096228, 0.139240, 0.107159, 0.079642, 0.130498, 0.112418, 0.067649, 0.071145, 0.082864, 0.035909, 0.093766, 0.100572, 0.030493, 0.026670, 0.034083, 0.059199, 0.019766, 0.049920, 0.019306, 0.015909, 0.024513, 0.032658, 0.016280, 0.006376, 0.004377, 0.005736, 0.009051, 0.015734, 0.004334, 0.004351, 0.003543, 0.001201, 0.003786, 0.001516, 0.000689, 0.000452, 0.000626, 0.000426, 0.000155, 0.000270, 0.000177, 0.000160, 0.000103, 0.000097, 0.000085, 0.000073, 0.000067, 0.000043, 0.000047, 0.000052],
                  [0.474882, 0.559165, 0.753305, 0.962662, 1.000000, 0.903066, 0.803115, 0.748720, 0.623323, 0.415379, 0.287977, 0.246387, 0.182959, 0.105115, 0.096228, 0.139240, 0.107159, 0.079642, 0.130498, 0.112418, 0.067649, 0.071145, 0.082864, 0.035909, 0.093766, 0.100572, 0.030493, 0.026670, 0.034083, 0.059199, 0.019766, 0.049920, 0.019306, 0.015909, 0.024513, 0.032658, 0.016280, 0.006376, 0.004377, 0.005736, 0.009051, 0.015734, 0.004334, 0.004351, 0.003543, 0.001201, 0.003786, 0.001516, 0.000689, 0.000452, 0.000626, 0.000426, 0.000155, 0.000270, 0.000177, 0.000160, 0.000103, 0.000097, 0.000085, 0.000073, 0.000067, 0.000043, 0.000047, 0.000052],
                  [0.474882, 0.559165, 0.753305, 0.962662, 1.000000, 0.903066, 0.803115, 0.748720, 0.623323, 0.415379, 0.287977, 0.246387, 0.182959, 0.105115, 0.096228, 0.139240, 0.107159, 0.079642, 0.130498, 0.112418, 0.067649, 0.071145, 0.082864, 0.035909, 0.093766, 0.100572, 0.030493, 0.026670, 0.034083, 0.059199, 0.019766, 0.049920, 0.019306, 0.015909, 0.024513, 0.032658, 0.016280, 0.006376, 0.004377, 0.005736, 0.009051, 0.015734, 0.004334, 0.004351, 0.003543, 0.001201, 0.003786, 0.001516, 0.000689, 0.000452, 0.000626, 0.000426, 0.000155, 0.000270, 0.000177, 0.000160, 0.000103, 0.000097, 0.000085, 0.000073, 0.000067, 0.000043, 0.000047, 0.000052],
                  [0.474882, 0.559165, 0.753305, 0.962662, 1.000000, 0.903066, 0.803115, 0.748720, 0.623323, 0.415379, 0.287977, 0.246387, 0.182959, 0.105115, 0.096228, 0.139240, 0.107159, 0.079642, 0.130498, 0.112418, 0.067649, 0.071145, 0.082864, 0.035909, 0.093766, 0.100572, 0.030493, 0.026670, 0.034083, 0.059199, 0.019766, 0.049920, 0.019306, 0.015909, 0.024513, 0.032658, 0.016280, 0.006376, 0.004377, 0.005736, 0.009051, 0.015734, 0.004334, 0.004351, 0.003543, 0.001201, 0.003786, 0.001516, 0.000689, 0.000452, 0.000626, 0.000426, 0.000155, 0.000270, 0.000177, 0.000160, 0.000103, 0.000097, 0.000085, 0.000073, 0.000067, 0.000043, 0.000047, 0.000052],
                  [0.474882, 0.559165, 0.753305, 0.962662, 1.000000, 0.903066, 0.803115, 0.748720, 0.623323, 0.415379, 0.287977, 0.246387, 0.182959, 0.105115, 0.096228, 0.139240, 0.107159, 0.079642, 0.130498, 0.112418, 0.067649, 0.071145, 0.082864, 0.035909, 0.093766, 0.100572, 0.030493, 0.026670, 0.034083, 0.059199, 0.019766, 0.049920, 0.019306, 0.015909, 0.024513, 0.032658, 0.016280, 0.006376, 0.004377, 0.005736, 0.009051, 0.015734, 0.004334, 0.004351, 0.003543, 0.001201, 0.003786, 0.001516, 0.000689, 0.000452, 0.000626, 0.000426, 0.000155, 0.000270, 0.000177, 0.000160, 0.000103, 0.000097, 0.000085, 0.000073, 0.000067, 0.000043, 0.000047, 0.000052],
                  [0.474882, 0.559165, 0.753305, 0.962662, 1.000000, 0.903066, 0.803115, 0.748720, 0.623323, 0.415379, 0.287977, 0.246387, 0.182959, 0.105115, 0.096228, 0.139240, 0.107159, 0.079642, 0.130498, 0.112418, 0.067649, 0.071145, 0.082864, 0.035909, 0.093766, 0.100572, 0.030493, 0.026670, 0.034083, 0.059199, 0.019766, 0.049920, 0.019306, 0.015909, 0.024513, 0.032658, 0.016280, 0.006376, 0.004377, 0.005736, 0.009051, 0.015734, 0.004334, 0.004351, 0.003543, 0.001201, 0.003786, 0.001516, 0.000689, 0.000452, 0.000626, 0.000426, 0.000155, 0.000270, 0.000177, 0.000160, 0.000103, 0.000097, 0.000085, 0.000073, 0.000067, 0.000043, 0.000047, 0.000052],
                  [0.474882, 0.559165, 0.753305, 0.962662, 1.000000, 0.903066, 0.803115, 0.748720, 0.623323, 0.415379, 0.287977, 0.246387, 0.182959, 0.105115, 0.096228, 0.139240, 0.107159, 0.079642, 0.130498, 0.112418, 0.067649, 0.071145, 0.082864, 0.035909, 0.093766, 0.100572, 0.030493, 0.026670, 0.034083, 0.059199, 0.019766, 0.049920, 0.019306, 0.015909, 0.024513, 0.032658, 0.016280, 0.006376, 0.004377, 0.005736, 0.009051, 0.015734, 0.004334, 0.004351, 0.003543, 0.001201, 0.003786, 0.001516, 0.000689, 0.000452, 0.000626, 0.000426, 0.000155, 0.000270, 0.000177, 0.000160, 0.000103, 0.000097, 0.000085, 0.000073, 0.000067, 0.000043, 0.000047, 0.000052],
                  [0.474882, 0.559165, 0.753305, 0.962662, 1.000000, 0.903066, 0.803115, 0.748720, 0.623323, 0.415379, 0.287977, 0.246387, 0.182959, 0.105115, 0.096228, 0.139240, 0.107159, 0.079642, 0.130498, 0.112418, 0.067649, 0.071145, 0.082864, 0.035909, 0.093766, 0.100572, 0.030493, 0.026670, 0.034083, 0.059199, 0.019766, 0.049920, 0.019306, 0.015909, 0.024513, 0.032658, 0.016280, 0.006376, 0.004377, 0.005736, 0.009051, 0.015734, 0.004334, 0.004351, 0.003543, 0.001201, 0.003786, 0.001516, 0.000689, 0.000452, 0.000626, 0.000426, 0.000155, 0.000270, 0.000177, 0.000160, 0.000103, 0.000097, 0.000085, 0.000073, 0.000067, 0.000043, 0.000047, 0.000052],
                  [0.247809, 0.264779, 0.311274, 0.399226, 0.514935, 0.679622, 0.867416, 0.994891, 1.000000, 0.863289, 0.667250, 0.482304, 0.347116, 0.264899, 0.208997, 0.165129, 0.128784, 0.102452, 0.088879, 0.085824, 0.087966, 0.090898, 0.093489, 0.088021, 0.064411, 0.041132, 0.039479, 0.056609, 0.046374, 0.024202, 0.026079, 0.030523, 0.035425, 0.060595, 0.033937, 0.027998, 0.029729, 0.018848, 0.014769, 0.010514, 0.007804, 0.008786, 0.007725, 0.008755, 0.009000, 0.013165, 0.005131, 0.004187, 0.003578, 0.001404, 0.000906, 0.001181, 0.000803, 0.000488, 0.000255, 0.000159, 0.000186, 0.000133, 0.000117, 0.000085, 0.000087, 0.000077, 0.000071, 0.000058],
                  [0.247809, 0.264779, 0.311274, 0.399226, 0.514935, 0.679622, 0.867416, 0.994891, 1.000000, 0.863289, 0.667250, 0.482304, 0.347116, 0.264899, 0.208997, 0.165129, 0.128784, 0.102452, 0.088879, 0.085824, 0.087966, 0.090898, 0.093489, 0.088021, 0.064411, 0.041132, 0.039479, 0.056609, 0.046374, 0.024202, 0.026079, 0.030523, 0.035425, 0.060595, 0.033937, 0.027998, 0.029729, 0.018848, 0.014769, 0.010514, 0.007804, 0.008786, 0.007725, 0.008755, 0.009000, 0.013165, 0.005131, 0.004187, 0.003578, 0.001404, 0.000906, 0.001181, 0.000803, 0.000488, 0.000255, 0.000159, 0.000186, 0.000133, 0.000117, 0.000085, 0.000087, 0.000077, 0.000071, 0.000058],
                  [0.247809, 0.264779, 0.311274, 0.399226, 0.514935, 0.679622, 0.867416, 0.994891, 1.000000, 0.863289, 0.667250, 0.482304, 0.347116, 0.264899, 0.208997, 0.165129, 0.128784, 0.102452, 0.088879, 0.085824, 0.087966, 0.090898, 0.093489, 0.088021, 0.064411, 0.041132, 0.039479, 0.056609, 0.046374, 0.024202, 0.026079, 0.030523, 0.035425, 0.060595, 0.033937, 0.027998, 0.029729, 0.018848, 0.014769, 0.010514, 0.007804, 0.008786, 0.007725, 0.008755, 0.009000, 0.013165, 0.005131, 0.004187, 0.003578, 0.001404, 0.000906, 0.001181, 0.000803, 0.000488, 0.000255, 0.000159, 0.000186, 0.000133, 0.000117, 0.000085, 0.000087, 0.000077, 0.000071, 0.000058],
                  [0.247809, 0.264779, 0.311274, 0.399226, 0.514935, 0.679622, 0.867416, 0.994891, 1.000000, 0.863289, 0.667250, 0.482304, 0.347116, 0.264899, 0.208997, 0.165129, 0.128784, 0.102452, 0.088879, 0.085824, 0.087966, 0.090898, 0.093489, 0.088021, 0.064411, 0.041132, 0.039479, 0.056609, 0.046374, 0.024202, 0.026079, 0.030523, 0.035425, 0.060595, 0.033937, 0.027998, 0.029729, 0.018848, 0.014769, 0.010514, 0.007804, 0.008786, 0.007725, 0.008755, 0.009000, 0.013165, 0.005131, 0.004187, 0.003578, 0.001404, 0.000906, 0.001181, 0.000803, 0.000488, 0.000255, 0.000159, 0.000186, 0.000133, 0.000117, 0.000085, 0.000087, 0.000077, 0.000071, 0.000058],
                  [0.247809, 0.264779, 0.311274, 0.399226, 0.514935, 0.679622, 0.867416, 0.994891, 1.000000, 0.863289, 0.667250, 0.482304, 0.347116, 0.264899, 0.208997, 0.165129, 0.128784, 0.102452, 0.088879, 0.085824, 0.087966, 0.090898, 0.093489, 0.088021, 0.064411, 0.041132, 0.039479, 0.056609, 0.046374, 0.024202, 0.026079, 0.030523, 0.035425, 0.060595, 0.033937, 0.027998, 0.029729, 0.018848, 0.014769, 0.010514, 0.007804, 0.008786, 0.007725, 0.008755, 0.009000, 0.013165, 0.005131, 0.004187, 0.003578, 0.001404, 0.000906, 0.001181, 0.000803, 0.000488, 0.000255, 0.000159, 0.000186, 0.000133, 0.000117, 0.000085, 0.000087, 0.000077, 0.000071, 0.000058],
                  [0.247809, 0.264779, 0.311274, 0.399226, 0.514935, 0.679622, 0.867416, 0.994891, 1.000000, 0.863289, 0.667250, 0.482304, 0.347116, 0.264899, 0.208997, 0.165129, 0.128784, 0.102452, 0.088879, 0.085824, 0.087966, 0.090898, 0.093489, 0.088021, 0.064411, 0.041132, 0.039479, 0.056609, 0.046374, 0.024202, 0.026079, 0.030523, 0.035425, 0.060595, 0.033937, 0.027998, 0.029729, 0.018848, 0.014769, 0.010514, 0.007804, 0.008786, 0.007725, 0.008755, 0.009000, 0.013165, 0.005131, 0.004187, 0.003578, 0.001404, 0.000906, 0.001181, 0.000803, 0.000488, 0.000255, 0.000159, 0.000186, 0.000133, 0.000117, 0.000085, 0.000087, 0.000077, 0.000071, 0.000058],
                  [0.247809, 0.264779, 0.311274, 0.399226, 0.514935, 0.679622, 0.867416, 0.994891, 1.000000, 0.863289, 0.667250, 0.482304, 0.347116, 0.264899, 0.208997, 0.165129, 0.128784, 0.102452, 0.088879, 0.085824, 0.087966, 0.090898, 0.093489, 0.088021, 0.064411, 0.041132, 0.039479, 0.056609, 0.046374, 0.024202, 0.026079, 0.030523, 0.035425, 0.060595, 0.033937, 0.027998, 0.029729, 0.018848, 0.014769, 0.010514, 0.007804, 0.008786, 0.007725, 0.008755, 0.009000, 0.013165, 0.005131, 0.004187, 0.003578, 0.001404, 0.000906, 0.001181, 0.000803, 0.000488, 0.000255, 0.000159, 0.000186, 0.000133, 0.000117, 0.000085, 0.000087, 0.000077, 0.000071, 0.000058],
                  [0.247809, 0.264779, 0.311274, 0.399226, 0.514935, 0.679622, 0.867416, 0.994891, 1.000000, 0.863289, 0.667250, 0.482304, 0.347116, 0.264899, 0.208997, 0.165129, 0.128784, 0.102452, 0.088879, 0.085824, 0.087966, 0.090898, 0.093489, 0.088021, 0.064411, 0.041132, 0.039479, 0.056609, 0.046374, 0.024202, 0.026079, 0.030523, 0.035425, 0.060595, 0.033937, 0.027998, 0.029729, 0.018848, 0.014769, 0.010514, 0.007804, 0.008786, 0.007725, 0.008755, 0.009000, 0.013165, 0.005131, 0.004187, 0.003578, 0.001404, 0.000906, 0.001181, 0.000803, 0.000488, 0.000255, 0.000159, 0.000186, 0.000133, 0.000117, 0.000085, 0.000087, 0.000077, 0.000071, 0.000058],
                  [0.247809, 0.264779, 0.311274, 0.399226, 0.514935, 0.679622, 0.867416, 0.994891, 1.000000, 0.863289, 0.667250, 0.482304, 0.347116, 0.264899, 0.208997, 0.165129, 0.128784, 0.102452, 0.088879, 0.085824, 0.087966, 0.090898, 0.093489, 0.088021, 0.064411, 0.041132, 0.039479, 0.056609, 0.046374, 0.024202, 0.026079, 0.030523, 0.035425, 0.060595, 0.033937, 0.027998, 0.029729, 0.018848, 0.014769, 0.010514, 0.007804, 0.008786, 0.007725, 0.008755, 0.009000, 0.013165, 0.005131, 0.004187, 0.003578, 0.001404, 0.000906, 0.001181, 0.000803, 0.000488, 0.000255, 0.000159, 0.000186, 0.000133, 0.000117, 0.000085, 0.000087, 0.000077, 0.000071, 0.000058],
                  [0.247809, 0.264779, 0.311274, 0.399226, 0.514935, 0.679622, 0.867416, 0.994891, 1.000000, 0.863289, 0.667250, 0.482304, 0.347116, 0.264899, 0.208997, 0.165129, 0.128784, 0.102452, 0.088879, 0.085824, 0.087966, 0.090898, 0.093489, 0.088021, 0.064411, 0.041132, 0.039479, 0.056609, 0.046374, 0.024202, 0.026079, 0.030523, 0.035425, 0.060595, 0.033937, 0.027998, 0.029729, 0.018848, 0.014769, 0.010514, 0.007804, 0.008786, 0.007725, 0.008755, 0.009000, 0.013165, 0.005131, 0.004187, 0.003578, 0.001404, 0.000906, 0.001181, 0.000803, 0.000488, 0.000255, 0.000159, 0.000186, 0.000133, 0.000117, 0.000085, 0.000087, 0.000077, 0.000071, 0.000058],
                  [0.247809, 0.264779, 0.311274, 0.399226, 0.514935, 0.679622, 0.867416, 0.994891, 1.000000, 0.863289, 0.667250, 0.482304, 0.347116, 0.264899, 0.208997, 0.165129, 0.128784, 0.102452, 0.088879, 0.085824, 0.087966, 0.090898, 0.093489, 0.088021, 0.064411, 0.041132, 0.039479, 0.056609, 0.046374, 0.024202, 0.026079, 0.030523, 0.035425, 0.060595, 0.033937, 0.027998, 0.029729, 0.018848, 0.014769, 0.010514, 0.007804, 0.008786, 0.007725, 0.008755, 0.009000, 0.013165, 0.005131, 0.004187, 0.003578, 0.001404, 0.000906, 0.001181, 0.000803, 0.000488, 0.000255, 0.000159, 0.000186, 0.000133, 0.000117, 0.000085, 0.000087, 0.000077, 0.000071, 0.000058],
                  [0.247809, 0.264779, 0.311274, 0.399226, 0.514935, 0.679622, 0.867416, 0.994891, 1.000000, 0.863289, 0.667250, 0.482304, 0.347116, 0.264899, 0.208997, 0.165129, 0.128784, 0.102452, 0.088879, 0.085824, 0.087966, 0.090898, 0.093489, 0.088021, 0.064411, 0.041132, 0.039479, 0.056609, 0.046374, 0.024202, 0.026079, 0.030523, 0.035425, 0.060595, 0.033937, 0.027998, 0.029729, 0.018848, 0.014769, 0.010514, 0.007804, 0.008786, 0.007725, 0.008755, 0.009000, 0.013165, 0.005131, 0.004187, 0.003578, 0.001404, 0.000906, 0.001181, 0.000803, 0.000488, 0.000255, 0.000159, 0.000186, 0.000133, 0.000117, 0.000085, 0.000087, 0.000077, 0.000071, 0.000058],
                  [0.670719, 0.676117, 0.689948, 0.713760, 0.742241, 0.781210, 0.830391, 0.882027, 0.935181, 0.979704, 1.000000, 0.985546, 0.926684, 0.830646, 0.705098, 0.569544, 0.444362, 0.342731, 0.269568, 0.221739, 0.193749, 0.178210, 0.167409, 0.152797, 0.130257, 0.102677, 0.077068, 0.060066, 0.053721, 0.056799, 0.065154, 0.068981, 0.059617, 0.043592, 0.032413, 0.029936, 0.032977, 0.033875, 0.028997, 0.024594, 0.022125, 0.016720, 0.013682, 0.020969, 0.023380, 0.010585, 0.007082, 0.005682, 0.004304, 0.004051, 0.002125, 0.001418, 0.000863, 0.000490, 0.000400, 0.000743, 0.000338, 0.000282, 0.000223, 0.000213, 0.000167, 0.000203, 0.000151, 0.000165],
                  [0.670719, 0.676117, 0.689948, 0.713760, 0.742241, 0.781210, 0.830391, 0.882027, 0.935181, 0.979704, 1.000000, 0.985546, 0.926684, 0.830646, 0.705098, 0.569544, 0.444362, 0.342731, 0.269568, 0.221739, 0.193749, 0.178210, 0.167409, 0.152797, 0.130257, 0.102677, 0.077068, 0.060066, 0.053721, 0.056799, 0.065154, 0.068981, 0.059617, 0.043592, 0.032413, 0.029936, 0.032977, 0.033875, 0.028997, 0.024594, 0.022125, 0.016720, 0.013682, 0.020969, 0.023380, 0.010585, 0.007082, 0.005682, 0.004304, 0.004051, 0.002125, 0.001418, 0.000863, 0.000490, 0.000400, 0.000743, 0.000338, 0.000282, 0.000223, 0.000213, 0.000167, 0.000203, 0.000151, 0.000165],
                  [0.670719, 0.676117, 0.689948, 0.713760, 0.742241, 0.781210, 0.830391, 0.882027, 0.935181, 0.979704, 1.000000, 0.985546, 0.926684, 0.830646, 0.705098, 0.569544, 0.444362, 0.342731, 0.269568, 0.221739, 0.193749, 0.178210, 0.167409, 0.152797, 0.130257, 0.102677, 0.077068, 0.060066, 0.053721, 0.056799, 0.065154, 0.068981, 0.059617, 0.043592, 0.032413, 0.029936, 0.032977, 0.033875, 0.028997, 0.024594, 0.022125, 0.016720, 0.013682, 0.020969, 0.023380, 0.010585, 0.007082, 0.005682, 0.004304, 0.004051, 0.002125, 0.001418, 0.000863, 0.000490, 0.000400, 0.000743, 0.000338, 0.000282, 0.000223, 0.000213, 0.000167, 0.000203, 0.000151, 0.000165],
                  [0.670719, 0.676117, 0.689948, 0.713760, 0.742241, 0.781210, 0.830391, 0.882027, 0.935181, 0.979704, 1.000000, 0.985546, 0.926684, 0.830646, 0.705098, 0.569544, 0.444362, 0.342731, 0.269568, 0.221739, 0.193749, 0.178210, 0.167409, 0.152797, 0.130257, 0.102677, 0.077068, 0.060066, 0.053721, 0.056799, 0.065154, 0.068981, 0.059617, 0.043592, 0.032413, 0.029936, 0.032977, 0.033875, 0.028997, 0.024594, 0.022125, 0.016720, 0.013682, 0.020969, 0.023380, 0.010585, 0.007082, 0.005682, 0.004304, 0.004051, 0.002125, 0.001418, 0.000863, 0.000490, 0.000400, 0.000743, 0.000338, 0.000282, 0.000223, 0.000213, 0.000167, 0.000203, 0.000151, 0.000165],
                  [0.670719, 0.676117, 0.689948, 0.713760, 0.742241, 0.781210, 0.830391, 0.882027, 0.935181, 0.979704, 1.000000, 0.985546, 0.926684, 0.830646, 0.705098, 0.569544, 0.444362, 0.342731, 0.269568, 0.221739, 0.193749, 0.178210, 0.167409, 0.152797, 0.130257, 0.102677, 0.077068, 0.060066, 0.053721, 0.056799, 0.065154, 0.068981, 0.059617, 0.043592, 0.032413, 0.029936, 0.032977, 0.033875, 0.028997, 0.024594, 0.022125, 0.016720, 0.013682, 0.020969, 0.023380, 0.010585, 0.007082, 0.005682, 0.004304, 0.004051, 0.002125, 0.001418, 0.000863, 0.000490, 0.000400, 0.000743, 0.000338, 0.000282, 0.000223, 0.000213, 0.000167, 0.000203, 0.000151, 0.000165],
                  [0.670719, 0.676117, 0.689948, 0.713760, 0.742241, 0.781210, 0.830391, 0.882027, 0.935181, 0.979704, 1.000000, 0.985546, 0.926684, 0.830646, 0.705098, 0.569544, 0.444362, 0.342731, 0.269568, 0.221739, 0.193749, 0.178210, 0.167409, 0.152797, 0.130257, 0.102677, 0.077068, 0.060066, 0.053721, 0.056799, 0.065154, 0.068981, 0.059617, 0.043592, 0.032413, 0.029936, 0.032977, 0.033875, 0.028997, 0.024594, 0.022125, 0.016720, 0.013682, 0.020969, 0.023380, 0.010585, 0.007082, 0.005682, 0.004304, 0.004051, 0.002125, 0.001418, 0.000863, 0.000490, 0.000400, 0.000743, 0.000338, 0.000282, 0.000223, 0.000213, 0.000167, 0.000203, 0.000151, 0.000165],
                  [0.670719, 0.676117, 0.689948, 0.713760, 0.742241, 0.781210, 0.830391, 0.882027, 0.935181, 0.979704, 1.000000, 0.985546, 0.926684, 0.830646, 0.705098, 0.569544, 0.444362, 0.342731, 0.269568, 0.221739, 0.193749, 0.178210, 0.167409, 0.152797, 0.130257, 0.102677, 0.077068, 0.060066, 0.053721, 0.056799, 0.065154, 0.068981, 0.059617, 0.043592, 0.032413, 0.029936, 0.032977, 0.033875, 0.028997, 0.024594, 0.022125, 0.016720, 0.013682, 0.020969, 0.023380, 0.010585, 0.007082, 0.005682, 0.004304, 0.004051, 0.002125, 0.001418, 0.000863, 0.000490, 0.000400, 0.000743, 0.000338, 0.000282, 0.000223, 0.000213, 0.000167, 0.000203, 0.000151, 0.000165],
                  [0.670719, 0.676117, 0.689948, 0.713760, 0.742241, 0.781210, 0.830391, 0.882027, 0.935181, 0.979704, 1.000000, 0.985546, 0.926684, 0.830646, 0.705098, 0.569544, 0.444362, 0.342731, 0.269568, 0.221739, 0.193749, 0.178210, 0.167409, 0.152797, 0.130257, 0.102677, 0.077068, 0.060066, 0.053721, 0.056799, 0.065154, 0.068981, 0.059617, 0.043592, 0.032413, 0.029936, 0.032977, 0.033875, 0.028997, 0.024594, 0.022125, 0.016720, 0.013682, 0.020969, 0.023380, 0.010585, 0.007082, 0.005682, 0.004304, 0.004051, 0.002125, 0.001418, 0.000863, 0.000490, 0.000400, 0.000743, 0.000338, 0.000282, 0.000223, 0.000213, 0.000167, 0.000203, 0.000151, 0.000165],
                  [0.670719, 0.676117, 0.689948, 0.713760, 0.742241, 0.781210, 0.830391, 0.882027, 0.935181, 0.979704, 1.000000, 0.985546, 0.926684, 0.830646, 0.705098, 0.569544, 0.444362, 0.342731, 0.269568, 0.221739, 0.193749, 0.178210, 0.167409, 0.152797, 0.130257, 0.102677, 0.077068, 0.060066, 0.053721, 0.056799, 0.065154, 0.068981, 0.059617, 0.043592, 0.032413, 0.029936, 0.032977, 0.033875, 0.028997, 0.024594, 0.022125, 0.016720, 0.013682, 0.020969, 0.023380, 0.010585, 0.007082, 0.005682, 0.004304, 0.004051, 0.002125, 0.001418, 0.000863, 0.000490, 0.000400, 0.000743, 0.000338, 0.000282, 0.000223, 0.000213, 0.000167, 0.000203, 0.000151, 0.000165],
                  [0.670719, 0.676117, 0.689948, 0.713760, 0.742241, 0.781210, 0.830391, 0.882027, 0.935181, 0.979704, 1.000000, 0.985546, 0.926684, 0.830646, 0.705098, 0.569544, 0.444362, 0.342731, 0.269568, 0.221739, 0.193749, 0.178210, 0.167409, 0.152797, 0.130257, 0.102677, 0.077068, 0.060066, 0.053721, 0.056799, 0.065154, 0.068981, 0.059617, 0.043592, 0.032413, 0.029936, 0.032977, 0.033875, 0.028997, 0.024594, 0.022125, 0.016720, 0.013682, 0.020969, 0.023380, 0.010585, 0.007082, 0.005682, 0.004304, 0.004051, 0.002125, 0.001418, 0.000863, 0.000490, 0.000400, 0.000743, 0.000338, 0.000282, 0.000223, 0.000213, 0.000167, 0.000203, 0.000151, 0.000165],
                  [0.670719, 0.676117, 0.689948, 0.713760, 0.742241, 0.781210, 0.830391, 0.882027, 0.935181, 0.979704, 1.000000, 0.985546, 0.926684, 0.830646, 0.705098, 0.569544, 0.444362, 0.342731, 0.269568, 0.221739, 0.193749, 0.178210, 0.167409, 0.152797, 0.130257, 0.102677, 0.077068, 0.060066, 0.053721, 0.056799, 0.065154, 0.068981, 0.059617, 0.043592, 0.032413, 0.029936, 0.032977, 0.033875, 0.028997, 0.024594, 0.022125, 0.016720, 0.013682, 0.020969, 0.023380, 0.010585, 0.007082, 0.005682, 0.004304, 0.004051, 0.002125, 0.001418, 0.000863, 0.000490, 0.000400, 0.000743, 0.000338, 0.000282, 0.000223, 0.000213, 0.000167, 0.000203, 0.000151, 0.000165],
                  [0.670719, 0.676117, 0.689948, 0.713760, 0.742241, 0.781210, 0.830391, 0.882027, 0.935181, 0.979704, 1.000000, 0.985546, 0.926684, 0.830646, 0.705098, 0.569544, 0.444362, 0.342731, 0.269568, 0.221739, 0.193749, 0.178210, 0.167409, 0.152797, 0.130257, 0.102677, 0.077068, 0.060066, 0.053721, 0.056799, 0.065154, 0.068981, 0.059617, 0.043592, 0.032413, 0.029936, 0.032977, 0.033875, 0.028997, 0.024594, 0.022125, 0.016720, 0.013682, 0.020969, 0.023380, 0.010585, 0.007082, 0.005682, 0.004304, 0.004051, 0.002125, 0.001418, 0.000863, 0.000490, 0.000400, 0.000743, 0.000338, 0.000282, 0.000223, 0.000213, 0.000167, 0.000203, 0.000151, 0.000165],
                  [0.968940, 0.969455, 0.970763, 0.972989, 0.975611, 0.979141, 0.983536, 0.988136, 0.992973, 0.997352, 1.000000, 0.999871, 0.995133, 0.984359, 0.964852, 0.934302, 0.890369, 0.832116, 0.759773, 0.675925, 0.584858, 0.492254, 0.404065, 0.322144, 0.253653, 0.199651, 0.157802, 0.127243, 0.106172, 0.092713, 0.084206, 0.078470, 0.073828, 0.068485, 0.061597, 0.053728, 0.046525, 0.041191, 0.038063, 0.036282, 0.033903, 0.029487, 0.023937, 0.019625, 0.017990, 0.018368, 0.017725, 0.013881, 0.009380, 0.006798, 0.005629, 0.004889, 0.004971, 0.005544, 0.003631, 0.001398, 0.000940, 0.000856, 0.000751, 0.001011, 0.000562, 0.000375, 0.000356, 0.000407],
                  [0.968940, 0.969455, 0.970763, 0.972989, 0.975611, 0.979141, 0.983536, 0.988136, 0.992973, 0.997352, 1.000000, 0.999871, 0.995133, 0.984359, 0.964852, 0.934302, 0.890369, 0.832116, 0.759773, 0.675925, 0.584858, 0.492254, 0.404065, 0.322144, 0.253653, 0.199651, 0.157802, 0.127243, 0.106172, 0.092713, 0.084206, 0.078470, 0.073828, 0.068485, 0.061597, 0.053728, 0.046525, 0.041191, 0.038063, 0.036282, 0.033903, 0.029487, 0.023937, 0.019625, 0.017990, 0.018368, 0.017725, 0.013881, 0.009380, 0.006798, 0.005629, 0.004889, 0.004971, 0.005544, 0.003631, 0.001398, 0.000940, 0.000856, 0.000751, 0.001011, 0.000562, 0.000375, 0.000356, 0.000407],
                  [0.968940, 0.969455, 0.970763, 0.972989, 0.975611, 0.979141, 0.983536, 0.988136, 0.992973, 0.997352, 1.000000, 0.999871, 0.995133, 0.984359, 0.964852, 0.934302, 0.890369, 0.832116, 0.759773, 0.675925, 0.584858, 0.492254, 0.404065, 0.322144, 0.253653, 0.199651, 0.157802, 0.127243, 0.106172, 0.092713, 0.084206, 0.078470, 0.073828, 0.068485, 0.061597, 0.053728, 0.046525, 0.041191, 0.038063, 0.036282, 0.033903, 0.029487, 0.023937, 0.019625, 0.017990, 0.018368, 0.017725, 0.013881, 0.009380, 0.006798, 0.005629, 0.004889, 0.004971, 0.005544, 0.003631, 0.001398, 0.000940, 0.000856, 0.000751, 0.001011, 0.000562, 0.000375, 0.000356, 0.000407],
                  [0.968940, 0.969455, 0.970763, 0.972989, 0.975611, 0.979141, 0.983536, 0.988136, 0.992973, 0.997352, 1.000000, 0.999871, 0.995133, 0.984359, 0.964852, 0.934302, 0.890369, 0.832116, 0.759773, 0.675925, 0.584858, 0.492254, 0.404065, 0.322144, 0.253653, 0.199651, 0.157802, 0.127243, 0.106172, 0.092713, 0.084206, 0.078470, 0.073828, 0.068485, 0.061597, 0.053728, 0.046525, 0.041191, 0.038063, 0.036282, 0.033903, 0.029487, 0.023937, 0.019625, 0.017990, 0.018368, 0.017725, 0.013881, 0.009380, 0.006798, 0.005629, 0.004889, 0.004971, 0.005544, 0.003631, 0.001398, 0.000940, 0.000856, 0.000751, 0.001011, 0.000562, 0.000375, 0.000356, 0.000407],
                  [0.968940, 0.969455, 0.970763, 0.972989, 0.975611, 0.979141, 0.983536, 0.988136, 0.992973, 0.997352, 1.000000, 0.999871, 0.995133, 0.984359, 0.964852, 0.934302, 0.890369, 0.832116, 0.759773, 0.675925, 0.584858, 0.492254, 0.404065, 0.322144, 0.253653, 0.199651, 0.157802, 0.127243, 0.106172, 0.092713, 0.084206, 0.078470, 0.073828, 0.068485, 0.061597, 0.053728, 0.046525, 0.041191, 0.038063, 0.036282, 0.033903, 0.029487, 0.023937, 0.019625, 0.017990, 0.018368, 0.017725, 0.013881, 0.009380, 0.006798, 0.005629, 0.004889, 0.004971, 0.005544, 0.003631, 0.001398, 0.000940, 0.000856, 0.000751, 0.001011, 0.000562, 0.000375, 0.000356, 0.000407],
                  [0.968940, 0.969455, 0.970763, 0.972989, 0.975611, 0.979141, 0.983536, 0.988136, 0.992973, 0.997352, 1.000000, 0.999871, 0.995133, 0.984359, 0.964852, 0.934302, 0.890369, 0.832116, 0.759773, 0.675925, 0.584858, 0.492254, 0.404065, 0.322144, 0.253653, 0.199651, 0.157802, 0.127243, 0.106172, 0.092713, 0.084206, 0.078470, 0.073828, 0.068485, 0.061597, 0.053728, 0.046525, 0.041191, 0.038063, 0.036282, 0.033903, 0.029487, 0.023937, 0.019625, 0.017990, 0.018368, 0.017725, 0.013881, 0.009380, 0.006798, 0.005629, 0.004889, 0.004971, 0.005544, 0.003631, 0.001398, 0.000940, 0.000856, 0.000751, 0.001011, 0.000562, 0.000375, 0.000356, 0.000407],
                  [0.968940, 0.969455, 0.970763, 0.972989, 0.975611, 0.979141, 0.983536, 0.988136, 0.992973, 0.997352, 1.000000, 0.999871, 0.995133, 0.984359, 0.964852, 0.934302, 0.890369, 0.832116, 0.759773, 0.675925, 0.584858, 0.492254, 0.404065, 0.322144, 0.253653, 0.199651, 0.157802, 0.127243, 0.106172, 0.092713, 0.084206, 0.078470, 0.073828, 0.068485, 0.061597, 0.053728, 0.046525, 0.041191, 0.038063, 0.036282, 0.033903, 0.029487, 0.023937, 0.019625, 0.017990, 0.018368, 0.017725, 0.013881, 0.009380, 0.006798, 0.005629, 0.004889, 0.004971, 0.005544, 0.003631, 0.001398, 0.000940, 0.000856, 0.000751, 0.001011, 0.000562, 0.000375, 0.000356, 0.000407],
                  [0.968940, 0.969455, 0.970763, 0.972989, 0.975611, 0.979141, 0.983536, 0.988136, 0.992973, 0.997352, 1.000000, 0.999871, 0.995133, 0.984359, 0.964852, 0.934302, 0.890369, 0.832116, 0.759773, 0.675925, 0.584858, 0.492254, 0.404065, 0.322144, 0.253653, 0.199651, 0.157802, 0.127243, 0.106172, 0.092713, 0.084206, 0.078470, 0.073828, 0.068485, 0.061597, 0.053728, 0.046525, 0.041191, 0.038063, 0.036282, 0.033903, 0.029487, 0.023937, 0.019625, 0.017990, 0.018368, 0.017725, 0.013881, 0.009380, 0.006798, 0.005629, 0.004889, 0.004971, 0.005544, 0.003631, 0.001398, 0.000940, 0.000856, 0.000751, 0.001011, 0.000562, 0.000375, 0.000356, 0.000407],
                  [0.968940, 0.969455, 0.970763, 0.972989, 0.975611, 0.979141, 0.983536, 0.988136, 0.992973, 0.997352, 1.000000, 0.999871, 0.995133, 0.984359, 0.964852, 0.934302, 0.890369, 0.832116, 0.759773, 0.675925, 0.584858, 0.492254, 0.404065, 0.322144, 0.253653, 0.199651, 0.157802, 0.127243, 0.106172, 0.092713, 0.084206, 0.078470, 0.073828, 0.068485, 0.061597, 0.053728, 0.046525, 0.041191, 0.038063, 0.036282, 0.033903, 0.029487, 0.023937, 0.019625, 0.017990, 0.018368, 0.017725, 0.013881, 0.009380, 0.006798, 0.005629, 0.004889, 0.004971, 0.005544, 0.003631, 0.001398, 0.000940, 0.000856, 0.000751, 0.001011, 0.000562, 0.000375, 0.000356, 0.000407],
                  [0.968940, 0.969455, 0.970763, 0.972989, 0.975611, 0.979141, 0.983536, 0.988136, 0.992973, 0.997352, 1.000000, 0.999871, 0.995133, 0.984359, 0.964852, 0.934302, 0.890369, 0.832116, 0.759773, 0.675925, 0.584858, 0.492254, 0.404065, 0.322144, 0.253653, 0.199651, 0.157802, 0.127243, 0.106172, 0.092713, 0.084206, 0.078470, 0.073828, 0.068485, 0.061597, 0.053728, 0.046525, 0.041191, 0.038063, 0.036282, 0.033903, 0.029487, 0.023937, 0.019625, 0.017990, 0.018368, 0.017725, 0.013881, 0.009380, 0.006798, 0.005629, 0.004889, 0.004971, 0.005544, 0.003631, 0.001398, 0.000940, 0.000856, 0.000751, 0.001011, 0.000562, 0.000375, 0.000356, 0.000407],
                  [0.968940, 0.969455, 0.970763, 0.972989, 0.975611, 0.979141, 0.983536, 0.988136, 0.992973, 0.997352, 1.000000, 0.999871, 0.995133, 0.984359, 0.964852, 0.934302, 0.890369, 0.832116, 0.759773, 0.675925, 0.584858, 0.492254, 0.404065, 0.322144, 0.253653, 0.199651, 0.157802, 0.127243, 0.106172, 0.092713, 0.084206, 0.078470, 0.073828, 0.068485, 0.061597, 0.053728, 0.046525, 0.041191, 0.038063, 0.036282, 0.033903, 0.029487, 0.023937, 0.019625, 0.017990, 0.018368, 0.017725, 0.013881, 0.009380, 0.006798, 0.005629, 0.004889, 0.004971, 0.005544, 0.003631, 0.001398, 0.000940, 0.000856, 0.000751, 0.001011, 0.000562, 0.000375, 0.000356, 0.000407],
                  [0.968940, 0.969455, 0.970763, 0.972989, 0.975611, 0.979141, 0.983536, 0.988136, 0.992973, 0.997352, 1.000000, 0.999871, 0.995133, 0.984359, 0.964852, 0.934302, 0.890369, 0.832116, 0.759773, 0.675925, 0.584858, 0.492254, 0.404065, 0.322144, 0.253653, 0.199651, 0.157802, 0.127243, 0.106172, 0.092713, 0.084206, 0.078470, 0.073828, 0.068485, 0.061597, 0.053728, 0.046525, 0.041191, 0.038063, 0.036282, 0.033903, 0.029487, 0.023937, 0.019625, 0.017990, 0.018368, 0.017725, 0.013881, 0.009380, 0.006798, 0.005629, 0.004889, 0.004971, 0.005544, 0.003631, 0.001398, 0.000940, 0.000856, 0.000751, 0.001011, 0.000562, 0.000375, 0.000356, 0.000407],
                  [1.000000, 0.999270, 0.997385, 0.994090, 0.990046, 0.984261, 0.976391, 0.967022, 0.955028, 0.940087, 0.923303, 0.903179, 0.879521, 0.854022, 0.824941, 0.792320, 0.756250, 0.717009, 0.674970, 0.630655, 0.584693, 0.537784, 0.490713, 0.442370, 0.395624, 0.351230, 0.308341, 0.267892, 0.230639, 0.197960, 0.169034, 0.143279, 0.121915, 0.104015, 0.088881, 0.076332, 0.066204, 0.057908, 0.051035, 0.045234, 0.040041, 0.035230, 0.030539, 0.025782, 0.020947, 0.016311, 0.012161, 0.008704, 0.006057, 0.004168, 0.002902, 0.002097, 0.001585, 0.001260, 0.001049, 0.000904, 0.000803, 0.000726, 0.000657, 0.000584, 0.000505, 0.000433, 0.000384, 0.000361],
                  [1.000000, 0.999270, 0.997385, 0.994090, 0.990046, 0.984261, 0.976391, 0.967022, 0.955028, 0.940087, 0.923303, 0.903179, 0.879521, 0.854022, 0.824941, 0.792320, 0.756250, 0.717009, 0.674970, 0.630655, 0.584693, 0.537784, 0.490713, 0.442370, 0.395624, 0.351230, 0.308341, 0.267892, 0.230639, 0.197960, 0.169034, 0.143279, 0.121915, 0.104015, 0.088881, 0.076332, 0.066204, 0.057908, 0.051035, 0.045234, 0.040041, 0.035230, 0.030539, 0.025782, 0.020947, 0.016311, 0.012161, 0.008704, 0.006057, 0.004168, 0.002902, 0.002097, 0.001585, 0.001260, 0.001049, 0.000904, 0.000803, 0.000726, 0.000657, 0.000584, 0.000505, 0.000433, 0.000384, 0.000361],
                  [1.000000, 0.999270, 0.997385, 0.994090, 0.990046, 0.984261, 0.976391, 0.967022, 0.955028, 0.940087, 0.923303, 0.903179, 0.879521, 0.854022, 0.824941, 0.792320, 0.756250, 0.717009, 0.674970, 0.630655, 0.584693, 0.537784, 0.490713, 0.442370, 0.395624, 0.351230, 0.308341, 0.267892, 0.230639, 0.197960, 0.169034, 0.143279, 0.121915, 0.104015, 0.088881, 0.076332, 0.066204, 0.057908, 0.051035, 0.045234, 0.040041, 0.035230, 0.030539, 0.025782, 0.020947, 0.016311, 0.012161, 0.008704, 0.006057, 0.004168, 0.002902, 0.002097, 0.001585, 0.001260, 0.001049, 0.000904, 0.000803, 0.000726, 0.000657, 0.000584, 0.000505, 0.000433, 0.000384, 0.000361],
                  [1.000000, 0.999270, 0.997385, 0.994090, 0.990046, 0.984261, 0.976391, 0.967022, 0.955028, 0.940087, 0.923303, 0.903179, 0.879521, 0.854022, 0.824941, 0.792320, 0.756250, 0.717009, 0.674970, 0.630655, 0.584693, 0.537784, 0.490713, 0.442370, 0.395624, 0.351230, 0.308341, 0.267892, 0.230639, 0.197960, 0.169034, 0.143279, 0.121915, 0.104015, 0.088881, 0.076332, 0.066204, 0.057908, 0.051035, 0.045234, 0.040041, 0.035230, 0.030539, 0.025782, 0.020947, 0.016311, 0.012161, 0.008704, 0.006057, 0.004168, 0.002902, 0.002097, 0.001585, 0.001260, 0.001049, 0.000904, 0.000803, 0.000726, 0.000657, 0.000584, 0.000505, 0.000433, 0.000384, 0.000361],
                  [1.000000, 0.999270, 0.997385, 0.994090, 0.990046, 0.984261, 0.976391, 0.967022, 0.955028, 0.940087, 0.923303, 0.903179, 0.879521, 0.854022, 0.824941, 0.792320, 0.756250, 0.717009, 0.674970, 0.630655, 0.584693, 0.537784, 0.490713, 0.442370, 0.395624, 0.351230, 0.308341, 0.267892, 0.230639, 0.197960, 0.169034, 0.143279, 0.121915, 0.104015, 0.088881, 0.076332, 0.066204, 0.057908, 0.051035, 0.045234, 0.040041, 0.035230, 0.030539, 0.025782, 0.020947, 0.016311, 0.012161, 0.008704, 0.006057, 0.004168, 0.002902, 0.002097, 0.001585, 0.001260, 0.001049, 0.000904, 0.000803, 0.000726, 0.000657, 0.000584, 0.000505, 0.000433, 0.000384, 0.000361],
                  [1.000000, 0.999270, 0.997385, 0.994090, 0.990046, 0.984261, 0.976391, 0.967022, 0.955028, 0.940087, 0.923303, 0.903179, 0.879521, 0.854022, 0.824941, 0.792320, 0.756250, 0.717009, 0.674970, 0.630655, 0.584693, 0.537784, 0.490713, 0.442370, 0.395624, 0.351230, 0.308341, 0.267892, 0.230639, 0.197960, 0.169034, 0.143279, 0.121915, 0.104015, 0.088881, 0.076332, 0.066204, 0.057908, 0.051035, 0.045234, 0.040041, 0.035230, 0.030539, 0.025782, 0.020947, 0.016311, 0.012161, 0.008704, 0.006057, 0.004168, 0.002902, 0.002097, 0.001585, 0.001260, 0.001049, 0.000904, 0.000803, 0.000726, 0.000657, 0.000584, 0.000505, 0.000433, 0.000384, 0.000361],
                  [1.000000, 0.999270, 0.997385, 0.994090, 0.990046, 0.984261, 0.976391, 0.967022, 0.955028, 0.940087, 0.923303, 0.903179, 0.879521, 0.854022, 0.824941, 0.792320, 0.756250, 0.717009, 0.674970, 0.630655, 0.584693, 0.537784, 0.490713, 0.442370, 0.395624, 0.351230, 0.308341, 0.267892, 0.230639, 0.197960, 0.169034, 0.143279, 0.121915, 0.104015, 0.088881, 0.076332, 0.066204, 0.057908, 0.051035, 0.045234, 0.040041, 0.035230, 0.030539, 0.025782, 0.020947, 0.016311, 0.012161, 0.008704, 0.006057, 0.004168, 0.002902, 0.002097, 0.001585, 0.001260, 0.001049, 0.000904, 0.000803, 0.000726, 0.000657, 0.000584, 0.000505, 0.000433, 0.000384, 0.000361],
                  [1.000000, 0.999270, 0.997385, 0.994090, 0.990046, 0.984261, 0.976391, 0.967022, 0.955028, 0.940087, 0.923303, 0.903179, 0.879521, 0.854022, 0.824941, 0.792320, 0.756250, 0.717009, 0.674970, 0.630655, 0.584693, 0.537784, 0.490713, 0.442370, 0.395624, 0.351230, 0.308341, 0.267892, 0.230639, 0.197960, 0.169034, 0.143279, 0.121915, 0.104015, 0.088881, 0.076332, 0.066204, 0.057908, 0.051035, 0.045234, 0.040041, 0.035230, 0.030539, 0.025782, 0.020947, 0.016311, 0.012161, 0.008704, 0.006057, 0.004168, 0.002902, 0.002097, 0.001585, 0.001260, 0.001049, 0.000904, 0.000803, 0.000726, 0.000657, 0.000584, 0.000505, 0.000433, 0.000384, 0.000361],
                  [1.000000, 0.999270, 0.997385, 0.994090, 0.990046, 0.984261, 0.976391, 0.967022, 0.955028, 0.940087, 0.923303, 0.903179, 0.879521, 0.854022, 0.824941, 0.792320, 0.756250, 0.717009, 0.674970, 0.630655, 0.584693, 0.537784, 0.490713, 0.442370, 0.395624, 0.351230, 0.308341, 0.267892, 0.230639, 0.197960, 0.169034, 0.143279, 0.121915, 0.104015, 0.088881, 0.076332, 0.066204, 0.057908, 0.051035, 0.045234, 0.040041, 0.035230, 0.030539, 0.025782, 0.020947, 0.016311, 0.012161, 0.008704, 0.006057, 0.004168, 0.002902, 0.002097, 0.001585, 0.001260, 0.001049, 0.000904, 0.000803, 0.000726, 0.000657, 0.000584, 0.000505, 0.000433, 0.000384, 0.000361],
                  [1.000000, 0.999270, 0.997385, 0.994090, 0.990046, 0.984261, 0.976391, 0.967022, 0.955028, 0.940087, 0.923303, 0.903179, 0.879521, 0.854022, 0.824941, 0.792320, 0.756250, 0.717009, 0.674970, 0.630655, 0.584693, 0.537784, 0.490713, 0.442370, 0.395624, 0.351230, 0.308341, 0.267892, 0.230639, 0.197960, 0.169034, 0.143279, 0.121915, 0.104015, 0.088881, 0.076332, 0.066204, 0.057908, 0.051035, 0.045234, 0.040041, 0.035230, 0.030539, 0.025782, 0.020947, 0.016311, 0.012161, 0.008704, 0.006057, 0.004168, 0.002902, 0.002097, 0.001585, 0.001260, 0.001049, 0.000904, 0.000803, 0.000726, 0.000657, 0.000584, 0.000505, 0.000433, 0.000384, 0.000361],
                  [1.000000, 0.999270, 0.997385, 0.994090, 0.990046, 0.984261, 0.976391, 0.967022, 0.955028, 0.940087, 0.923303, 0.903179, 0.879521, 0.854022, 0.824941, 0.792320, 0.756250, 0.717009, 0.674970, 0.630655, 0.584693, 0.537784, 0.490713, 0.442370, 0.395624, 0.351230, 0.308341, 0.267892, 0.230639, 0.197960, 0.169034, 0.143279, 0.121915, 0.104015, 0.088881, 0.076332, 0.066204, 0.057908, 0.051035, 0.045234, 0.040041, 0.035230, 0.030539, 0.025782, 0.020947, 0.016311, 0.012161, 0.008704, 0.006057, 0.004168, 0.002902, 0.002097, 0.001585, 0.001260, 0.001049, 0.000904, 0.000803, 0.000726, 0.000657, 0.000584, 0.000505, 0.000433, 0.000384, 0.000361],
                  [1.000000, 0.999270, 0.997385, 0.994090, 0.990046, 0.984261, 0.976391, 0.967022, 0.955028, 0.940087, 0.923303, 0.903179, 0.879521, 0.854022, 0.824941, 0.792320, 0.756250, 0.717009, 0.674970, 0.630655, 0.584693, 0.537784, 0.490713, 0.442370, 0.395624, 0.351230, 0.308341, 0.267892, 0.230639, 0.197960, 0.169034, 0.143279, 0.121915, 0.104015, 0.088881, 0.076332, 0.066204, 0.057908, 0.051035, 0.045234, 0.040041, 0.035230, 0.030539, 0.025782, 0.020947, 0.016311, 0.012161, 0.008704, 0.006057, 0.004168, 0.002902, 0.002097, 0.001585, 0.001260, 0.001049, 0.000904, 0.000803, 0.000726, 0.000657, 0.000584, 0.000505, 0.000433, 0.000384, 0.000361],
                  [1.000000, 0.999852, 0.999471, 0.998803, 0.997979, 0.996795, 0.995173, 0.993224, 0.990701, 0.987512, 0.983867, 0.979405, 0.974028, 0.968064, 0.961033, 0.952838, 0.943358, 0.932505, 0.920167, 0.906277, 0.890747, 0.873537, 0.854600, 0.833054, 0.809731, 0.784679, 0.757030, 0.726855, 0.694299, 0.660576, 0.624980, 0.586885, 0.548652, 0.509766, 0.469829, 0.429500, 0.390179, 0.351695, 0.314585, 0.279443, 0.246141, 0.215645, 0.188098, 0.163259, 0.140915, 0.121379, 0.104480, 0.089828, 0.077196, 0.066230, 0.056675, 0.048349, 0.040858, 0.034082, 0.027860, 0.022132, 0.016989, 0.012536, 0.008930, 0.006193, 0.004261, 0.003007, 0.002271, 0.001929]])

    N = 4096
    number_of_band = 64

    H = np.zeros(N)
    for band in range(number_of_band):
        fL = f[band]
        fH = f[band + 1]
        for k in range(fL, fH):
            H[k] = a[note_number - 41][band]

    for k in range(1, int(N / 2)):
        H[N - k] = H[k]

    h = np.real(np.fft.ifft(H, N))

    for m in range(int(N / 2)):
        tmp = h[m]
        h[m] = h[int(N / 2) + m]
        h[int(N / 2) + m] = tmp

    J = 128

    b = np.zeros(J + 1)
    w = Hanning_window(J + 1)
    offset = int(N / 2) - int(J / 2)
    for m in range(J + 1):
        b[m] = h[offset + m] * w[m]

    # 1st string
    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    # decay
    p1 = 80
    p2 = 0
    p3 = 120 / 12
    p4 = 20
    decay = (1 - 1 / (1 + np.exp(-(note_number - p1) / p3))) * p4 + p2

    # d
    p1 = 55
    p2 = 0
    p3 = 120 / 12
    p4 = 0.5
    d = (1 - 1 / (1 + np.exp(-(note_number - p1) / p3))) * p4 + p2

    num = np.power(10, -3 * T / decay)
    den = np.sqrt((1 - d) * (1 - d) + 2 * d * (1 - d) * np.cos((2 * np.pi * f0) / fs) + d * d)

    c = num / den
    if c > 1:
        c = 1

    D = int(T * fs - d)
    e = T * fs - d - int(T * fs - d)
    g = (1 - e) / (1 + e)

    s0 = np.zeros(length_of_s)
    s1 = np.zeros(length_of_s)
    s2 = np.zeros(length_of_s)

    np.random.seed(0)
    number_of_partial = int(20000 / f0)
    for i in range(number_of_partial):
        theta = (np.random.rand() * 2 - 1) * np.pi
        for n in range(D + 1 + J):
            s0[n] += np.sin(2 * np.pi * f0 * (i + 1) * n / fs + theta)

    mean_of_s0 = 0
    for n in range(J, D + 1 + J):
        mean_of_s0 += s0[n]

    mean_of_s0 /= D + 1
    for n in range(D + 1 + J):
        s0[n] -= mean_of_s0

    for n in range(D + 1 + J, length_of_s):
        # fractional delay
        s1[n] = -g * s1[n - 1] + g * s0[n - D] + s0[n - D - 1]

        # filter
        s2[n] = c * ((1 - d) * s1[n] + d * s1[n - 1])

        # feedback
        s0[n] += s2[n]

    u0 = s0

    # 2nd string
    f0 = 440 * np.power(2, (note_number - 69 + 0.01) / 12)
    T = 1 / f0

    # decay
    p1 = 80
    p2 = 0
    p3 = 120 / 12
    p4 = 20
    decay = (1 - 1 / (1 + np.exp(-(note_number - p1) / p3))) * p4 + p2

    # d
    p1 = 55
    p2 = 0
    p3 = 120 / 12
    p4 = 0.5
    d = (1 - 1 / (1 + np.exp(-(note_number - p1) / p3))) * p4 + p2

    num = np.power(10, -3 * T / decay)
    den = np.sqrt((1 - d) * (1 - d) + 2 * d * (1 - d) * np.cos((2 * np.pi * f0) / fs) + d * d)

    c = num / den
    if c > 1:
        c = 1

    D = int(T * fs - d)
    e = T * fs - d - int(T * fs - d)
    g = (1 - e) / (1 + e)

    s0 = np.zeros(length_of_s)
    s1 = np.zeros(length_of_s)
    s2 = np.zeros(length_of_s)

    np.random.seed(0)
    number_of_partial = int(20000 / f0)
    for i in range(number_of_partial):
        theta = (np.random.rand() * 2 - 1) * np.pi
        for n in range(D + 1 + J):
            s0[n] += np.sin(2 * np.pi * f0 * (i + 1) * n / fs + theta)

    mean_of_s0 = 0
    for n in range(J, D + 1 + J):
        mean_of_s0 += s0[n]

    mean_of_s0 /= D + 1
    for n in range(D + 1 + J):
        s0[n] -= mean_of_s0

    for n in range(D + 1 + J, length_of_s):
        # fractional delay
        s1[n] = -g * s1[n - 1] + g * s0[n - D] + s0[n - D - 1]

        # filter
        s2[n] = c * ((1 - d) * s1[n] + d * s1[n - 1])

        # feedback
        s0[n] += s2[n]

    u1 = s0

    # 3rd string
    f0 = 440 * np.power(2, (note_number - 69 - 0.01) / 12)
    T = 1 / f0

    # decay
    p1 = 80
    p2 = 0
    p3 = 120 / 12
    p4 = 20
    decay = (1 - 1 / (1 + np.exp(-(note_number - p1) / p3))) * p4 + p2

    # d
    p1 = 55
    p2 = 0
    p3 = 120 / 12
    p4 = 0.5
    d = (1 - 1 / (1 + np.exp(-(note_number - p1) / p3))) * p4 + p2

    num = np.power(10, -3 * T / decay)
    den = np.sqrt((1 - d) * (1 - d) + 2 * d * (1 - d) * np.cos((2 * np.pi * f0) / fs) + d * d)

    c = num / den
    if c > 1:
        c = 1

    D = int(T * fs - d)
    e = T * fs - d - int(T * fs - d)
    g = (1 - e) / (1 + e)

    s0 = np.zeros(length_of_s)
    s1 = np.zeros(length_of_s)
    s2 = np.zeros(length_of_s)

    np.random.seed(0)
    number_of_partial = int(20000 / f0)
    for i in range(number_of_partial):
        theta = (np.random.rand() * 2 - 1) * np.pi
        for n in range(D + 1 + J):
            s0[n] += np.sin(2 * np.pi * f0 * (i + 1) * n / fs + theta)

    mean_of_s0 = 0
    for n in range(J, D + 1 + J):
        mean_of_s0 += s0[n]

    mean_of_s0 /= D + 1
    for n in range(D + 1 + J):
        s0[n] -= mean_of_s0

    for n in range(D + 1 + J, length_of_s):
        # fractional delay
        s1[n] = -g * s1[n - 1] + g * s0[n - D] + s0[n - D - 1]

        # filter
        s2[n] = c * ((1 - d) * s1[n] + d * s1[n - 1])

        # feedback
        s0[n] += s2[n]

    u2 = s0

    # mix
    if note_number <= 32:
        s3 = u0
    elif note_number <= 52:
        s3 = u0 + u1
    elif note_number <= 108:
        s3 = u0 + u1 + u2

    # comb filter
    excitation = np.array([0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.08, 0.08, 0.08, 0.08])
    p = excitation[note_number - 21]
    s4 = np.zeros(length_of_s)
    for n in range(length_of_s):
        t = n - T * fs * p
        m = int(t)
        delta = t - m
        if m < 0:
            s4[n] = s3[n] - (delta * s3[m + 1 + D + 1] + (1 - delta) * s3[m + D + 1])
        else:
            s4[n] = s3[n] - (delta * s3[m + 1] + (1 - delta) * s3[m])

    s5 = np.zeros(length_of_s)
    for n in range(length_of_s):
        for m in range(J + 1):
            if n - m >= 0:
                s5[n] += b[m] * s4[n - m]

    s6 = np.zeros(length_of_s)
    for n in range(length_of_s - J):
        s6[n] = s5[J + n]

    # DC cancel
    fc = 5
    Q = 1 / np.sqrt(2)
    a, b = HPF(fs, fc, Q)
    s7 = filter(a, b, s6)
    s7 /= np.max(np.abs(s7))

    # A part
    VCF_A = np.array([0])
    VCF_D = np.array([T * 5])
    VCF_S = np.array([0])
    VCF_R = np.array([T * 5])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([f0 * 32])
    VCF_depth = np.array([f0 * 512])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]
        if vcf[n] > 20000:
            vcf[n] = 20000

    sa = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                sa[n] += b[m] * s7[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                sa[n] += -a[m] * sa[n - m]

    VCA_A = np.array([T * 1])
    VCA_D = np.array([T * 20])
    VCA_S = np.array([0])
    VCA_R = np.array([T * 20])
    VCA_gate = np.array([gate])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        sa[n] *= vca[n]

    # B part
    VCF_A = np.array([0])
    VCF_D = np.array([decay * 0.2])
    VCF_S = np.array([0])
    VCF_R = np.array([decay * 0.2])
    VCF_gate = np.array([gate])
    VCF_duration = np.array([duration])
    VCF_offset = np.array([f0 * 8])
    VCF_depth = np.array([f0 * 128])

    vcf = ADSR(fs, VCF_A[0], VCF_D[0], VCF_S[0], VCF_R[0], VCF_gate[0], VCF_duration[0])
    for n in range(length_of_s):
        vcf[n] = VCF_offset[0] + vcf[n] * VCF_depth[0]
        if vcf[n] > 20000:
            vcf[n] = 20000

    sb = np.zeros(length_of_s)
    Q = 1 / np.sqrt(2)
    for n in range(length_of_s):
        a, b = LPF(fs, vcf[n], Q)
        for m in range(0, 3):
            if n - m >= 0:
                sb[n] += b[m] * s7[n - m]

        for m in range(1, 3):
            if n - m >= 0:
                sb[n] += -a[m] * sb[n - m]

    VCA_A = np.array([T * 1])
    VCA_D = np.array([0])
    VCA_S = np.array([1])
    VCA_R = np.array([0])
    VCA_gate = np.array([duration])
    VCA_duration = np.array([duration])
    VCA_offset = np.array([0])
    VCA_depth = np.array([1])

    vca = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca[n] = VCA_offset[0] + vca[n] * VCA_depth[0]

    for n in range(length_of_s):
        sb[n] *= vca[n]

    s8 = sa * 0.5 + sb * 0.5

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
        s8[n] *= vca[n]

    s8 *= velocity / 127 / np.max(np.abs(s8))

    return s8

def electric_piano(fs, note_number, velocity, gate):
    duration = gate + 1

    length_of_s = int(fs * duration)
    sa = np.zeros(length_of_s)
    sb = np.zeros(length_of_s)
    s = np.zeros(length_of_s)

    f0 = 440 * np.power(2, (note_number - 69) / 12)
    T = 1 / f0

    VCO_A = np.array([0, 0, 0, 0])
    VCO_D = np.array([0, 0, 0, 0])
    VCO_S = np.array([1, 1, 1, 1])
    VCO_R = np.array([0, 0, 0, 0])
    VCO_gate = np.repeat(duration, 4)
    VCO_duration = np.repeat(duration, 4)
    VCO_offset = np.array([14, 1, 1, 1]) * f0
    VCO_depth = np.array([0, 0, 0, 0])

    VCA_A = np.array([0, 0, 0, 0])
    VCA_D = np.array([1, 1, 2, 4])
    VCA_S = np.array([0, 0, 0, 0])
    VCA_R = np.array([1, 0.1, 2, 0.1])
    VCA_gate = np.repeat(gate, 4)
    VCA_duration = np.repeat(duration, 4)
    VCA_offset = np.array([0, 0, 0, 0])
    VCA_depth = np.array([1, 1, 1, 1])

    # A part
    vco_m = ADSR(fs, VCO_A[0], VCO_D[0], VCO_S[0], VCO_R[0], VCO_gate[0], VCO_duration[0])
    for n in range(length_of_s):
        vco_m[n] = VCO_offset[0] + vco_m[n] * VCO_depth[0]

    vca_m = ADSR(fs, VCA_A[0], VCA_D[0], VCA_S[0], VCA_R[0], VCA_gate[0], VCA_duration[0])
    for n in range(length_of_s):
        vca_m[n] = VCA_offset[0] + vca_m[n] * VCA_depth[0]

    vco_c = ADSR(fs, VCO_A[1], VCO_D[1], VCO_S[1], VCO_R[1], VCO_gate[1], VCO_duration[1])
    for n in range(length_of_s):
        vco_c[n] = VCO_offset[1] + vco_c[n] * VCO_depth[1]

    vca_c = ADSR(fs, VCA_A[1], VCA_D[1], VCA_S[1], VCA_R[1], VCA_gate[1], VCA_duration[1])
    for n in range(length_of_s):
        vca_c[n] = VCA_offset[1] + vca_c[n] * VCA_depth[1]

    xm = 0
    xc = 0
    for n in range(length_of_s):
        sa[n] = vca_c[n] * np.sin(2 * np.pi * xc + vca_m[n] * np.sin(2 * np.pi * xm))
        delta_m = vco_m[n] / fs
        xm += delta_m
        if xm >= 1:
            xm -= 1

        delta_c = vco_c[n] / fs
        xc += delta_c
        if xc >= 1:
            xc -= 1

    # B part
    vco_m = ADSR(fs, VCO_A[2], VCO_D[2], VCO_S[2], VCO_R[2], VCO_gate[2], VCO_duration[2])
    for n in range(length_of_s):
        vco_m[n] = VCO_offset[2] + vco_m[n] * VCO_depth[2]

    vca_m = ADSR(fs, VCA_A[2], VCA_D[2], VCA_S[2], VCA_R[2], VCA_gate[2], VCA_duration[2])
    for n in range(length_of_s):
        vca_m[n] = VCA_offset[2] + vca_m[n] * VCA_depth[2]

    vco_c = ADSR(fs, VCO_A[3], VCO_D[3], VCO_S[3], VCO_R[3], VCO_gate[3], VCO_duration[3])
    for n in range(length_of_s):
        vco_c[n] = VCO_offset[3] + vco_c[n] * VCO_depth[3]

    vca_c = ADSR(fs, VCA_A[3], VCA_D[3], VCA_S[3], VCA_R[3], VCA_gate[3], VCA_duration[3])
    for n in range(length_of_s):
        vca_c[n] = VCA_offset[3] + vca_c[n] * VCA_depth[3]

    xm = 0
    xc = 0
    for n in range(length_of_s):
        sb[n] = vca_c[n] * np.sin(2 * np.pi * xc + vca_m[n] * np.sin(2 * np.pi * xm))
        delta_m = vco_m[n] / fs
        xm += delta_m
        if xm >= 1:
            xm -= 1

        delta_c = vco_c[n] / fs
        xc += delta_c
        if xc >= 1:
            xc -= 1

    s = sa * 0.5 + sb * 0.5

    s *= velocity / 127 / np.max(np.abs(s))

    return s
