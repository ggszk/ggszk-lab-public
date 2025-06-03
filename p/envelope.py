import numpy as np

def ADSR(fs, A, D, S, R, gate, duration):
    A = int(fs * A)
    D = int(fs * D)
    R = int(fs * R)
    gate = int(fs * gate)
    duration = int(fs * duration)
    e = np.zeros(duration)
    if A != 0:
        for n in range(A):
            e[n] = (1 - np.exp(-5 * n / A)) / (1 - np.exp(-5))

    if D != 0:
        for n in range(A, gate):
            e[n] = 1 + (S - 1) * (1 - np.exp(-5 * (n - A) / D))

    else:
        for n in range(A, gate):
            e[n] = S

    if R != 0:
        for n in range(gate, duration):
            e[n] = e[gate - 1] - e[gate - 1] * (1 - np.exp(-5 * (n - gate + 1) / R))

    return e

def cosine_envelope(fs, delay, A, S, R, gate, duration):
    delay = int(fs * delay)
    A = int(fs * A)
    R = int(fs * R)
    gate = int(fs * gate)
    duration = int(fs * duration)
    e = np.zeros(duration)
    if A != 0:
        for n in range(delay, delay + A):
            e[n] = S * (0.5 - 0.5 * np.cos(np.pi * (n - delay) / A))

    for n in range(delay + A, gate):
            e[n] = S

    if R != 0:
        for n in range(gate, min(gate + R, duration)):
            e[n] = S * (0.5 + 0.5 * np.cos(np.pi * (n - gate) / R))

    return e
