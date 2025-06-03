import numpy as np

def Hanning_window(N):
    w = np.zeros(N)
    if N % 2 == 0:
        for n in range(N):
            w[n] = 0.5 - 0.5 * np.cos(2 * np.pi * n / N)
    else:
        for n in range(N):
            w[n] = 0.5 - 0.5 * np.cos(2 * np.pi * (n + 0.5) / N)
    return w
