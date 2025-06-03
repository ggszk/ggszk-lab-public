import numpy as np

def sine_wave(fs, vco, duration):
    length_of_s = int(fs * duration)
    s = np.zeros(length_of_s)
    x = 0
    for n in range(length_of_s):
        s[n] = np.sin(2 * np.pi * x)
        delta = vco[n] / fs
        x += delta
        if x >= 1:
            x -= 1

    return s

def sawtooth_wave(fs, vco, duration):
    length_of_s = int(fs * duration)
    s = np.zeros(length_of_s)
    x = 0
    for n in range(length_of_s):
        s[n] = -2 * x + 1
        delta = vco[n] / fs
        if 0 <= x and x < delta:
            t = x / delta
            d = -t * t + 2 * t - 1
            s[n] += d
        elif 1 - delta < x and x <= 1:
            t = (x - 1) / delta
            d = t * t + 2 * t + 1
            s[n] += d

        x += delta
        if x >= 1:
            x -= 1

    return s

def square_wave(fs, vco, duration):
    length_of_s = int(fs * duration)
    s = np.zeros(length_of_s)
    x = 0
    for n in range(length_of_s):
        if x < 0.5:
            s[n] = 1
        else:
            s[n] = -1

        delta = vco[n] / fs
        if 0 <= x and x < delta:
            t = x / delta
            d = -t * t + 2 * t - 1
            s[n] += d
        elif 1 - delta < x and x <= 1:
            t = (x - 1) / delta
            d = t * t + 2 * t + 1
            s[n] += d

        if 0.5 <= x and x < 0.5 + delta:
            t = (x - 0.5) / delta
            d = -t * t + 2 * t - 1
            s[n] -= d
        elif 0.5 - delta < x and x <= 0.5:
            t = (x - 0.5) / delta
            d = t * t + 2 * t + 1
            s[n] -= d

        x += delta
        if x >= 1:
            x -= 1

    return s

def white_noise(fs, duration):
    length_of_s = int(fs * duration)
    s = np.zeros(length_of_s)
    np.random.seed(0)
    for n in range(length_of_s):
        s[n] = (np.random.rand() * 2) - 1

    return s
