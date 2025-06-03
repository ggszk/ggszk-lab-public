import numpy as np
from scipy.io import wavfile

def wave_read_16bit_mono(file_name):
    fs, s = wavfile.read(file_name)
    s = s.astype(np.float)
    length_of_s = len(s)
    np.random.seed(0)
    for n in range(length_of_s):
        s[n] = s[n] + 32768 + (np.random.rand() - 0.5)
        s[n] = s[n] / 65536 * 2 - 1

    return fs, s

def wave_write_16bit_mono(fs, s, file_name):
    length_of_s = len(s)
    for n in range(length_of_s):
        s[n] = (s[n] + 1) / 2 * 65536
        s[n] = int(s[n] + 0.5)
        if s[n] > 65535:
            s[n] = 65535
        elif s[n] < 0:
            s[n] = 0

        s[n] -= 32768

    wavfile.write(file_name, fs, s.astype(np.int16))

def wave_read_16bit_stereo(file_name):
    fs, s = wavfile.read(file_name)
    s = s.astype(np.float)
    length_of_s = len(s)
    np.random.seed(0)
    for n in range(length_of_s):
        s[n, 0] = s[n, 0] + 32768 + (np.random.rand() - 0.5)
        s[n, 0] = s[n, 0] / 65536 * 2 - 1

    for n in range(length_of_s):
        s[n, 1] = s[n, 1] + 32768 + (np.random.rand() - 0.5)
        s[n, 1] = s[n, 1] / 65536 * 2 - 1

    return fs, s

def wave_write_16bit_stereo(fs, s, file_name):
    length_of_s = len(s)
    for n in range(length_of_s):
        s[n, 0] = (s[n, 0] + 1) / 2 * 65536
        s[n, 0] = int(s[n, 0] + 0.5)
        if s[n, 0] > 65535:
            s[n, 0] = 65535
        elif s[n, 0] < 0:
            s[n, 0] = 0

        s[n, 0] -= 32768

        s[n, 1] = (s[n, 1] + 1) / 2 * 65536
        s[n, 1] = int(s[n, 1] + 0.5)
        if s[n, 1] > 65535:
            s[n, 1] = 65535
        elif s[n, 1] < 0:
            s[n, 1] = 0

        s[n, 1] -= 32768

    wavfile.write(file_name, fs, s.astype(np.int16))
