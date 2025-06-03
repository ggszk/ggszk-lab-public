import numpy as np
from envelope import ADSR
import matplotlib.pyplot as plt

fs = 44100

A = 0.1
D = 0.4
S = 0.5
R = 0.4
gate = 1
duration = 2
offset = 0
depth = 1

length_of_s = int(fs * duration)

e = ADSR(fs, A, D, S, R, gate, duration)
for n in range(length_of_s):
    e[n] = offset + e[n] * depth

t = np.zeros(length_of_s)
for n in range(length_of_s):
    t[n] = n / fs

plt.figure()
plt.plot(t, e)
plt.axis([0, 2, 0, 1])
plt.xlabel('time [s]')
plt.ylabel('amplitude')
plt.savefig('p6_5.png')

plt.show()
