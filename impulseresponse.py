SWEEPFILE = 'sweep.wav'
RECFILE = 'rec.wav'

import wavfile
from scipy import signal
import numpy as np

def ratio(dB):
    return np.power(10, dB * 1.0 / 20)

def padarray(A, length, before=0):
    t = length - len(A) - before
    if t > 0:
        width = (before, t) if A.ndim == 1 else ([before, t], [0, 0])
        return np.pad(A, pad_width=width, mode='constant')
    else:
        width = (before, 0) if A.ndim == 1 else ([before, 0], [0, 0])
        return np.pad(A[:length - before], pad_width=width, mode='constant')

def filter20_20k(x, sr): # filters everything outside out 20 - 20000 Hz
    nyq = 0.5 * sr
    sos = signal.butter(5, [20.0 / nyq, 20000.0 / nyq], btype='band', output='sos')
    return signal.sosfilt(sos, x)

sr, a, br = wavfile.read(SWEEPFILE, normalized=True)
sr, b, br = wavfile.read(RECFILE, normalized=True)

a = padarray(a, sr*50, before=sr*10)
b = padarray(b, sr*50, before=sr*10)
h = np.zeros_like(b)

for chan in [0, 1]:
    b1 = b[:,chan]

    b1 = filter20_20k(b1, sr)
    ffta = np.fft.rfft(a)
    fftb = np.fft.rfft(b1)
    ffth = fftb / ffta
    h1 = np.fft.irfft(ffth)
    h1 = filter20_20k(h1, sr)

    h[:,chan] = h1

h = h[:10 * sr,:]
h *= ratio(dB=40)

wavfile.write('IR.wav', sr, h, normalized=True)