
import numpy as np
import matplotlib.pyplot as plt
import time

def FFT(h):
    N = h.shape[0]
    H = np.fft.fft(h)
    H = np.roll(H, int(N/2))
    return H/N

def DFT(h):
    N = h.shape[0]
    out = []
    for k in range(N):
        f = 0
        for n in range(N):
            f += (np.exp(-2j*np.pi*k*n/N))*h[n]
        out.append(f)
    out = np.array(out)/N
    out = np.roll(out, int(N/2))
    return out

def DFT_array(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    out = np.dot(M, x)/N
    out = np.roll(out, int(N/2))
    return out

def IDFT(H):
    N = H.shape[0]
    H = np.roll(H, -int(N/2))
    out = []
    for k in range(N):
        f = 0
        for n in range(N):
            f += (np.exp(2j*np.pi*k*n/N))*H[n]
        out.append(f)
    return out

def IDFT_array(x):
    N = x.shape[0]
    x = np.roll(x, -int(N/2))
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    out = np.dot(M, x)
    return out