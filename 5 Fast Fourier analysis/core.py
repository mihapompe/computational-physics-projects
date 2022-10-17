
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import time
import scipy.fft as sc

def FFT(h, delta = 1/44100):
    N = h.shape[0]
    H = np.fft.fft(h)/N
    H = np.fft.fftshift(H)
    v = np.fft.fftfreq(int(N), delta)
    v = np.fft.fftshift(v)
    power = H.real**2 + H.imag**2
    return H, v, power

def FFT_scipy(h, delta = 1/44100):
    N = h.shape[0]
    H = sc.fft(h)
    v = sc.fftfreq(int(N), delta)
    return H


def autocorelation(n, h):
    N = len(h)
    out = 0
    for i in range(0, N-n):
        out += h[i+n]*h[i]
    return out/(N-n)

def IFFT(H):
    N = H.shape[0]
    H = np.fft.fftshift(H)*N
    h = np.fft.ifft(H)
    return h

def corelation(F, G):
    if len(F) > len(G):
        F = F[:len(G)]
    elif len(F) < len(G):
        G = G[:len(F)]
    N = len(F)
    cor = np.flip(F)*G
    #cor = np.fft.fftshift(cor)
    return cor/N

def corelation_numpy(f, g):
    return signal.fftconvolve(f, g)

