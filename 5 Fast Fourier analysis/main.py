import core
from visuals import *
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import optimize 
import pandas as pd
import clean_data

data = []
Hs = []
hs = []
raw_cors = []
raw_cors_four = []
auto_cors = []
h_cors = []
filenames = ["mix22", "bubomono", "bubo2mono", "mix", "mix1", "mix2"]

cor_with_bubo1 = []
cor_with_bubo2 = []

# # Get data and generate FFTs
# for i, filename in enumerate(filenames):
#     with open(f"audio_data/{filename}.txt", "r") as file:
#         raw_data = file.read()[:-1]
#     raw_data = raw_data.split("\n")
#     raw_data = np.array(list(map(int, raw_data)))
#     data.append(raw_data)
#     x = np.linspace(0, len(raw_data), len(raw_data))
#     H, v, P = core.FFT(raw_data)
#     Hs.append([H, v, P])
#     h_ = core.IFFT(H)
#     hs.append(h_)
#     cor_raw_four = core.corelation(H, H)
#     cor_raw = core.corelation_numpy(raw_data, raw_data)
#     raw_cors.append(cor_raw)
#     #h_cors.append(core.IFFT(cor_raw))
#     cor = cor_raw_four.real**2 + cor_raw_four.imag**2
#     raw_cors_four.append(cor)
#     auto_cors.append(cor)
# #     plt.subplot(3,2,i+1)
# #     plt.plot(cor_raw)
# #     print(filename)
# # plt.show()

def correlate_with_owls():
    for i in [5, 3,4]:
        plt.subplot(2,1,1)
        F1, _, F1p = core.FFT(core.corelation(data[1], data[i])[::2])
        F2, _, F2p = core.FFT(core.corelation(data[2], data[i])[::2])
        F1, F2 = F1.real, F2.real
        plt.plot(F1p[len(F1p)//2:], alpha=0.8, label=filenames[i])
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(F2p[len(F2p)//2:], alpha=0.8, label=filenames[i])
        plt.legend()
        print(i)
    plt.show()

def time_complexity():
    filenames = ["mix22"]
    t_len = 17
    test_time = np.concatenate((2**np.arange(0, t_len), 2**np.arange(0, t_len)+1))#np.linspace(1000, 22050, 150)#[10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 150000, 180000, 200000, 220502]
    # Get data and generate FFTs
    for i, filename in enumerate(filenames):
        time_data = []
        time_data_scipy = []
        time_data_cor = []
        time_data_cor_numpy = []
        with open(f"audio_data/{filename}.txt", "r") as file:
            raw_data = file.read()[:-1]
        raw_data = raw_data.split("\n")
        raw_data = np.array(list(map(int, raw_data)))
        print(len(raw_data))
        data.append(raw_data)
        x = np.linspace(0, len(raw_data), len(raw_data))
        for t in test_time:
            time1 = time.time()
            H, v, P = core.FFT(raw_data[:int(t)])
            dt = time.time()-time1
            time_data.append(dt)
            time1 = time.time()
            H1 = core.FFT_scipy(raw_data[:int(t)])
            dt = time.time()-time1
            time_data_scipy.append(dt)
        Hs.append([H, v, P])
        h_ = core.IFFT(H)
        hs.append(h_)
        for t in test_time:
            time1 = time.time()
            cor_raw = core.corelation(H[:int(t)], H[:int(t)])
            dt = time.time()-time1
            time_data_cor.append(dt)
            time1 = time.time()
            cor_raw = core.corelation_numpy(raw_data[:int(t)], raw_data[:int(t)])
            dt = time.time()-time1
            time_data_cor_numpy.append(dt)
        raw_cors.append(cor_raw)
        h_cors.append(core.IFFT(cor_raw))
        cor = cor_raw.real**2 + cor_raw.imag**2
        auto_cors.append(cor)
    dot_size = 10
    plt.scatter(test_time[:t_len], time_data[:t_len], s = dot_size, label=r"FFT Numpy $2^m$")
    plt.scatter(test_time[:t_len], time_data_scipy[:t_len], s = dot_size, label=r"FFT Scipy $2^m$")
    plt.scatter(test_time[:t_len], time_data_cor[:t_len], s=dot_size, label=r"Autocor. FFT $2^m$")
    plt.scatter(test_time[:t_len], time_data_cor_numpy[:t_len], s=dot_size, label=r"Autocor. Scipy $2^m$")
    plt.scatter(test_time[t_len:], time_data[t_len:], s = dot_size, label=r"FFT Numpy $2^m+1$")
    plt.scatter(test_time[t_len:], time_data_scipy[t_len:], s = dot_size, label=r"FFT Scipy $2^m+1$")
    plt.scatter(test_time[t_len:], time_data_cor[t_len:], s=dot_size, label=r"Autocor. FFT $2^m+1$")
    plt.scatter(test_time[t_len:], time_data_cor_numpy[t_len:], s=dot_size, label=r"Autocor. Scipy $2^m+1$")
    plt.xlabel("N")
    plt.ylabel("t [s]")
    plt.legend()
    plt.title("Computation time of FFT and autocorrelation")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig("graphs/time_FFT.pdf")
    plt.show()


def lidar_analysis():
    plt.figure(figsize=(11,13))
    for i, label in enumerate(["long", "short"]):
        data = clean_data.clean_data(label)
        data[2] = 2*data[2]/max(data[2])-1
        data[1] = data[1]-min(data[1])
        data[1] = 2*data[1]/max(data[1])-1
        delta = (data[0, -1] - data[0, 0])/len(data[0])
        print(delta, 1/delta)
        H_i, v_i, power_i = core.FFT(data[1], delta)
        H_o, v_o, power_o = core.FFT(data[2], delta)
        
        plt.subplot(4,2,1+i)
        plt.title(label.capitalize() + " range measurement")
        plt.plot(data[2][:50], label="Square signal")
        plt.plot(data[1][:50], label="Measured signal")
        plt.ylabel(r"$\propto U$")
        plt.legend()
        plt.subplot(4,2,3+i)
        plt.title("Power spectrum")
        l = len(v_i)
        plt.plot(v_i[l//2:], power_i[l//2:], label="Square signal")
        plt.plot(v_o[l//2:], power_o[l//2:], label="Measured signal")
        plt.xlabel(r"$\nu [Hz]$", loc="right")
        plt.legend()
        cor = core.corelation(H_i, H_o)
        plt.subplot(4,2,5+i)
        #plt.yscale("log")
        plt.title("Correlation spectrum")
        plt.xlabel(r"$\nu [Hz]$", loc="right")
        power = cor.real**2 +  cor.imag**2
        plt.plot(v_i[l//2:], power[l//2:])
        plt.subplot(4,2,7+i)
        plt.title("Correlated signal")
        plt.xlabel("t [s]", loc = "right")
        icor = core.IFFT(cor)
        print(len(data[1]), len(icor))
        l_icor = len(icor)
        plt.plot(np.arange(l_icor)*delta, icor)
    plt.suptitle("LIDAR data analysis\nSpectrum peaks at 49.95 Hz")
    plt.savefig("graphs/lidar.pdf")
    plt.show()


def spectrogram():
    for i in range(6):
        plt.figure(figsize=(12,8))
        #i = 0
        plt.subplot(2,3,1)
        plt.ylabel("Original signal")
        plt.title("Input signal")
        plt.plot(np.arange(len(data[i][10:]))*0.00002067, data[i][10:])
        plt.xlabel("t [s]", loc="right")
        plt.subplot(2,3,2)
        l = len(Hs[i][2])
        plt.plot(Hs[i][1][l//2:-int(l/2.5)], Hs[i][2][l//2:-int(l/2.5)]/1000)
        plt.xlabel("v [Hz]", loc="right")
        plt.title("Power spectrum")
        plt.subplot(2,3,3)
        plt.specgram(data[i], Fs=44100)#, scale="dB")
        plt.title(filenames[i])
        plt.xlabel("t [s]", loc="right")
        cbar = plt.colorbar()
        cbar.set_label("psd")

        plt.subplot(2,3,4)
        plt.ylabel("Autocorrelated signal")
        plt.plot(np.arange(len(raw_cors[i][10:]))*0.00002067/2, raw_cors[i][10:]*1e-7)
        plt.xlabel("t [s]", loc="right")
        plt.subplot(2,3,5)
        plt.xlabel("v [Hz]", loc="right")
        l = len(raw_cors_four[i])
        plt.plot(Hs[i][1][l//2:-int(l/2.5)], raw_cors_four[i][l//2:-int(l/2.5)]*10000)
        plt.subplot(2,3,6)
        plt.specgram(raw_cors[i], Fs=2*44100)#, scale="dB")
        plt.xlabel("t [s]", loc="right")
        cbar = plt.colorbar()
        cbar.set_label("psd")

        plt.suptitle("Owl signal analysis")
        plt.savefig(f"graphs/spectrogram{i}.pdf")
        print(i)
        #plt.show()


def bird_analysis():
    plt.figure(figsize=(15,8))
    color = ["brown", "red", "violet", "green", "blue", "orange"]
    for i in range(6):
        H, v, P = Hs[i]
        cor = auto_cors[i]
        plt.subplot(2,1,1)
        mask = (v > 0)*(v < 700)
        plt.plot(v[mask], P[mask], linewidth=1.5, label = filenames[i], alpha=0.8)
        plt.legend()
        plt.subplot(2,1,2)
        mask = (v > 0)*(v < 700)
        plt.plot(v[mask], cor[mask], linewidth=1.5, label = filenames[i], alpha=0.8)
        plt.legend()
    plt.suptitle("Power spectrum of original and autocorrelated signal")
    plt.savefig("graphs/spectrum.pdf")
    plt.show()

#spectrogram()
#time_complexity()
#bird_analysis()
lidar_analysis()