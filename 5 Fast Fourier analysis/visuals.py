import numpy as np
import matplotlib.pyplot as plt
import core
from scipy import optimize 
import time
import pandas as pd

def plot_DFT(F, f, t, title="DFT", filename="DFT"):
    N = len(F)
    N = np.arange(N)-N/2
    plt.subplot(311)
    plt.ylabel("$h(t)$")
    plt.plot(t, f, label="f")
    plt.subplot(312)
    plt.ylabel("$H_n$")
    plt.plot(N, F.real, label="Re(H)")
    plt.plot(N, F.imag, label="Im(H)")
    plt.legend()
    plt.subplot(313)
    plt.ylabel("$|H_n|^2$")
    plt.plot(N, F.real**2 + F.imag**2, label="F real")
    plt.xlabel(r"$\omega [1/s]$")
    #plt.yscale("log")
    plt.suptitle(title)
    plt.savefig("graphs/"+filename+".pdf")
    plt.show()
    plt.clf()
    return

def plot_DFT2(F, f1, f2, t1, t2, title="DFT", filename="DFT"):
    N = len(F)
    N = np.arange(N)-N/2
    plt.subplot(211)
    plt.ylabel("$h(t)$")
    plt.plot(t2, f2, label="$h$")
    plt.plot(t1, f1, label=r"$F^{-1}(F(h))$")
    plt.legend()
    plt.subplot(212)
    plt.ylabel("$|H_n|^2$")
    plt.plot(N, F.real**2 + F.imag**2, label="F real")
    plt.xlabel(r"$\omega [1/s]$")
    #plt.yscale("log")
    plt.suptitle(title)
    plt.savefig("graphs/"+filename+".pdf")
    plt.show()
    plt.clf()
    return

def exercise_spectrum():
    plt.figure(figsize=(13,6))
    sampling_freq = [882, 1378, 2756, 5512, 11025, 44100]
    notes =      [233.08,    246.94,   277.18,   293.66,   329.63, 369.99,    392.00,   415.30,    440,   466.16,  493.88,  554.37,   587.33,   659.25]
    note_names = ["$A^\#_3$", "$B_3$", "$C^\#_4$", "$D_4$", "$E_4$", "$F^\#_4$", "$G_4$", "$G^\#_4$", "$A_4$", "$A^\#_4$", "$B_4$", "$C^\#_5$", "$D_5$", "$E_5$"]
    for i, freq in enumerate(sampling_freq[::-1]):
        with open(f"audio_files/Bach.{freq}.txt", "r") as file:
            raw_data = file.read()[:-1]
        raw_data = raw_data.split("\n")
        raw_data = np.array(list(map(int, raw_data)))
        F = core.FFT(raw_data)
        N_ = raw_data.shape[0]
        F = np.fft.fft(raw_data)/N_
        N = np.fft.fftfreq(N_, 1/freq)
        # N = len(F)
        # N = (np.arange(N)-N/2)*freq/N
        mask = (1000 > N) * (N > 200)
        plt.plot(N[mask], F[mask].real**2 + F[mask].imag**2, label=f"v_s = {freq} Hz")  
    plt.vlines(notes, 0, 55000)
    for i, note in enumerate(notes):
        plt.text(note+2, 40000, note_names[i]+f" {note} Hz", rotation="vertical")
    plt.xlabel(r"$\nu [Hz]$")
    plt.ylabel("$|H_n|^2$")
    plt.legend()
    plt.suptitle(f"Power spectrum")
    plt.savefig(f"graphs/spectrum.pdf")
    plt.show()
    plt.clf()
    return


def exercise_DFT_functions():
    T = 2*np.pi
    n = 100
    t = np.linspace(0, T, n, endpoint=False)
    N = np.arange(n)-int(n/2)

    f_gauss = np.exp(-t**2)
    f_trig = np.sin(3*t)+np.cos(6*t)
    f_const = t**0

    F_gauss = core.DFT_array(f_gauss)
    F_trig = core.DFT_array(f_trig)
    F_const = core.DFT_array(f_const)

    # F_gauss = core.FFT(f_gauss)
    # F_trig = core.FFT(f_trig)
    # F_const = core.FFT(f_const)

    plot_DFT(F_const, f_const, t, "$h(t) = 1$", "DFT_const")
    plot_DFT(F_trig, f_trig, t, "$h(t) = sin(3t)+cos(6t)$", "DFT_trig")
    plot_DFT(F_gauss, f_gauss, t, "$h(t) = e^{-t^2}$", "DFT_Gauss")

    T = 1.9*np.pi
    n = 100
    t = np.linspace(0, T, n, endpoint=False)
    N = np.arange(n)-int(n/2)

    f = np.sin(3*t)+np.cos(6*t)
    F = core.DFT_array(f)

    plot_DFT(F, f, t, "Leakage\nh(t) = sin(3t)+cos(6t)", "leakage")
    return


def exercise_undersampling():
    T = 2*np.pi
    n = 40
    t1 = np.linspace(0, T, n, endpoint=False)
    N = np.arange(n)-int(n/2)

    v = 25
    f1 = np.cos(v*t1)+np.sin(5*t1)
    F = core.DFT_array(f1)
    t2 = np.linspace(t1[0], t1[-1], len(t1)*10)
    f2 = np.cos(v*t2)+np.sin(5*t2)
    plot_DFT2(F, f1, f2, t1, t2, f"Undersampling\n h(t) = cos({v}t)+sin(5t), v_s = 20 Hz", "undersampling")


def exercise_timing():
    num = 100
    times = []
    x = np.linspace(1, 500, 100)
    for n in x:
        print(n)
        T = 2*np.pi
        t = np.linspace(0, T, int(n), endpoint=False)
        N = np.arange(int(n))-int(n/2)
        f_trig = np.sin(3*t)+np.cos(6*t)
        dt1 = time.time()
        F_trig_vec = core.DFT_array(f_trig)
        dt2 = time.time()
        F_trig = core.DFT(f_trig)
        dt3 = time.time()
        F_trig_np = core.FFT(f_trig)
        dt4 = time.time()
        times.append([dt2-dt1, dt3-dt2, dt4-dt3])
    times = np.array(times).T

    def fit(x, a, b):
        return x*a+b

    times = np.sqrt(times)
    plt.plot(x, times[0], label="Vector")
    plt.plot(x, times[1], label="For loop")
    plt.plot(x, times[2], label="FFT")
    plt.ylabel(r"$\sqrt{t} [s]$")
    #plt.yscale("log")
    plt.xlabel("N")
    plt.title("Computation time")
    pars, cov = optimize.curve_fit(fit, x, times[0])
    stdevs = np.sqrt(np.diag(cov))
    plt.errorbar(x, [fit(i, *pars) for i in x], yerr=0*np.ones(len(x)), label="Fit {:.2e} N + {:.2e}".format(pars[0], pars[1]))    #stdevs[0]*np.ones(len(x))
    pars, cov = optimize.curve_fit(fit, x, times[1])
    stdevs = np.sqrt(np.diag(cov))
    plt.errorbar(x, [fit(i, *pars) for i in x], yerr=0*np.ones(len(x)), label="Fit {:.2e} N + {:.2e}".format(pars[0], pars[1]))
    pars, cov = optimize.curve_fit(fit, x, times[2])
    stdevs = np.sqrt(np.diag(cov))
    plt.errorbar(x, [fit(i, *pars) for i in x], yerr=0*np.ones(len(x)), label="Fit {:.2e} N + {:.2e}".format(pars[0], pars[1]))
    plt.legend()
    plt.savefig("graphs/time_vs_N.pdf")
    plt.show()

def exercise_inverse():
    T = 2*np.pi
    n = 100
    t = np.linspace(0, T, n, endpoint=False)
    N = np.arange(n)-int(n/2)

    v = 5
    f = np.cos(v*t)+np.sin(v*2*t+1)
    F = core.DFT_array(f)
    f_inv = core.IDFT_array(F)

    F2 = core.DFT(f)
    f_inv2 = core.IDFT(F)

    plt.subplot(211)
    plt.ylabel("$h$")
    plt.plot(t, f, label=r"$h$")
    plt.plot(t, f_inv, label=r"$F^{-1}(F(h))$ Vector")
    plt.plot(t, f_inv2, label=r"$F^{-1}(F(h))$ for loop")
    plt.legend()
    plt.subplot(212)
    plt.ylabel(r"$|h - F^{-1}(F(h))|$")
    plt.plot(t, np.abs(f_inv-f), label="Vector DFT error")
    plt.plot(t, np.abs(f_inv2-f), label="For loop DFT error")
    plt.legend()
    plt.xlabel("t")
    plt.suptitle("Inverse Fourier transform")
    plt.savefig("graphs/inverse.pdf")
    plt.show()
    return

def exercise_zero_padding():
    T = 1.5*np.pi
    n = 100
    t = np.linspace(0, T, n, endpoint=False)
    N = np.arange(n)-int(n/2)


    d = 100
    
    f = np.cos(3*t)+np.sin(8*t)+np.sin(30*t)
    f = np.concatenate((f,np.zeros(d)))
    t2 = np.linspace(0, T, n+d, endpoint=False)
    F = core.DFT_array(f)
    print(F.shape, f.shape, t2.shape)
    plot_DFT(F, f, t2, "Zero padding", "zero_padding")


def exercise_stock_market():
    plt.figure(figsize=(12,10))
    for i, t in enumerate([10, 1]):
        index_data = pd.read_csv(f"index_data/index_{t}_year.csv")
        index_data["Date"] = pd.to_datetime(index_data["Date"]) 
        days = []
        for d in index_data["Date"]:
            days.append((d.year-2020)*365+d.month*31+d.day)
        index_data["Date_days"] = days

        def fit(x, k, n):
            return k * x + n
        pars, cov = optimize.curve_fit(fit, index_data["Date_days"], index_data[" High"])
        stdevs = np.sqrt(np.diag(cov))

        plt.subplot(2, 2, 2*i+1)
        if i == 0:
            plt.title("S&P 500 price - 10 years")
        else:
            plt.title("S&P 500 price - 1 year")
        plt.plot(index_data["Date"], index_data[" High"], label="Historical data")
        plt.plot(index_data["Date"], [fit(i, *pars) for i in index_data["Date_days"]], label="Fit {:.2e} $/day * t + {:.2e}".format(pars[0], pars[1]))
        plt.ylabel("Price [$]")
        plt.xlabel("t")
        plt.legend()


        index_data[" High"] -= pars[0]*index_data["Date_days"]+pars[1]
        # plt.subplot(132)
        # plt.plot(index_data["Date_days"], index_data[" High"])

        f = index_data[" High"]
        F = core.FFT(f)
        N_ = f.shape[0]
        F = np.fft.fft(f)/N_
        N = np.fft.fftfreq(N_, 1)

        mask = (N > 0)
        F_2 = F.real**2 + F.imag**2

        plt.subplot(2, 2, 2*i+2)
        if i == 0:
            plt.title("Price spectrum - 10 years")
        else:
            plt.title("Price spectrum - 1 year")
        plt.plot(N[mask], F_2[mask])
        plt.xlabel(r"$\nu [1/day]$")
        plt.yscale("log")
        plt.ylabel("$|H_n|^2$")
    plt.suptitle(f"S&P 500 index - Historical data")
    plt.savefig(f"graphs/index.pdf")
    plt.show()
    plt.clf()
    return