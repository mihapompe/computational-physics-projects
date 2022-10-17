# =============================================================================
# Newton's law
# Author: Miha Pompe
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import time
import pandas as pd
from scipy import optimize, special
import diffeq_2 as de
from scipy.integrate import odeint
from matplotlib import cm
import scipy.signal as sig

# Parameters
a = 0
b = 30
tol = 1e-6
hmax = 1
hmin = 0.0001
x0 = 0.0
v0 = 0.0

def f(x):
    return -np.sin(x)

def f_v(x, t, args=None):
    d = np.zeros_like(x)
    d[0] = x[1]     # x' = v
    d[1] = -np.sin(x[0])    # v' = -x
    return d

def f_resonance(x, t, args=None):
    if args is not None:
        v, omega, beta = args
    d = np.zeros_like(x)
    d[0] = x[1]
    d[1] = v*np.cos(omega*t)-np.sin(x[0])-beta*x[1]
    return d

def f_van_der_pol(x, t, args = None):
    if args is not None:
        v, omega, lamda = args
    d = np.zeros_like(x)
    d[0] = x[1]
    d[1] = v*np.cos(omega*t)+lamda*(1-x[0]**2)*x[1]-x[0]
    return d

methods_names = ["Euler", "Heun", "Midpoint", "RK2", "RK4", "Predictor-corrector", "RK45", "RKF", "Verlet", "PEFRL"]

def solve(i, f_name, x0, v0, t, args = None):
    methods = [de.euler, de.heun, de.rk2a, de.rk2b, de.rku4, de.pc4, de.rk45, de.rkf, de.verlet, de.pefrl]
    if f_name == "case1":
        f1 = f_v
        f2 = f
    elif f_name == "case2":
        f1 = f_resonance
    elif f_name == "case3":
        f1 = f_van_der_pol
    if 0 <= i < 6:
        x = methods[i](f1, np.array([x0, v0]), t)
        x = x.T
        x, v = x[0], x[1]
    elif i == 6:
        if f_name in ["case2", "case3"]:
            x, _ = methods[i](f1, np.array([x0, v0]), t, args)
        else:
            x, _ = methods[i](f1, np.array([x0, v0]), t)
        x = x.T
        x, v = x[0], x[1]
    elif i == 7:
        t, x = methods[i](f1, a, b, np.array([x0, v0]), tol, hmax, hmin)
        x, v, t = x[:,0], x[:,1], t
    elif 7 < i <= 9:
        x, v = methods[i](f2, x0, v0, t)
    return x, v, t

def exercise_1():
    width, height = 1, 3
    t = np.linspace(a, b, 100)
    x0s = np.linspace(1, 3, 100)
    lim = x0s<=2
    xs, vs = [], []
    plt.figure(figsize=(15,5))
    plt.suptitle("Mathematical oscillator, $x'' + sin(x) = 0$, Verlet")

    plt.subplot(width, height, 1)
    for x0 in x0s:
        x, v, t = solve(8, "case1", x0, v0, t)
        xs.append(x)
        vs.append(v)
        t_max = t <= 10
        if x0 <= 2:
            plt.plot(t[t_max], x[t_max]/x0, c = cm.winter(x0/2))
    plt.xlabel("$t$")
    plt.ylabel("$x/x_0$", loc="top")
    plt.title("DE solution")

    plt.subplot(width, height, 2)
    for i in range(len(xs)):
        if x0s[i] <= 2:
            plt.plot(xs[i]/x0s[i], vs[i], c = cm.winter(x0s[i]/2))
    plt.colorbar(cm.ScalarMappable(norm=cm.colors.Normalize(1, 2), cmap=cm.winter), label = "$x_0$", location="right")
    plt.xlabel("$x/x_0$")
    plt.ylabel("$v$", loc="top")
    plt.title("Phase space")

    t0s = []
    for i in range(len(xs)):
        peak, _ = sig.find_peaks(xs[i])
        t0s.append(t[peak[0]])
    plt.subplot(width, height, 3)
    plt.ylabel("$t_0$", loc="top")
    plt.xlabel("$x_0$")
    plt.title("Oscillation time")
    plt.plot(x0s, t0s, label="Numerical solution")
    plt.plot(x0s, 4*special.ellipk(np.sin(x0s/2)**2), c="red", label="Eliptic integral")
    plt.legend()

    plt.savefig("graphs/method_analysis.pdf")
    plt.show()

def exercise_2():
    width, height = 3, 2
    times = []
    x0 = 1.0
    v0 = 0.0
    t = np.linspace(1, 1000, 2000)
    x_real, v_real, t_real = solve(9, "case1", x0, v0, t)
    E0 = 1-np.cos(x0)
    plt.figure(figsize=(10,15))
    for method_num in range(0, 10):
        t = np.linspace(1, 500, 10000)
        t1 = time.time()
        x, v, t = solve(method_num, "case1", x0, v0, t)
        dt = t1-time.time()
        times.append(dt)
        peaks, _ = sig.find_peaks(x)
        plt.subplot(width, height, 1)
        plt.ylabel("$n^{th} aplitude$")
        plt.xlabel("$n$")
        plt.yscale("log")
        plt.title("Amplitude deviation")
        plt.plot(np.abs(1-x[peaks]), label=methods_names[method_num])

        plt.subplot(width, height, 2)
        if method_num != 0:
            plt.title("Energy")
            plt.xlabel("$t$")
            plt.ylabel("$|E-E_0|$")
            plt.yscale("log")
            p = 30
            plt.plot(t[::p], np.abs(1-np.cos(x[::p])+v[::p]**2/2-E0), label=methods_names[method_num])

        plt.subplot(width, height, 3)
        plt.title("Absolute error")
        plt.yscale("log")
        plt.xlabel("t")
        plt.ylabel("$|x-x_{ref}|$")
        p = 10
        if len(x) != len(x_real):
            x_real2, v_real2, t_real2 = solve(9, "case1", x0, v0, t)
            plt.plot(t[::p], np.abs(x-x_real2)[::p], label=methods_names[method_num])
        else:
            plt.plot(t[::p], np.abs(x-x_real)[::p], label=methods_names[method_num])

        plt.subplot(width, height, 5)
        plt.plot(x, v, label=methods_names[method_num])
        plt.xlabel("$x$")
        plt.ylabel("$v$")
        plt.title("Phase space")
        ax = plt.gca()
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        plt.legend()

    plt.subplot(width, height, 4)
    ns = 10**np.arange(1, 5)

    for method_num in range(1, 10):
        times = []
        for n in ns:
            t = np.linspace(0, 500, int(n))
            t1 = time.time()
            x, v, t = solve(method_num, "case1", x0, v0, t)
            dt = time.time()-t1
            times.append(dt)
        plt.plot(ns, times, label=methods_names[method_num])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of points")
    plt.ylabel("t[s]")
    plt.title("Time complexity")

    plt.subplot(width, height, 6)
    t = np.linspace(0, 10, 100)
    for v0 in np.linspace(0, 3.0, 100):
        x, v, t = solve(9, "case1", x0, v0, t)
        plt.plot(x, v, c = cm.winter(v0/3.5))
    plt.xlabel("x")
    plt.ylabel("v")
    ax = plt.gca()
    lim_max = 3
    ax.set_xlim([-lim_max, lim_max])
    ax.set_ylim([-lim_max, lim_max])
    plt.colorbar(cm.ScalarMappable(norm=cm.colors.Normalize(0, 3), cmap=cm.winter), label = "$v_0$", location="right")
    plt.title("Phase space for different $v_0$")

    plt.savefig("graphs/method_comparison3.pdf")
    plt.show()
    
def exercise_3():
    width, height = 2, 2
    t = np.linspace(0, 10, 50)
    x0 = 1.0
    v0 = 0.0
    plt.figure(figsize=(13,10))
    plt.suptitle(r"Excited damped mathematical oscillator, $x'' + \beta x' + sin(x) = v cos(\omega t)$, RK45")

    vs = np.linspace(0.5, 10, 500)
    omegas = np.linspace(0, 5, 500)
    betas = np.linspace(0, 1, 100)

    for v_ in vs:
        #v = 10.0
        omega = 2/3
        beta = 0.5
        args = [v_, omega, beta]
        plt.subplot(width, height, 1)
        x, v, t = solve(6, "case2", x0, v0, t, args)
        plt.plot(x, v, c = cm.winter(v_/10))
    plt.xlabel("x")
    plt.ylabel("v")
    plt.title(r"Phase space for different $v$, $\omega = 2/3$, $\beta = 0.5$")
    plt.colorbar(cm.ScalarMappable(norm=cm.colors.Normalize(0.5, 10), cmap=cm.winter), label = "$v$", location="right")
    print("Graph1")

    t = np.linspace(0, 10, 50)
    for omega in omegas:
        v_ = 0.2
        #omega = 2/3
        beta = 0.5#0.7
        args = [v_, omega, beta]
        x, v, t = solve(6, "case2", x0, v0, t, args)
        t_max = t < 10
        plt.subplot(width, height, 2)
        plt.plot(x[t_max], v[t_max], c = cm.winter(omega/5))
    plt.xlabel("x")
    plt.ylabel("v")
    plt.title(r"Phase space for different $\omega$, $v = 0.2$, $\beta = 0.5$")
    plt.colorbar(cm.ScalarMappable(norm=cm.colors.Normalize(0, 5), cmap=cm.winter), label = "$\omega$", location="right")
    print("Graph2")

    t = np.linspace(0, 10, 50)
    for beta in betas:
        v_ = 0.2
        omega = 2/3
        #beta = 0.5#0.7
        args = [v_, omega, beta]
        x, v, t = solve(6, "case2", x0, v0, t, args)
        plt.subplot(width, height, 3)
        plt.plot(x, v, c = cm.winter(beta))
    plt.xlabel("x")
    plt.ylabel("v")
    plt.title(r"Phase space for different $\beta$, $v = 0.2$, $\omega = 2/3$")
    plt.colorbar(cm.ScalarMappable(norm=cm.colors.Normalize(0.1, 1), cmap=cm.winter), label = r"$\beta$", location="right")
    print("Graph3")

    t = np.linspace(0, 100, 500)
    omegas = np.linspace(0, 3, 100)
    betas = np.linspace(0.1, 0.5, 5)
    plt.subplot(width, height, 4)
    for beta in betas:
        print(beta)
        peaks = []
        for omega in omegas:
            v_ = 0.2
            args = [v_, omega, beta]
            x, v, t = solve(6, "case2", x0, v0, t, args)
            peak, _ = sig.find_peaks(np.abs(x))
            peaks.append(np.abs(x[peak[-1]]))
        peaks = np.array(peaks)
        plt.plot(omegas, peaks/v_, c = cm.winter(beta))
    plt.xlabel("$\omega$")
    plt.ylabel("$x_0/v$")
    plt.title("Resonance curve")
    plt.colorbar(cm.ScalarMappable(norm=cm.colors.Normalize(0.1, 0.5), cmap=cm.winter), label = r"$\beta$", location="right")

    plt.savefig("graphs/resonance.pdf")
    plt.show()

def exercise_4():
    width, height = 1, 3
    x0 = 0.0
    v0 = 1.0
    plt.figure(figsize=(15,6))
    plt.suptitle(r"Van der Pol oscillator, $x'' - \lambda x' (1-x^2) + x = v cos(\omega t)$, RK45")

    vs = np.linspace(0.5, 10, 100)
    t = np.linspace(0, 10, 200)
    for v_ in vs:
        omega = 1.0
        lamda = 1.0
        args = [v_, omega, lamda]
        plt.subplot(width, height, 1)
        x, v, t = solve(6, "case3", x0, v0, t, args)
        plt.plot(x, v, c = cm.winter(v_/10))
    plt.xlabel("x")
    plt.ylabel("v")
    plt.title(r"Phase space for different $v$, $\omega = 1$, $\lambda = 1$")
    plt.colorbar(cm.ScalarMappable(norm=cm.colors.Normalize(0.5, 10), cmap=cm.winter), label = "$v$", location="bottom")
    print("Graph1")

    omegas = np.linspace(1, 10, 5)
    t = np.linspace(0, 5, 200)
    for omega in omegas:
        v_ = 10.0
        lamda = 1.0
        args = [v_, omega, lamda]
        x, v, t = solve(6, "case3", x0, v0, t, args)
        t_max = t < 10
        plt.subplot(width, height, 2)
        plt.plot(x[t_max], v[t_max], c = cm.winter(omega/10))
    plt.xlabel("x")
    plt.ylabel("v")
    plt.title(r"Phase space for different $\omega$, $v = 10$, $\lambda = 1$")
    plt.colorbar(cm.ScalarMappable(norm=cm.colors.Normalize(1, 10), cmap=cm.winter), label = "$\omega$", location="bottom")
    print("Graph2")

    lamdas = np.linspace(0.5, 5, 100)
    t = np.linspace(0, 10, 100)
    for lamda in lamdas:
        v_ = 10.0
        omega = 1.0
        args = [v_, omega, lamda]
        x, v, t = solve(6, "case3", x0, v0, t, args)
        plt.subplot(width, height, 3)
        plt.plot(x, v, c = cm.winter(lamda/5))
    plt.xlabel("x")
    plt.ylabel("v")
    ax = plt.gca()
    lim_max = 13
    ax.set_xlim([-lim_max, lim_max])
    ax.set_ylim([-lim_max, lim_max])
    plt.title(r"Phase space for different $\lambda$, $v = 10$, $\omega = 1$")
    plt.colorbar(cm.ScalarMappable(norm=cm.colors.Normalize(0.5, 5), cmap=cm.winter), label = r"$\lambda$", location="bottom")
    print("Graph3")

    plt.savefig("graphs/van_der_pol.pdf")
    plt.show()

if __name__ == "__main__":
    #exercise_1()
    exercise_2()
    #exercise_3()
    #exercise_4()