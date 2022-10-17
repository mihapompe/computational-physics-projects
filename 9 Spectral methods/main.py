# =============================================================================
# Spectral methods for IVP PDE
# Miha Pompe
# =============================================================================

import core
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize 

# =============================================================================
# Parameters
a = 1
N = 100
t_max = 1
x = np.linspace(0, a, N)
t = np.linspace(0, t_max, N)
x_0 = core.gauss(x)  # initial value
h_x = x[1]-x[0]
h_t = t[1]-t[0]
x_0[0] = 0
x_0[-1] = 0
parameters = [N, a, t_max, h_x, h_t]
# plt.figure(figsize=(15,5))
# =============================================================================

# =============================================================================
# TO-DO
# + Gauss
#   + 3D plot fourier                                       f(0) = f(a)
#   + 3D plot of fourier, spline, difference between them   Dirichlet
#   + Stability N vs max                                    Dirichlet
#   + Time vs N (for both methods)                          Dirichlet
#   + Peak at a/2 vs time for fourier (energy)              Dirichlet and f(0) = f(a)
# + Sine
#   + 3D plot of fourier, spline, difference                Dirichlet
#   + Peak at a/2 sin(odd*pi*x/a) vs time vs freq           Dirichlet
# + Square
# =============================================================================


# =============================================================================
# Fourier method - Equal
# T, time = core.fourier_solve(x_0, t, parameters)

# Fourier method - Dirichlet
# T, time = core.fourier_solve(x_0, t, parameters, True)

# Spline method - Dirichlet
# T, time = core.spline_solve(x_0, t, x, parameters)

# Fourier method - Dirichlet - Sine
# x_0 = np.sin(np.pi*x/a)
# T, time = core.fourier_solve(x_0, t, parameters, True)

# Spline method - Sine
# x_0 = np.sin(np.pi*x/a)
# T, time = core.spline_solve(x_0, t, x, parameters)

# Fourier method - Square
# x_0 = core.square(x)
# T, time = core.fourier_solve(x_0, t, parameters, True)
# =============================================================================

##################################
# T, time = core.fourier_solve(x_0, t, parameters)
# core.plot_T(x, t, T, "Fourier method, T(0) = T(a)", "T", "fourier_gauss")
##################################

##################################
# T, time = core.fourier_solve(x_0, t, parameters, True)
# core.plot_T(x, t, T, "Fourier method, Dirichlet", "T", "fourier_gauss_dirichlet")
##################################

##################################
# T, time = core.spline_solve(x_0, t, x, parameters)
# core.plot_T(x, t, T, "Spline method, Dirichlet", "T", "spline_gauss_dirichlet")
##################################

##################################
# T1, time = core.fourier_solve(x_0, t, parameters, True)
# T2, time = core.spline_solve(x_0, t, x, parameters)
# core.plot_T(x, t, np.abs(T1-T2), "Difference between methods, Dirichlet", r"$\Delta T$", "difference_gauss")
##################################

##################################
# T_mid = []
# times1 = []
# times2 = []
# Ns = 10**np.linspace(1, 2, 20)
# for N_ in Ns:
#     print(N_)
#     x = np.linspace(0, a, int(N_))
#     t = np.linspace(0, t_max, int(N_))  
#     x_0 = core.gauss(x)  # initial value
#     h_x = x[1]-x[0]
#     h_t = t[1]-t[0]
#     x_0[0] = 0
#     x_0[-1] = 0  
#     parameters = [int(N_), a, t_max, h_x, h_t]
#     T, time = core.fourier_solve(x_0, t, parameters, True)
#     times1.append(time)
#     T, time = core.spline_solve(x_0, t, x, parameters)
#     times2.append(time)

# def fit(x, k, n):
#     return k * x ** n

# plt.scatter(Ns, times1, label = "Fourier method")
# pars, cov = optimize.curve_fit(fit, Ns, times1)
# print(pars) 
# plt.plot(Ns, [fit(i, *pars) for i in Ns], label=f"Fourier fit n = {round(pars[1],1)}")
# plt.scatter(Ns, times2, label = "Spline method")
# pars, cov = optimize.curve_fit(fit, Ns, times2) 
# print(pars)
# plt.plot(Ns, [fit(i, *pars) for i in Ns], label=f"Spline fit n = {round(pars[1],1)}")
# plt.legend()
# plt.xlabel("N")
# plt.ylabel("t [s]")
# plt.yscale("log")
# plt.xscale("log")
# plt.title("Computation time")
# plt.savefig("graphs/time.pdf")
# plt.show()
##################################

##################################
# max1 = []
# max2 = []
# Ns = 10**np.linspace(0.8, 2.5, 10)
# for N_ in Ns:
#     print(N_/max(Ns)*100)
#     x = np.linspace(0, a, int(N_))
#     t = np.linspace(0, t_max, int(N_))  
#     x_0 = core.gauss(x)  # initial value
#     h_x = x[1]-x[0]
#     h_t = t[1]-t[0]
#     x_0[0] = 0
#     x_0[-1] = 0  
#     parameters = [int(N_), a, t_max, h_x, h_t]
#     T, time = core.fourier_solve(x_0, t, parameters, True)
#     l = [max(T[int(N_/4)]), max(T[int(N_/2)]), max(T[int(3*N_/4)]), max(T[-1])]
#     max1.append(l)
#     T, time = core.spline_solve(x_0, t, x, parameters)
#     l = [max(T[int(N_/4)]), max(T[int(N_/2)]), max(T[int(3*N_/4)]), max(T[-1])]
#     max2.append(l)
# max1 = np.array(max1).T
# max2 = np.array(max2).T
# ts = [0.25, 0.5, 0.75, 1]
# for i in range(4):
#     plt.plot(Ns, max1[i], label=f"Fourier t = {ts[i]}")
#     plt.plot(Ns, max2[i], label=f"Spline t = {ts[i]}")
# plt.legend()
# plt.title("Stability")
# plt.xlabel("N")
# plt.ylabel("T")
# plt.savefig("graphs/stability.pdf")
# plt.show()
##################################

##################################
# mid1 = []
# mid2 = []
# times = []
# Ns = [300]
# for N_ in Ns:
#     print(N_/max(Ns)*100)
#     x = np.linspace(0, a, int(N_))
#     t = np.linspace(0, t_max, int(N_))  
#     times.append(t)
#     x_0 = core.gauss(x)  # initial value
#     h_x = x[1]-x[0]
#     h_t = t[1]-t[0]
#     x_0[0] = 0
#     x_0[-1] = 0  
#     parameters = [int(N_), a, t_max, h_x, h_t]
#     T, time = core.fourier_solve(x_0, t, parameters)
#     mid1.append(np.sum(T, axis=1)/np.sum(T[0]))#T[:,N_//2])
#     T, time = core.fourier_solve(x_0, t, parameters, True)
#     mid2.append(np.sum(T, axis=1)/np.sum(T[0]))#T[:,N_//2])
# for i in range(len(Ns)):
#     plt.plot(times[i], mid1[i], label=f"T(0) = T(a)")
#     plt.plot(times[i], mid2[i], label=f"Dirichlet")
# plt.legend()
# plt.title("Energy, D = 0.5")
# plt.xlabel("t")
# plt.ylabel(r"$S/S_0$")
# plt.savefig("graphs/energy2.pdf")
# plt.show()
##################################


##################################
# x_0 = np.sin(5*np.pi*x/a)
# T, time = core.fourier_solve(x_0, t, parameters, True)
# core.plot_T(x, t, T, "High frequency sine function\nDirichlet, Fourier method", "T", "sine_high")
##################################


##################################
# x_0 = core.square(x)
# T, time = core.fourier_solve(x_0, t, parameters, True)
# core.plot_T(x, t, T, "Square function, Dirichlet, Fourier method", "T", "square")
##################################

##################################
# x_0 = core.delta(x)
# T, time = core.fourier_solve(x_0, t, parameters, True)
# core.plot_T(x, t, T, "Delta function, Dirichlet, Fourier method", "T", "delta")
##################################