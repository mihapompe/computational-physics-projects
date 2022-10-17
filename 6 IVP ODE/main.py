# =============================================================================
# 
# Author: Miha Pompe
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import time
import pandas as pd
from scipy import optimize
import diffeq_tsint as de
from scipy.integrate import odeint
from matplotlib import cm

# TO DO
# 1. exercise f1
# - DE solution 4x4
#       - T0 = 21
#          - T(t) solution to DE for each method at h = 5, 
#          - error(t) for each method
#       - T0 = -15
#          - same
# - T_error(h), h [10^11, 10^2], pri t = 1e-3, try max_local_error(h), global_error(h), log-log scale, fit lines
# - T_error(h, t) 3D graph
# - time[n], n \propto 1/dt, h not constant
# - Stability, if the final point diverges, chose one method 
#       - max_error(h), colormap, h = complex, color = error
#       - show example of divergence
# - test other implementations in Python for solving DEs
# 2. exercise f2
# - DE solution, for different k, dif A, dif delta (3x1 graph)





if __name__ == "__main__":
    methods = [de.euler, de.heun, de.rk2a, de.rk2b, de.rku4, de.rk45, de.pc4, odeint]
    methods_names = ["Euler", "Heun", "Midpoint", "RK2", "RK4", "RK45", "Predictor-corrector", "Odeint"]

    # Parameters
    T_zun = -5.0
    k = 0.1
    A = 1.0
    delta = 10.0
    x0 = 21.0 # -15.0        # Initial value
    a = 0.0
    b = 1e-3
    tol = 1e-3
    hmax = 1
    hmin = 0.0001

    # Functions
    def f(x, t):
        return t
    
    def f1(T, t):
        return -k*(T-T_zun)
    
    def f2(T, t):
        return -k*(T-T_zun) + A*np.sin(2*np.pi/24*(t-delta))
    
    def linear_fit(x, k, n):
        return k*x+n





    ##### Exercise 6: Extra #####
    # - DE solution, for different k, dif A, dif delta (3x1 graph)
    # plt.figure(figsize=(14,6))
    # plt.suptitle(r"Solution to $\frac{dT}{dt} = -k(T - T_{out}) + Asin(\frac{2\pi}{24}(t-\delta))$"+ ". "+ r"Parameters $h = 1, k = 0.1, A = 1, \delta = 10, T_0 = 21^{\circ}C, T_{out} = -5 ^{\circ}C$")
    # t = np.linspace(0, 100, 100)
    # plt.subplot(1,3,1)
    # for k in np.linspace(0.05, 1, 100):
    #     y = odeint(f2, x0, t)
    #     y = y.flatten()
    #     plt.plot(t, y, c=cm.viridis(k))
    # plt.title(r"$k \in [0.05, 1]$")
    # plt.xlabel("t")
    # plt.ylabel("T")
    # plt.subplot(1,3,2)
    # k = 0.1
    # for A in np.linspace(0.1, 1, 100):
    #     y = odeint(f2, x0, t)
    #     y = y.flatten()
    #     plt.plot(t, y, c=cm.viridis(A))
    # plt.title(r"$A \in [0.1, 1]$")
    # plt.xlabel("t")
    # plt.ylabel("T")
    # plt.subplot(1,3,3)
    # A = 1
    # for delta in np.linspace(0, 10, 100):
    #     y = odeint(f2, x0, t)
    #     y = y.flatten()
    #     plt.plot(t, y, c=cm.viridis(delta/10))
    # plt.title(r"$\delta \in [0, 10]$")
    # plt.xlabel("t")
    # plt.ylabel("T")
    # plt.savefig("graphs/f2.pdf")
    # plt.show()
    ####################


    ##### Exercise 5: Stability #####
    # a = 0.0
    # b = 100.0
    # for i, method in enumerate(methods):
    #     h_s = np.power(10,np.linspace(-1, 1.8, 100))
    #     error_h = []
    #     data = []
    #     for h in h_s:
    #         t = np.arange(a, b, h)
    #         y_0 = T_zun+np.exp(-k*t)*(x0-T_zun)
    #         t1 = time.time()
    #         if i == 7:
    #             y = method(f1, x0, t).flatten()
    #         else:
    #             y = method(f1, x0, t)
    #         dt = time.time() - t1
    #         error = max(np.abs(y - y_0))
    #         #hs = np.ones(len(error))*h
    #         data.append(error)
    #     data = np.array(data)
    #     if len(h_s[data >= 1]) == 0:
    #         plt.plot(h_s, data, label=f"{methods_names[i]}")
    #     else:
    #         h0 = h_s[data >= 1][0]
    #         plt.plot(h_s, data, label=f"{methods_names[i]}, h0 = {round(h0,1)}")
    # plt.hlines(1, h_s[0], h_s[-1], colors="red")
    # plt.xscale("log")
    # plt.legend()
    # plt.title("Method stability")
    # plt.xlabel("h")
    # plt.ylabel("Max absolute error")
    # plt.yscale("log")
    # plt.savefig("graphs/stability.pdf")
    # plt.show()
    ####################


    ##### Exercise 4: Time analysis #####
    # a = 0.0
    # b = 1.0
    # for i, method in enumerate(methods):
    #     h_s = np.power(10,np.linspace(-5, -1, 10))
    #     error_h = []
    #     data = []
    #     for h in h_s:
    #         t = np.arange(a, b, h)
    #         y_0 = T_zun+np.exp(-k*t)*(x0-T_zun)
    #         t1 = time.time()
    #         if i == 7:
    #             y = method(f1, x0, t).flatten()
    #         else:
    #             y = method(f1, x0, t)
    #         dt = time.time() - t1
    #         error = np.abs(y - y_0)
    #         hs = np.ones(len(error))*h
    #         data.append(dt)
    #     pars, cov = optimize.curve_fit(linear_fit, np.log10(h_s), np.log10(data))
    #     plt.plot(h_s, data, label=f"{methods_names[i]}, k = {round(pars[0],2)}, n = {round(pars[1],2)}")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.xlabel("h")
    # plt.ylabel("Computation time [s]")
    # plt.title("Computation time\nFit k*x+n")
    # plt.legend()
    # plt.savefig("graphs/time.pdf")
    # plt.show()
    ####################


    ##### Exercise 3: Error(t, h) #####
    # a = 0.0
    # b = 1.0
    # data = []
    # for i, method in enumerate([de.rk45]):#methods):
    #     h_s = np.linspace(0.001, 0.1, 100)
    #     error_h = []
    #     for h in h_s:
    #         t = np.arange(a, b, h)
    #         y_0 = T_zun+np.exp(-k*t)*(x0-T_zun)
    #         if i == 7:
    #             y = method(f1, x0, t).flatten()
    #         else:
    #             y = method(f1, x0, t)
    #         error = np.abs(y - y_0)
    #         hs = np.ones(len(error))*h
    #         data.append([t, hs, error])
    # data = np.array(data).T
    # data2 = []
    # data2.append(np.concatenate(data[0]))
    # data2.append(np.concatenate(data[1]))
    # data2.append(np.concatenate(data[2]))
    # data2 = np.array(data2)
    # ax = plt.axes(projection='3d')
    # ax.set_xlabel("t")
    # ax.set_ylabel("h")
    # ax.set_zlabel("Global error")
    # ax.scatter3D(data2[0], data2[1], data2[2], c=data2[2], cmap='viridis')
    # ax.set_title('Time progression of global error with rk45 method')
    # plt.savefig("graphs/error_3D.pdf")
    # plt.show()
    ####################

    
    ##### Exercise 2: Error(h) #####
    # a = 0.0
    # b = 1.0
    # data = []
    # for i, method in enumerate(methods):
    #     h_s = np.power(10,np.linspace(-5.5, -1, 30))
    #     error_h = []
    #     for h in h_s:
    #         t = np.arange(a, b, h)
    #         y_0 = T_zun+np.exp(-k*t)*(x0-T_zun)
    #         if i == 7:
    #             y = method(f1, x0, t).flatten()
    #         else:
    #             y = method(f1, x0, t)
    #         error = np.abs(y - y_0)
    #         hs = np.ones(len(error))*h
    #         data.append([t, hs, error])
    #         error_h.append(error[-1])
    #     error_h = np.array(error_h)
    #     pars, cov = optimize.curve_fit(linear_fit, np.log10(h_s[h_s > 1e-2]), np.log10(error_h[h_s > 1e-2]))
    #     plt.plot(h_s, error_h, label=f"{methods_names[i]}, k = {round(pars[0],1)}")
    # plt.legend()
    # plt.title("Global error for different step sizes\n Linear fit k*x+n")
    # plt.ylabel("Global error")
    # plt.xlabel("h")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.savefig("graphs/error_h.pdf")
    # plt.show()
    # plt.clf()
    ####################


    ##### Exercise 1: Solution #####
    # h = 0.1 
    # a = 0.0
    # b = 100.0
    # plt.figure(figsize=(10,13))
    # plt.suptitle(r"Solution to $\frac{dT}{dt} = -k(T - T_{out}), h = 0.1, k = 0.1, T_{out} = -5 ^{\circ}C$")
    # for j, x0 in enumerate([21.0, -15.0]):
    #     error = []
    #     plt.subplot(3,2,1+j)
    #     plt.title(f"Solution, with T0 = {x0} C")
    #     t, x = de.rkf(f1, a, b, x0, tol, hmax, hmin)
    #     plt.plot(t, x, label="RKF")
    #     x_exact = T_zun+np.exp(-k*t)*(x0-T_zun)
    #     plt.plot(t, x_exact, label="Analytical solution")
    #     error.append([t, np.abs(x-x_exact)])
     
    #     for i, method in enumerate(methods):
    #         t = np.linspace(a, b, int((b-a)/h))
    #         y_0 = T_zun+np.exp(-k*t)*(t-T_zun)
    #         if i == 7:
    #             y = method(f1, x0, t).flatten()
    #         else:
    #             y = method(f1, x0, t)
    #         error.append([t, np.abs(y-y_0)])
    #         plt.plot(t, y, "-", label=methods_names[i])
    #     plt.xlabel(r"$t$")
    #     plt.ylabel(r"$T [^{\circ}C]$")
    #     plt.legend()

    #     plt.subplot(3,2,3+j)
    #     plt.title("Absolute error of the solution")
    #     for i, (t, x) in enumerate(error):
    #         if i == 0:
    #             plt.plot(t, x, label="RKF")
    #         else:
    #             plt.plot(t, x, label=methods_names[i-1])
    #     plt.yscale("log")
    #     plt.xlabel(r"$t$")
    #     plt.ylabel(r"$|T_e - T| [^{\circ}C]$")
    #     plt.legend()

    #     plt.subplot(3,2,5+j)
    #     t = np.linspace(0, 100, 100)
    #     for k in np.linspace(0.05, 1, 100):
    #         y = odeint(f1, x0, t)
    #         y = y.flatten()
    #         plt.plot(t, y, c=cm.viridis(k))
    #     plt.title(r"$k \in [0.05, 1]$")
    #     plt.xlabel("t")
    #     plt.ylabel(r"$T [^{\circ}C]$")
    #     k = 0.1

    # plt.savefig("graphs/solution.pdf")
    # plt.show()
    ##############################
