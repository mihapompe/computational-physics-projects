# =============================================================================
# Differnce method
# Author: Miha Pompe
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import time
from matplotlib import cm

# =============================================================================
# To Do
# - Coherent state
#       - Real, Imag, PDS in 3D
#       - Error PDS in 3D
#       + Peaks over time for different N
#       + Error of peaks for different N {10, 50, 100, 200, 300, 400} log
#       + Max error vs N
# - Wave packet
# - Improvements
# =============================================================================

# =============================================================================
# Basic parameters
# =============================================================================

k = 0.04
lamda = 10
alpha = k ** (1/4)
omega = k ** (1/2)
x_min = -40
x_max = 40
t_max = 30
N = 200

# gauss
# lamda = 0.25
# sigma = 1/20
# k_0 = 50*np.pi
# x_min = -0.5
# x_max = 1.5

# =============================================================================
# Core functions
# =============================================================================

def generate_data(N):
    x = np.linspace(x_min, x_max, N)
    t = np.linspace(0, t_max, N)
    #V = 0
    V = k / 2 * x**2
    dx = (x_max - x_min)/N
    dt = t_max/N
    b = 1j * dt / (2 * dx**2)
    a = - b / 2
    d = 1 + b + 1j * dt / 2 * V 
    psi_0 = np.sqrt(alpha/np.sqrt(np.pi))*np.exp(-alpha**2*(x-lamda)**2/2)
    #psi_0 = (2*np.pi*sigma**2)**(-1/4)*np.exp(1j*k_0*(x-lamda))*np.exp(-(x-lamda)**2/(2*sigma)**2)
    A = a*np.eye(N,N,k=-1) + d*np.eye(N,N,k=0) + a*np.eye(N,N,k=1)
    A_inv = np.linalg.inv(A)
    A_con = np.conjugate(A)
    psi = [psi_0]
    for i in range(len(t)-1):
        psi.append(A_inv @ A_con @ psi[i])
    psi = np.array(psi)
    psi_analytical = []
    ksi = alpha * x
    ksi_lamda = alpha * lamda
    for t_ in t:
        psi_analytical.append(np.sqrt(alpha/np.sqrt(np.pi))*np.exp(-0.5*(ksi-ksi_lamda*np.cos(omega*t_))**2-1j*(omega*t_/2+ksi*ksi_lamda*np.sin(omega*t_)-1/4*ksi_lamda**2*np.sin(2*omega*t_))))
        #psi_analytical.append((2*np.pi*sigma**2)**(-1/4)/(1+1j*t_/(2*sigma**2))**(1/2)*np.exp((-(x-lamda)**2/(2*sigma)**2+1j*k_0*(x-lamda)-1j*k_0**2*t_/2)/(1+1j*t_/(2*sigma**2))))
    np.save(f"data/gx_{N}.npy", x)
    np.save(f"data/gt_{N}.npy", t)
    np.save(f"data/gpsi_{N}.npy", psi)
    np.save(f"data/gpsi_a_{N}.npy", psi_analytical)
    return x, t, psi, psi_analytical

def load_data(N,r=0):
    if r != 0:
        x = np.load(f"data/x_{N}_{r}.npy")
        t = np.load(f"data/t_{N}_{r}.npy")
        psi = np.load(f"data/psi_{N}_{r}.npy")
        psi_analytical = np.load(f"data/psi_a_{N}_{r}.npy")
    else:    
        x = np.load(f"data/x_{N}.npy")
        t = np.load(f"data/t_{N}.npy")
        psi = np.load(f"data/psi_{N}.npy")
        psi_analytical = np.load(f"data/psi_a_{N}.npy")
    return x, t, psi, psi_analytical

def plot_psi(x, t, psi, value_type = "abs", title="", z_label="", filename = "something"):
    if value_type == "abs":
        psi = psi.real**2 + psi.imag**2
    elif value_type == "real":
        psi = psi.real
    elif value_type == "imag":
        psi = psi.imag
    x, t = np.meshgrid(x, t)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, t, psi, rstride=1, cstride=1, cmap='winter', edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel(z_label)
    ax.view_init(25, 40)
    plt.savefig(f"graphs/{filename}.pdf")
    plt.show()
    return

def pds(psi):
    return psi.real**2 + psi.imag**2

def error_vs_time():
    Ns = [300, 400, 500, 600, 700, 800, 1000]
    for N in Ns:
        print(N)
        x, t, psi, psi_analytical = load_data(N)
        error = np.max(np.abs(pds(psi)-pds(psi_analytical)), axis=1)
        plt.plot(t, error, label=f"N = {N}")
    plt.legend()
    plt.title("Maximum absolute error")
    plt.xlabel("t")
    plt.ylabel(r"max$|\psi - \psi_a|$")
    plt.savefig("graphs/error_vs_time.pdf")
    plt.show()
    return

def error_vs_N():
    Ns = [300, 400, 500, 600, 700, 800, 1000]
    error = []
    for N in Ns:
        print(N)
        x, t, psi, psi_analytical = load_data(N)
        error.append(np.max(np.abs(pds(psi)-pds(psi_analytical))))
    plt.plot(Ns, error)
    plt.title("Overall maximum absolute error")
    plt.xlabel("N")
    plt.ylabel(r"max$|\psi - \psi_a|$")
    plt.savefig("graphs/error_vs_N.pdf")
    plt.show()  

def peaks_over_time():
    Ns = [300, 400, 500, 600, 700, 800, 1000]
    for N in Ns:
        print(N)
        x, t, psi, psi_analytical = load_data(N)
        error = np.max(np.abs(pds(psi)), axis=1)
        plt.plot(t, error, label=f"N = {N}")
    plt.plot(t, np.max(pds(psi_analytical),axis=1), label=f"Analytical")
    plt.legend()
    plt.title("Peaks over time")
    plt.xlabel("t")
    plt.ylabel(r"max $\psi$")
    plt.savefig("graphs/peaks.pdf")
    plt.show()

def plot_state():
    Ns = [500]#, 400, 500, 600, 700, 800, 1000]
    for N in Ns:
        print(N)
        x, t, psi, psi_analytical = load_data(N)
        fig, ax = plt.subplots()
        im = ax.pcolormesh(x, t, np.abs(pds(psi)-pds(psi_analytical)), cmap=plt.get_cmap("jet"))
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im)
        ax.set_title("Absolute error, N = 500")
        plt.savefig("graphs/coherent_error.pdf")
        plt.show()

def gauss_plot():
    Ns = [500]#[200, 300, 400, 500]#, 600, 700, 800, 1000]
    for N in Ns:
        generate_data(N)
        #print(N)
        x, t, psi, psi_analytical = load_data(N)
        #plt.plot(x, pds(psi)[190])
        fig, ax = plt.subplots()
        im = ax.pcolormesh(x, t, np.abs(pds(psi)-pds(psi_analytical)), cmap=plt.get_cmap("jet"))
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im)
        ax.set_title("Absolute error, N = 300")
        plt.savefig("graphs/gauss_error.pdf")
        plt.show()

def generate_data_high(N, r):
    x = np.linspace(x_min, x_max, N)
    t = np.linspace(0, t_max, N)
    #V = 0
    V = k / 2 * x**2
    dx = (x_max - x_min)/N
    dt = t_max/N
    b = 1j * dt / (2 * dx**2)
    # d = 1 + 1j * dt / 2 * V - b / 2 * (-5/2)
    # a = - b / 2 * (4/3)
    # c = - b / 2 * (-1/12)
    psi_0 = np.sqrt(alpha/np.sqrt(np.pi))*np.exp(-alpha**2*(x-lamda)**2/2)
    #psi_0 = (2*np.pi*sigma**2)**(-1/4)*np.exp(1j*k_0*(x-lamda))*np.exp(-(x-lamda)**2/(2*sigma)**2)
    A = (1 + 1j * dt / 2 * V)*np.eye(N,N,k=0)
    #r = 4
    c_k = [[-2,1],[-5/2,4/3,-1/12],[-49/18,3/2,-3/20,1/90],[-205/72,8/5,-1/5,8/315,-1/560],[-5269/1800,5/3,-5/21,5/126,-5/1008,1/3150],[-5369/1800,12/7,-15/56,10/189,-1/112,2/1925,-1/16632]]
    for r_i in range(r+1):
        print(r_i)
        if r_i == 0:
            A = A+c_k[r-1][r_i]*np.eye(N,N,k=r_i)*(-b/2)
        else:
            A = A+c_k[r-1][r_i]*(np.eye(N,N,k=-r_i) + np.eye(N,N,k=r_i))*(-b/2)
    #A = c*np.eye(N,N,k=-2) + a*np.eye(N,N,k=-1) + d*np.eye(N,N,k=0) + a*np.eye(N,N,k=1) + c*np.eye(N,N,k=2)
    A_inv = np.linalg.inv(A)
    A_con = np.conjugate(A)
    psi = [psi_0]
    for i in range(len(t)-1):
        psi.append(A_inv @ A_con @ psi[i])
    psi = np.array(psi)
    psi_analytical = []
    ksi = alpha * x
    ksi_lamda = alpha * lamda
    for t_ in t:
        psi_analytical.append(np.sqrt(alpha/np.sqrt(np.pi))*np.exp(-0.5*(ksi-ksi_lamda*np.cos(omega*t_))**2-1j*(omega*t_/2+ksi*ksi_lamda*np.sin(omega*t_)-1/4*ksi_lamda**2*np.sin(2*omega*t_))))
        #psi_analytical.append((2*np.pi*sigma**2)**(-1/4)/(1+1j*t_/(2*sigma**2))**(1/2)*np.exp((-(x-lamda)**2/(2*sigma)**2+1j*k_0*(x-lamda)-1j*k_0**2*t_/2)/(1+1j*t_/(2*sigma**2))))
    np.save(f"data/x_{N}_{r}.npy", x)
    np.save(f"data/t_{N}_{r}.npy", t)
    np.save(f"data/psi_{N}_{r}.npy", psi)
    np.save(f"data/psi_a_{N}_{r}.npy", psi_analytical)
    return x, t, psi, psi_analytical

def higher():
    Ns = [500]#[200, 300, 400, 500]#, 600, 700, 800, 1000]
    for N in Ns:
        generate_data_high(N)
        x, t, psi, psi_analytical = load_data(N)
        fig, ax = plt.subplots()
        im = ax.pcolormesh(x, t, np.abs(pds(psi)-pds(psi_analytical)), cmap=plt.get_cmap("jet"))    #-pds(psi_analytical)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im)
        ax.set_title("Absolute error, r = 4, N = 500")
        plt.savefig("graphs/higher_r4.pdf")
        plt.show()


def higher_analysis():
    Ns = [200]#[200, 300, 400, 500]#, 600, 700, 800, 1000]
    error = []
    for N in Ns:
        for r in range(1,7):
            generate_data_high(N,r)
            x, t, psi, psi_analytical = load_data(N,r)
            error.append(np.max(np.abs(pds(psi)-pds(psi_analytical))))
    plt.plot(np.arange(1,7), error)
    plt.xlabel("r")
    plt.ylabel(r"$|\psi - \psi_a|$")
    plt.title("Absolute error, coherent state, N = 200")
    plt.savefig("graphs/error_higher.pdf")
    plt.show()

def time_complexity():
    Ns = np.arange(10, 500, 40)
    t = []
    for N in Ns:
        time1 = time.time()
        generate_data(N)
        dt = time.time() - time1
        t.append(dt)
    plt.plot(Ns, t)
    plt.xlabel("N")
    plt.ylabel("t [s]")
    plt.title("Time complexity")
    plt.savefig("graphs/time_vs_N.pdf")
    plt.show()

if __name__ == "__main__":
    Ns = np.arange(10, 500, 40)
    t = []
    for N in Ns:
        time1 = time.time()
        generate_data(N)
        dt = time.time() - time1
        t.append(dt)
    plt.plot(Ns, t)
    plt.xlabel("N")
    plt.ylabel("t [s]")
    plt.title("Time complexity")
    plt.savefig("graphs/time_vs_N.pdf")
    plt.show()

    
