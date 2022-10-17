import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import time
import scipy.fft as sc
from numpy import fft

a = 1
sigma = 0.1
t_max = 1
D = 0.5

def plot_T(x, t, T, title="", z_label="T", filename = "something"):
    x, t = np.meshgrid(x, t)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, t, T, rstride=1, cstride=1, cmap='winter', edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel(z_label)
    ax.view_init(25, 40)
    plt.savefig(f"graphs/{filename}.pdf")
    plt.show()
    return


def gauss(x):
    return np.exp(-(x-a/2)**2/sigma**2)

def square(x):
    out = np.zeros(len(x))
    out += (x>a/4)*(x<3*a/4)
    return out

def delta(x):
    out = np.zeros(len(x))
    out[len(x)//2] = 1
    return out


def fourier_solve(x_0, t, parameters, Dirichlet = False):
    t1 = time.time()
    N, a, t_max, h_x, h_t = parameters
    if Dirichlet:
        x_0 = np.concatenate((-x_0, x_0))
        t = np.linspace(0, 2*t_max, 2*N)
        N = 2*N
    T = np.zeros((N, N), dtype=complex)
    fft_T0 = fft.fft(x_0)
    f_k = fft.fftfreq(N, h_x)
    for i in range(N):
        T[:, i] = np.exp(-4. * D * np.pi**2. * f_k[i]**2. * t) * fft_T0[i]
    for j in range(N):
        T[j, :] = fft.ifft(T[j, :])
    if Dirichlet:
        T = T[:N//2,N//2:]
    T = T.real
    dt = time.time() - t1
    return T, dt


def B_k(delta, k, x0):
    if(x0 <= delta*(k-2)):
        return 0.
    elif(x0 <= delta*(k-1)):
        return (x0 - (k-2)*delta)**3/delta**3
    elif(x0 <= delta*k):
        return (x0 - (k-2)*delta)**3/delta**3 - 4*(x0 - (k-1)*delta)**3/delta**3
    elif(x0 <= delta*(k+1)):
        return -(x0 - (k+2)*delta)**3/delta**3 + 4*(x0 - (k+1)*delta)**3/delta**3
    elif(x0 <= delta*(k+2)):
        return -(x0 - delta*(k+2))**3/delta**3
    else:
        return 0

def spline_solve(x_0, t, x, parameters):
    t1 = time.time()
    N, a, t_max, h_x, h_t = parameters
    dx = a/N
    dt = t_max/N
    A = np.diag(4*np.ones(N), 0) + np.diag(np.ones(N-1), -1) + np.diag(np.ones(N-1), 1)
    B = 6*D/dx**2*(np.diag(-2*np.ones(N), 0) + np.diag(np.ones(N-1), -1) + np.diag(np.ones(N-1), 1))
    C = np.linalg.inv(A-dt/2*B)@(A+dt/2*B)
    c0 = np.linalg.solve(A, x_0)
    T = []
    for t_ in t:
        if t_ == 0:
            c = c0
        c = C@c
        T_xs = []
        for xi in range(len(x)):
            T_x = 0
            for k in range(xi-1,N+2):
                if k == -1:
                    c_k = -c[1]
                elif k == 0 or k == N:
                    c_k = 0
                elif k == N+1:
                    c_k = -c[N-1]
                else:
                    c_k = c[k]
                T_x += c_k*B_k(dx, k, x[xi])
            T_xs.append(T_x)
        T.append(T_xs)
    T = np.array(T)
    T = np.concatenate((np.array([x_0]), T[:-1]))
    dt = time.time() - t1
    return T, dt