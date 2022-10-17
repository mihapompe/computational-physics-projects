# =============================================================================
# Eigen values and eigen vectors
# Author: Miha Pompe
# =============================================================================

import time
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import special as sp

# =============================================================================
# Core functions
# =============================================================================


def projection(A):
    A = A.copy()
    n = len(A)
    c_i = A[:, 0]
    e_i = np.zeros(n)
    e_i[0] = 1
    c_i_norm = np.linalg.norm(c_i)
    u = c_i - c_i_norm*e_i
    u_norm = np.linalg.norm(u)
    Q = np.eye(n) - 2*np.outer(u, u)/np.dot(u, u)
    return Q

def Householder_projection(A):
    # Returns Q R = A
    n = len(A)
    Q_s = np.eye(n)
    A_ = A.copy()
    for i in range(n-1):
        Q = projection(A_)
        Q_big = np.eye(n)
        Q_big[i:, i:] = Q.copy()
        Q_s = np.copy(Q_big) @ Q_s
        A_ = Q_s @ A
        A_ = A_[i+1:, i+1:]
    return Q_s.T, Q_s@A

def projection_tri_diag(A):
    A = A.copy()
    n = len(A)
    d_i = A[1:, 0]
    d_i_norm = np.linalg.norm(d_i)
    e_i = np.zeros(n-1)
    e_i[0] = 1
    u = d_i - d_i_norm*e_i
    u_norm = np.linalg.norm(u)
    Q = np.eye(n-1) - 2*np.outer(u, u)/u_norm**2
    #Q = np.eye(n-1) - 2*np.outer(d_i, d_i)/d_i_norm**2
    return Q

def Householder_tri_diag(A):
    n = len(A)
    Q_s = np.eye(n)
    Q_s_inv = np.eye(n)
    A_ = A.copy()
    for i in range(n-2):
        Q = projection_tri_diag(A_)
        Q_big = np.eye(n)
        Q_big[i+1:, i+1:] = Q.copy()
        Q_s = np.copy(Q_big) @ Q_s
        Q_s_inv = Q_s_inv @ np.copy(Q_big)
        A_ = Q_s @ A @ Q_s_inv
        A_ = A_[i+1:, i+1:]
    return Q_s @ A @ Q_s_inv, Q_s_inv

def Jacobi(A):
    n = len(A)
    A_ = A.copy()
    Z = np.eye(n)
    for i in range(10):
        for p in range(n-1):
            for q in range(n-1-p):
                theta = (A_[q,q]-A_[p,p])/2*A_[p,q]
                t = (-1 + 2*(theta > 0)/(abs(theta)+np.sqrt(theta**2+1)))
                c = 1/np.sqrt(t**2+1)
                s = t*c
                P = np.eye(n)
                P[p,p] = c
                P[q,q] = c
                P[p,q] = s
                P[q,p] = -s
                Z = Z @ P
                A_ = P.T @ A_ @ P
    return A_, Z

def QR_iteration(A, Z_old, method="Householder"):
    Z = np.eye(len(A))
    A_ = A.copy()
    for i in range(100):
        if method == "Householder":
            Q, R = Householder_projection(A_)
        elif method == "Numpy":
            Q, R = numpy_QR(A_)
        Z = Z @ Q
        A_ = R.copy() @ Q.copy()
    Z = Z_old @ Z
    Z_inv = Z@ Z_old
    return np.diag(A_), Z

def numpy_QR(A):
    return np.linalg.qr(A)

def numpy_diagonalize(A, n_max= None):
    if n_max == None:
        n_max = len(A)
    w, v = np.linalg.eig(A)
    id = w.argsort()
    w = w[id]
    v = v[id]
    return w[:n_max], v[:n_max]

def matrix_element_q(i, j):
    return np.sqrt(i+j+1)/2*(np.abs(j-i) == 1)

def matrix_element_q2(i, j):
    return (np.sqrt(j*(j-1))*(i == j-2) + (2*j+1)*(i == j) + np.sqrt((j+1)*(j+2))*(i == j+2))/2

def divide_factorial(i, j):
    if i == j:
        return 1
    elif i > j:
        return np.prod(np.linspace(j+1, i, int(i-j)))
    else:
        return 1/np.prod(np.linspace(i+1, j, int(j-i)))

def matrix_element_q4(i, j):
    a = (i == j+4) + 4*(2*j+3)*(i == (j+2)) + 12*(2*j**2+2*j+1)*(i == j) + 16*j*(2*j**2-3*j+1)*(i == j-2) + 16*j*(j**3-6*j**2+11*j-6)*(i == j-4)
    if a == 0:
        return 0
    b = divide_factorial(i, j)
    return a*2**((i-j-8)/2)*np.sqrt(b)

def Hamiltonian_matrix(N, lamda, type="q4"):
    # N - matrix size
    # H = H_0 + lambda q^4
    i, j = np.array(np.meshgrid(np.linspace(1, N, N), np.linspace(1, N, N)))
    i = i.reshape(N**2)
    j = j.reshape(N**2)
    E0 = np.linspace(0, N, N)+1/2
    if type == "q4":
        mat = []
        for t in range(len(i)):
            mat.append(matrix_element_q4(i[t], j[t]))
        mat = np.array(mat)
        mat = mat.reshape(N, N)
    elif type == "q2":
        mat = matrix_element_q2(i, j)
        mat = mat.reshape(N, N)
        mat = mat @ mat
    elif type == "q":
        mat = matrix_element_q(i, j)
        mat = mat.reshape(N, N)
        mat = mat @ mat @ mat @ mat
    return lamda * mat + E0*np.eye(N)

def Hamiltonian_matrix_2(N):
    i, j = np.array(np.meshgrid(np.linspace(1, N, N), np.linspace(1, N, N)))
    i = i.reshape(N**2)
    j = j.reshape(N**2)
    mat = matrix_element_q2(i, j)
    mat = mat.reshape(N, N)
    return -5/2*mat + mat @ mat /10 + np.eye(N)

def base_functions(x, n):
    return 1/np.sqrt(2**n*sp.factorial(n)*np.sqrt(np.pi))*np.exp(-x**2/2)*sp.hermite(n)(x)

# =============================================================================
# Visualization
# =============================================================================

def draw_H_0_base():
    x = np.linspace(-10, 10, 500)
    for i in range(10):
        y = base_functions(x, i)+i+1/2
        plt.plot(x, y)
    x = np.linspace(-5, 5, 100)
    plt.plot(x, x**2/2)
    plt.show()
    return

def draw_state(eigen_values, eigen_vectors, lamda):
    x = np.linspace(-5, 5, 500)
    for i, eigen in enumerate(eigen_values):
        y = np.zeros(500)
        for j in range(len(eigen_vectors[i])):
            y += base_functions(x, j)*eigen_vectors[i][j]
        plt.plot(x, y+eigen*(i + 1/2))
    x = np.linspace(-5, 5, 500)
    plt.plot(x, x**2/2+lamda*x**4)
    plt.show()
    return

def plot_matrix(A):
    plt.matshow(A)
    plt.colorbar()
    plt.show()

def plot_Hamiltonian_matrix():
    N = 30
    method = ["q4", "q2", "q"]
    fig, ax = plt.subplots(1,3, figsize=(10,5))
    for lamda in range(3):
        H = Hamiltonian_matrix(N, 0.5, method[lamda])
        ax[lamda].matshow(H)
        ax[lamda].set_ylabel("$i$")
        ax[lamda].set_xlabel("$j$")
        ax[lamda].set_title(f"Metoda {method[lamda]}")
        #ax[lamda].colorbar()
    plt.suptitle("Hamiltonova matrika\nVelikost matrike $N = 30, \lambda = 0.5$")
    plt.savefig("matrika.pdf")
    plt.show()
    return


def accuracy_of_diagonalization():
    return

def eigen_values_vs_lambda():
    N = 20
    lam = np.linspace(0, 1, 20)
    eig = []
    eig_max = []
    x = []
    fig, ax = plt.subplots(2, sharex=True)
    fig.suptitle("Lastne vrednosti matrike velikosti 20 v odvisnosti od $\lambda$\n in relativna napaka izračuna")
    for lamda in lam:
        H = Hamiltonian_matrix(N, lamda)
        H_tri, Z_old = Householder_tri_diag(H)
        eig_h, Z_h = QR_iteration(H_tri, Z_old, "Householder")
        eig_nq, Z_nq = QR_iteration(H_tri, Z_old, "Numpy")
        eig_n, Z_n = numpy_diagonalize(H)
        Z_n = Z_n.T
        eig.append([eig_h, eig_nq, eig_n])
        eig_max.append([np.max(eig_h), np.max(eig_nq), np.max(eig_n)])
        x.append(lamda*np.ones(len(eig_h)))
    eig = np.array(eig)
    eig_max = np.array(eig_max)
    x = np.array(x).flatten()
    ax[0].scatter(x, eig[:,0], label="Householder")
    ax[0].scatter(x, eig[:,1], label="Numpy QR")
    ax[0].scatter(x, eig[:,2], label="Numpy diag")
    ax[0].set_ylabel("$E$")
    ax[0].legend()
    
    x = np.linspace(0, 1, 20)
    ax[1].plot(x, np.abs(eig_max[:,2]- eig_max[:,0])/eig_max[:,2], label="Householder napaka")
    ax[1].plot(x, np.abs(eig_max[:,2]- eig_max[:,1])/eig_max[:,2], label="Numpy tri diag. napaka")
    ax[1].legend()
    ax[1].set_ylabel("$\Delta E$")
    ax[1].set_xlabel("$\lambda$")
    ax[1].set_yscale("log")

    plt.savefig("eigen_vs_lambda.pdf")
    plt.show()
    return

def eigen_values_vs_N():
    N_all = np.linspace(100, 500, 5)
    eig = []
    eig_max = []
    x = []
    lamda = 0.5

    plt.title("Lastne vrednosti pri različnih velikostih matrik")
    for N in N_all:
        print(f"Starting N = {int(N)}")
        H = Hamiltonian_matrix(int(N), lamda)
        H_tri, Z_old = Householder_tri_diag(H)
        eig_h, Z_h = QR_iteration(H_tri, Z_old, "Householder")
        eig_nq, Z_nq = QR_iteration(H_tri, Z_old, "Numpy")
        eig_n, Z_n = numpy_diagonalize(H)
        Z_n = Z_n.T
        eig.append([eig_h, eig_nq, eig_n])
        eig_max.append([np.max(eig_h), np.max(eig_nq), np.max(eig_n)])
        x.append(int(N)*np.ones(len(eig_h)))
    eig = np.array(eig)
    eig_max = np.array(eig_max)
    x = np.array(x)
    #print(x)
    eig2 = []
    eig2_max = []
    x = np.hstack(x)
    #print(np.hstack(eig[:,0]))
    eig2.append(np.hstack(eig[:,0]))
    eig2.append(np.hstack(eig[:,1]))
    eig2.append(np.hstack(eig[:,2]))
    eig2_max.append(np.hstack(eig_max[:,0]))
    eig2_max.append(np.hstack(eig_max[:,1]))
    eig2_max.append(np.hstack(eig_max[:,2]))
    for i, t in enumerate(eig):
        plt.plot(np.sort(np.flip(t[0])), label=f"N = {N_all[i]}")
    plt.ylabel("$E_n [\hbar\omega]$")
    plt.xlabel("$n$")
    plt.legend()
    plt.savefig("eigen_vs_N.pdf")
    plt.show()
    
    plt.title("Relativna napaka lastnih vrednosti")
    plt.plot(N_all, np.abs(eig2_max[2]- eig2_max[0])/eig2_max[2], label="Householder napaka")
    plt.plot(N_all, np.abs(eig2_max[2]- eig2_max[1])/eig2_max[2], label="Numpy tri diag. napaka")
    plt.legend()
    plt.ylabel("$\Delta E$")
    plt.xlabel("$N$")
    plt.yscale("log")

    plt.savefig("eigen_vs_N2.pdf")
    plt.show()
    return

def time_vs_N():
    N_all = np.linspace(1, 500, 20)
    lamda = 0.5
    times = []
    for N in N_all:
        print(f"Starting N = {int(N)}")
        time1 = time.time()
        H = Hamiltonian_matrix(int(N), lamda)
        time2 = time.time()
        dt1 = time2-time1
        H_tri, Z_old = Householder_tri_diag(H)
        time3 = time.time()
        dt2 = time3 - time2
        QR_iteration(H_tri, Z_old, "Householder")
        time4 = time.time()
        dt3 = time4 - time3
        QR_iteration(H_tri, Z_old, "Numpy")
        time5 = time.time()
        dt4 = time5 - time4
        numpy_diagonalize(H)
        time6 = time.time()
        dt5 = time6 - time5
        print(dt5)
        times.append([dt2+dt3, dt2+dt4, dt5])
    times = np.array(times).T

    title = ["Householder", "Numpy QR", "Numpy diag"]
    for i, t in enumerate(times):
        plt.plot(N_all, t, label=title[i])
    plt.title("Čas diagonalizacije")
    plt.ylabel("$t [s]$")
    plt.yscale("log")
    plt.xlabel("$N$")
    plt.legend()
    plt.savefig("N_vs_time.pdf")
    plt.show()

def q_n_vs_N():
    q_n = ["q4", "q2", "q"]
    titles = ["$[q_{ij}^4]$", "$[q_{ij}^2]^2$", "$[q_{ij}]^4$"]
    ts = []
    n = np.linspace(3, 101, 98)
    for j, q in enumerate(q_n):
        t = []
        print(q)
        for N in range(3, 101):
            time1 = time.time()
            H = Hamiltonian_matrix(N, 0.5, q)
            time2 = time.time()
            #eig, Z = numpy_diagonalize(H)
            
            dt = time2-time1
            t.append(dt)
        ts.append(t)
        plt.plot(n, t, label=titles[j])
    plt.legend()
    plt.xlabel("$N$")
    plt.yscale("log")
    plt.ylabel("t [s]")
    plt.title("Časovna zahtevnost računanja perturbacijskih matrik")
    plt.savefig("q_vs_N.pdf")
    plt.show()

def H2_vs_H():
    N = 100
    plt.figure(figsize=(10,5))
    H = Hamiltonian_matrix(N, 0.5, "q2")
    H2 = Hamiltonian_matrix_2(N)
    eig1, Z = numpy_diagonalize(H)
    eig2, Z = numpy_diagonalize(H2)

    plt.subplot(121)
    plt.plot(np.sort(eig1), label="$H$") 
    plt.plot(np.sort(eig2), label="$H_2$")
    plt.xlabel("$n$")
    plt.ylabel("$E [\hbar\omega]$")  
    x = np.linspace(0, N-1, N-1)
    plt.plot(x, x, label="$H_0$")
    plt.legend()

    plt.subplot(122)
    n = 20
    plt.plot(np.sort(eig1)[:n], label="$H$") 
    plt.plot(np.sort(eig2)[:n], label="$H_2$")
    plt.xlabel("$n$")
    plt.ylabel("$E [\hbar\omega]$")  
    x = np.linspace(0, N-1, N-1)
    plt.plot(x[:n], x[:n], label="$H_0$")
    plt.legend()


    plt.suptitle("Lastne energije različnih potencialov\n$\lambda = 0.5, N = 100$")
    plt.savefig("H2_H.pdf")
    plt.show()

# Time vs N, compare different diagonalization techniques Done
# Absolute error of implemented diagonalization methods vs N    Done

def draw_H_states():
    N = 4
    plt.figure(figsize=(6,11))
    H = Hamiltonian_matrix(N, 0.5, "q2")
    H2 = Hamiltonian_matrix_2(N)
    eig1, Z1 = numpy_diagonalize(H, 5)
    eig2, Z2 = numpy_diagonalize(H2, 5)

    plt.subplot(311)
    x = np.linspace(-5, 5, 500)
    for i in range(5):
        y = base_functions(x, i)+(i+1/2)
        plt.plot(x, y)
        y = x**2/2
    
    plt.plot(x[y < 6], y[y<6], label="Potencial $V = q^2/2$", c="black")
    plt.legend()
    plt.xlabel("$q$")
    plt.ylabel("$E [\hbar\omega]$")



    lamda = 0.5
    plt.subplot(312)
    
    for i, eigen in enumerate(eig1):
        y = np.zeros(500)
        for j in range(len(Z1[i])):
            y += base_functions(x, j)*Z1[i][j]
        if np.all(y+eigen*(i + 1/2) < 50):
            plt.plot(x, y+eigen+(i + 1/2))
    x = np.linspace(-5, 5, 500)
    y = x**2/2+lamda*x**4
    plt.plot(x[y < 15], y[y<15], label="Potencial $V = q^2/2 + \lambda q^4$", c="black")
    plt.legend()
    plt.xlabel("$q$")
    plt.ylabel("$E [\hbar\omega]$")
    
    plt.subplot(313)
    for i, eigen in enumerate(eig2):
        y = np.zeros(500)
        for j in range(len(Z2[i])):
            y += base_functions(x, j)*Z2[i][j]
        plt.plot(x, y+eigen+(i + 1/2))
    x = np.linspace(-5, 5, 500)
    y = -2*x**2+x**4/10
    plt.plot(x[y < 10], y[y<10], label="Potencial $V = -2q^2 + q^4/10$", c="black")
    plt.legend()
    plt.xlabel("$q$")
    plt.ylabel("$E [\hbar\omega]$")

    plt.suptitle("Lastne funkcije različnih potencialov\n$\lambda = 0.5, N = 5$")
    plt.savefig("lastne_f_vs_potenciali.pdf")
    plt.show()
    return


# Draw states

if __name__ == "__main__":

    ##### Exercises #####
    #plot_Hamiltonian_matrix()
    #eigen_values_vs_lambda()
    #eigen_values_vs_N()
    #time_vs_N()
    #q_n_vs_N()
    #H2_vs_H()
    draw_H_states()
    # H4 = Hamiltonian_matrix(100, 0.5, "q4")
    # H2 = Hamiltonian_matrix(100, 0.5, "q2")
    # H1 = Hamiltonian_matrix(100, 0.5, "q")
    # print(np.sum(np.abs(H4)))
    # print(np.sum(np.abs(H4-H2))/np.sum(np.abs(H4)))
    # print(np.sum(np.abs(H4-H1))/np.sum(np.abs(H4)))

    # H = Hamiltonian_matrix(10, 0.00000001)
    # H, Z_old = Householder_tri_diag(H)
    # eig, Z = QR_iteration(H, Z_old, "Householder")   
    # draw_state(eig, Z, 0.00000001)


    # H = Hamiltonian_matrix(10, 0.0001)
    # H, Z_old = Householder_tri_diag(H)
    # eig, Z = QR_iteration(H, Z_old, "Householder")
    # print(eig)
    # draw_state(eig, Z.T, 0)
    # H = Hamiltonian_matrix(10, 0.0001)
    # eig, Z = numpy_diagonalize(H)
    # print(eig)
    # draw_state(eig, Z.T, 0)