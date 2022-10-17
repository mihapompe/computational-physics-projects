from bvp import fd, shoot
import numpy as np
import matplotlib.pyplot as plt
from diffeq import rk4
import time
from matplotlib import cm
import scipy as sc

# Infinite potential well


def f(x, t):
    return np.array([x[1], -10*x[0]])

def exercise_1(f, a, b, t):
    N = 300
    bs = []
    ks = np.linspace(0.5, 20, N)
    for k in ks:
        y = rk4(lambda x, t: np.array([x[1], -k**2*x[0]]), [0.0, 1.0], t)
        #plt.plot(t, y)
        bs.append(y[-1, 0])
    bs = np.array(bs)
    #plt.show()
    
    # Find roots
    roots = []
    for i in range(1, N-1):
        if np.abs(bs[i-1]) > np.abs(bs[i]) < np.abs(bs[i+1]):
            roots.append([ks[i], bs[i]])
    roots = np.array(roots).T
    print(roots[0])
    plt.xlabel("k")
    plt.ylabel(r"$\psi(b)$")
    plt.title("Roots of the shooting method")
    plt.plot(ks, bs)
    plt.scatter(roots[0], roots[1], c = "red")
    plt.savefig("graphs/roots.pdf")
    plt.show()


def exercise_2():
    errors = []
    for i, k in enumerate(k_roots[:-2]):
        x = rk4(lambda x, t: np.array([x[1], -k**2*x[0]]), [0.0, 1.0], t)
        x = x[:,0]/max(x[:,0])
        analytic = np.sin(k*t)
        analytic = analytic/max(analytic)
        errors.append(np.abs(analytic-x))
        #x_norm = x[:,0]/(np.sum(x[:,0]))
        E = int(round(k_roots[i]/k_roots[0],0))**2
        plt.plot(t, x+E, label= r"$\psi$"+str(i+1)+r", $E = $"+str(E)+r" $E_1$")
        plt.fill_between(t, E, x+E)
    potx = np.linspace(0, 1, 1000)
    pot = np.zeros(1000)
    pot[0] = 17
    pot[-1] = 17
    plt.plot(potx, pot, label="Potential", c= "black")
    plt.title("Infinite potential well - shooting method")
    plt.ylabel("$E [E_1]$")
    plt.xlabel("x")
    plt.legend()
    plt.savefig("graphs/shoot_infinite.pdf")
    plt.show()

    for i, k in enumerate(k_roots[:-2]):    
        plt.plot(t, errors[i],label=r"$\psi$"+str(i+1))
    plt.xlabel("x")
    plt.ylabel(r"$|\psi - \psi_0|$")
    plt.title("Absolute error")
    plt.legend()
    plt.savefig("graphs/error_shoot_infinite.pdf")
    plt.show()
    return

def exercise_3():
    bs = []
    times = []
    Ns = np.linspace(2, 1000, 200)
    for N in Ns:
        t = np.linspace(0, 1, int(N))
        t1 = time.time()
        x = rk4(lambda x, t: np.array([x[1], -np.pi**2*x[0]]), [0.0, 1.0], t)
        dt = time.time() - t1
        times.append(dt)
        bs.append(x[-1,0])
    plt.plot(Ns, np.abs(bs))
    plt.yscale("log")
    plt.xlabel("N")
    plt.ylabel(r"$\psi(b)$")
    plt.title("Global error in relation to number of steps")
    plt.savefig("graphs/error_N.pdf")
    #plt.show()
    plt.clf()

    plt.plot(Ns, times)
    plt.ylabel("time [s]")
    plt.xlabel("N")
    plt.title("Time complexity of the shooting method")
    plt.savefig("graphs/time_complexity_shoot.pdf")
    plt.show()
    return

def generate_Z(N, h, V = None):
    if V is None:
        V = np.zeros(N-2)
    
    Z = np.zeros((N-2,N-2))
    for i in range(N-2):
        if i == 0:
            Z[i,i+1] = 1
        elif i == N-3:
            Z[i,i-1] = 1
        else:
            Z[i,i+1] = 1
            Z[i,i-1] = 1
        Z[i,i] = -2 - h**2*V[i]
    return -Z/h**2

def finite_differences(t, V = None):
    h = np.abs(t[0]-t[1])
    N = len(t)
    Z = generate_Z(N, h, V)
    #eigenValues, eigenVectors = np.linalg.eig(Z)
    eigenValues, eigenVectors = np.linalg.eigh(Z)
    eigenVectors = eigenVectors.T
    #idx = eigenValues.argsort()[::-1]   
    #eigenValues = eigenValues[idx]
    eigenValues = eigenValues/eigenValues[0]
    #eigenVectors = eigenVectors[:,idx]
    #eigenVectors = eigenVectors[::-1]
    eig_vectors = []
    for vec in eigenVectors:
        avec = np.abs(vec)
        direction = 1
        eigvec = []
        for i in range(1, N-3):
            if avec[i-1] > avec[i] and avec[i] < avec[i+1] and vec[i] < 0:
                direction *= -1
            eigvec.append(avec[i]*direction)
            if avec[i-1] > avec[i] and avec[i] < avec[i+1] and vec[i] > 0:
                direction *= -1
        eig_vectors.append(eigvec/max(eigvec))
    return eig_vectors, eigenValues

def exercise_4():
    N = 205
    eig_vectors, eigenValues = finite_differences(t)
    plt.figure(figsize=(12,5))
    for i in range(4):
        sol = eig_vectors[i]+eigenValues[i]
        plt.subplot(121)
        plt.plot(t[2:-2], sol, label=r"$\psi$"+str(i+1))
        plt.fill_between(t[2:-2], eigenValues[i], sol)
        analytic = np.sin(np.pi*(i+1)*t)
        analytic = analytic/max(analytic)
        #plt.plot(t[2:-2], analytic[2:-2]+eigenValues[i])
        error = np.abs(analytic[2:-2]-eig_vectors[i])
        plt.subplot(122)
        plt.plot(t[2:-2], error)
        plt.yscale("log")
    plt.subplot(121)
    potx = np.linspace(0, 1, 1000)
    pot = np.zeros(1000)
    pot[0] = 17
    pot[-1] = 17
    plt.plot(potx, pot, label="Potential", c= "black")
    plt.title("Infinite potential well - finite differences method")
    plt.ylabel("$E [E_1]$")
    plt.xlabel("x")
    plt.legend()

    plt.subplot(122)
    plt.title("Absolute difference")
    plt.xlabel("x")
    plt.ylabel(r"$|\psi - \psi_0|$")
    plt.savefig("graphs/differences_infinite.pdf")
    plt.show()

def exercise_5():
    times = []
    Ns = np.linspace(10,2000,50)
    plt.figure(figsize=(8,5))
    for N in Ns:
        print(N/20)
        t = np.linspace(0, 1, int(N))
        t1 = time.time()
        _, values = finite_differences(t)
        dt = time.time()- t1
        times.append(dt)
        if N > 500:
            values = values[:500]
        plt.plot(values, c=cm.winter(N/2000))
    x = np.arange(500)
    plt.plot(x**2, c="red")
    plt.xlabel("n")
    plt.ylabel("$n^{th}$ eigenvalue")
    plt.title("Eigen values of finite differences method")
    plt.colorbar(cm.ScalarMappable(norm=cm.colors.Normalize(10, 2000), cmap=cm.winter), label = r"$N$", location="right")
    plt.savefig("graphs/eigen_diff.pdf")
    plt.show()

    plt.plot(Ns, times)
    plt.ylabel("t [s]")
    plt.xlabel("N")
    plt.title("Time complexity of finite differences method")
    plt.savefig("graphs/time_diff.pdf")
    plt.show()


k_roots = np.arange(1, 7)*np.pi#[ 3.17391304,  6.30434783,  9.43478261, 12.56521739, 15.69565217, 18.82608696]
N = 100
t = np.linspace(0, 1, N)
a = 0.0
b = 0.0

# V = np.zeros(N)
# V[3*N//4:] = 100
# V[:N//4] = 100
# vec, values = finite_differences(t, V)
# print(values)
# plt.plot(t[2:-2], vec[1])
# plt.show()





#exercise_1(f, a, b, t)
#exercise_2()
#exercise_3()
#exercise_4()
#exercise_5()

# Finite potential well

def V(t):
    if -0.5 < t < 0.5:
        return 0
    return 20

def V2(t):
    return 30*t**2

def V3(t):
    return -1/t


def exercise_6(t, V):
    N = 300
    bs = []
    ks = np.linspace(0, 30, N)
    
    for E in ks:
        def f(x, t):
            return np.array([x[1], (-E+V(t))*x[0]])
        # y = rk4(f, [0.0, 0.001], t)
        # y = y[:,0]/max(y[:,0])
        y = shoot(f, 0, 0, 0.01, 0.1, t, 1e-3)
        y = y/max(np.abs(y))
        # plt.plot(t, y)
        bs.append(y[-1])
    bs = np.array(bs)
    # plt.show()
    
    # Find roots
    roots = []
    for i in range(1, N-1):
        if np.abs(bs[i-1]) > np.abs(bs[i]) < np.abs(bs[i+1]):
            roots.append([ks[i], bs[i]])
    roots = np.array(roots).T
    print(roots[0])
    # plt.xlabel("k")
    # plt.ylabel(r"$\psi(b)$")
    # plt.title("Roots of the shooting method")
    # plt.plot(ks, bs)
    # plt.scatter(roots[0], roots[1], c = "red")
    # plt.savefig("graphs/roots_finite.pdf")
    # plt.show()
    return roots[0]

#Es = [ 20.96655518,  70.88294314, 124.79264214, 186.02341137]

def exercise_7(Es, V):
    errors = []
    for i, E in enumerate(Es):
        def f(x, t):
            return np.array([x[1], (-E+V(t))*x[0]])
        # x = rk4(f, [0.0, 0.001], t)#shoot(f, 1, 0, 1.0, 2.0, t, 1e-3)
        # x = x[:,0]/max(x[:,0])
        x = shoot(f, 0, 0, 0.01, 0.1, t, 1e-3)
        x = x/max(np.abs(x))
        # analytic = np.sin(k*t)
        # analytic = analytic/max(analytic)
        # errors.append(np.abs(analytic-x))
        #x_norm = x[:,0]/(np.sum(x[:,0]))
        #E_ = int(round(Es[i]/Es[0],0))**2
        E_ = Es[i]
        plt.plot(t, x+E_, label= r"$\psi$"+str(i+1)+r", $E = $"+str(round(E_,1)))
        plt.fill_between(t, E_, x+E_)
    potx = np.linspace(-1, 1, 1000)
    pot = []
    for x in potx:
        pot.append(V(x))
    plt.plot(potx, pot, label="Potential", c= "black")
    plt.title("Finite potential well - shooting method")
    plt.ylabel("$E$")
    plt.xlabel("x")
    plt.legend()
    plt.savefig("graphs/shoot_finite.pdf")
    plt.show()


t = np.linspace(-1, 1, N)
#Es = exercise_6(t, V)
#print(Es)
#Es = 20.0-np.array([19.08, 16.36])
#exercise_7(Es, V)

#x = shoot()

Vs = []
for i in t:
    Vs.append(V(i))


vec, val = finite_differences(t, Vs)
plt.plot(vec[5])
plt.show()

