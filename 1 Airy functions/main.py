# =============================================================================
# Izračun Airyjevih funkcij
# Avtor: Miha Pompe
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc
import scipy.optimize as op
divide_neg, divide_poz = -6, 7
size = 1

# =============================================================================
# Aproksimacija okoli izhodišča
# - Uporaba knjižnice Numpy nam omogoča večjo natančnost. V kolikor le-te ne 
#   uporabljamo le težko dosežemo napako 1e-10. 
# =============================================================================

def Airy_m(x, n):
    alpha = 0.355028053887817239
    betta = 0.258819403792806798
    k = np.linspace(1, n, n)
    f = 1 + np.sum(np.cumprod((x**3)/((3*k-1)*3*k)))
    g = (1+np.sum(np.cumprod((x**3)/((3*k+1)*3*k))))*x
    Ai = alpha*f-betta*g
    Bi = np.sqrt(3)*(alpha*f+betta*g)
    return [Ai, Bi]

def minimize_absolute_error(x, f, err=1e-10, k=1):
    Ai, _, Bi, _ = sc.airy(x)
    Ai_m, Bi_m = 1, 1
    while np.abs(Ai_m-Ai) > err or np.abs(Bi_m-Bi) > err:
        k += 1
        Ai_m, Bi_m = f(x, k)
    return [Ai_m, Bi_m]

def minimize_relative_error(x, f, err=1e-10, k=1):
    Ai, _, Bi, _ = sc.airy(x)
    Ai_m, Bi_m = 1, 1
    while np.abs((Ai_m-Ai)/Ai) > err or np.abs((Bi_m-Bi)/Bi) > err:
        k += 1
        Ai_m, Bi_m = f(x, k)
    return [Ai_m, Bi_m]

results = []
x_m = np.linspace(divide_neg, divide_poz, 200)
for x in x_m:
    Ai, _, Bi, _ = sc.airy(x)
    Ai_m, Bi_m = minimize_absolute_error(x, Airy_m)
    results.append([Ai, Bi, Ai_m, Bi_m])
results = np.array(results).T
_, _, Ai_m, Bi_m = results

print("Aproksimacija Airyjevih funcij z Maclaurinovima vrstama      SUCCESSFUL")

height, width = 2,2
plt.figure(figsize=(10,10))
plt.suptitle("Aproksimacija Airyjevih funcij z Maclaurinovima vrstama")

plt.subplot(height, width,1)
plt.plot(x_m, results[0], label="Ai")
plt.plot(x_m, results[2], label="Ai_m")
plt.plot(x_m, results[1], label="Bi")
plt.plot(x_m, results[3], label="Bi_m")
plt.legend()
plt.xlabel("x")
plt.ylabel("log(y)")
plt.yscale("log")

plt.subplot(height, width,2)
m = 120
plt.plot(x_m[:m], results[0][:m], label="Ai")
plt.plot(x_m[:m], results[2][:m], label="Ai_m")
plt.plot(x_m[:m], results[1][:m], label="Bi")
plt.plot(x_m[:m], results[3][:m], label="Bi_m")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(height, width, 3)
plt.plot(x_m, np.abs((results[0]-results[2])), label="Ai napaka")
plt.plot(x_m, np.abs((results[1]-results[3])), label="Bi napaka")
plt.yscale("log")
plt.title("Absolutna napaka")
plt.xlabel("x")
plt.ylabel("log(|y - y_m|)")
plt.legend()

plt.subplot(height, width, 4)
plt.plot(x_m, np.abs((results[0]-results[2])/results[0]), label="Ai napaka")
plt.plot(x_m, np.abs((results[1]-results[3])/results[1]), label="Bi napaka")
plt.yscale("log")
plt.title("Relativna napaka")
plt.xlabel("x")
plt.ylabel("log(|(y - y_m)/y|)")
plt.legend()

plt.savefig("aprox_z_vrsto.pdf")
plt.clf()

# =============================================================================
# Aproksimacija z asimptotsko vrsto
# =============================================================================

def L(z):  
    return (1 + np.sum(np.cumprod(((-1/2)+(5)/((72)*s)+s/(2))/z)))

def P(z):
    return 1 + np.sum(np.cumprod(-1/z**2*(41/72-385/(10368*s)-3*s/2+s**2+25/(5184*(-1+2*s)))))

def Q(z):
    return 5/(z*72)*(1 + np.sum(np.cumprod(-1/z**2*(25+288*s**2*(-13+72*s**2))/(10368*s*(1+2*s)))))

def Airy_a_poz(x):
    x = (x)
    #x = x.astype(np.float128)
    ksi = (2/3)*np.abs(x)**(3/2)
    Ai_a = ((np.exp(-ksi))*L(-ksi)/((2)*(np.pi)**(1/2)*x**(1/4)))
    Bi_a = ((np.exp(ksi))*L(ksi)/((np.sqrt(np.pi))*x**((1/4))))
    return [Ai_a, Bi_a]

def Airy_a_neg(x, n):
    s = np.linspace(1, n, n)
    ksi = 2/3*np.power(np.abs(x),3/2)
    Ai_a = (np.sin(ksi-np.pi/4)*Q(ksi)+np.cos(ksi-np.pi/4)*P(ksi))/(np.sqrt(np.pi)*(-x)**(1/4))
    Bi_a = (-np.sin(ksi-np.pi/4)*P(ksi)+np.cos(ksi-np.pi/4)*Q(ksi))/(np.sqrt(np.pi)*(-x)**(1/4))
    return [Ai_a, Bi_a]

num_steps = 10
s = np.linspace(1, num_steps, num_steps)
s = s.astype(np.float128)
x = np.linspace(divide_poz, 30, 20*num_steps)
x_ = np.linspace(-30, divide_neg, 20*num_steps)
real_results = []
aprox_results = []
for i in range(len(x)):
    m, _, n, _ = sc.airy(x[i])
    o, _, p, _ = sc.airy(x_[i])
    real_results.append([(m), (n), o, p])
    aprox_results.append([*Airy_a_poz(x[i]), *Airy_a_neg(x_[i], 10)])
real_results = np.array(real_results).T
aprox_results = np.array(aprox_results).T
error = np.abs(real_results-aprox_results)
rel_error = np.abs(error/real_results)

print("Aproksimacija Airyjevih funcij z asimptotskimi vrstami       SUCCESSFUL")

height, width = 3,2
plt.figure(figsize=(10,16))
plt.suptitle("Aproksimacija Airyjevih funcij z asimptotskimi vrstami")

plt.subplot(height, width, 1)
plt.title("Aproksimacija za pozitivna števila")
plt.plot(x, real_results[0], label="Ai")
plt.plot(x, real_results[1], label="Bi")
plt.plot(x, aprox_results[0], label="Ai_a")
plt.plot(x, aprox_results[1], label="Bi_a")
plt.legend()
plt.xlabel("x")
plt.ylabel("log(y)")
plt.yscale("log")

plt.subplot(height, width, 3)
plt.title("Absolutina napaka")
plt.plot(x, error[0], label="Ai napaka")
plt.plot(x, error[1], label="Bi napaka")
plt.legend()
plt.xlabel("x")
plt.ylabel("log(|y-y_a|)")
plt.yscale("log")

plt.subplot(height, width, 5)
plt.title("Relativna napaka")
plt.plot(x, rel_error[0], label="Ai napaka")
plt.plot(x, rel_error[1], label="Bi napaka")
plt.legend()
plt.xlabel("x")
plt.ylabel("log(|(y-y_a)/y|)")
plt.yscale("log")

plt.subplot(height, width, 2)
plt.title("Aproksimacija za negativna števila")
plt.plot(x_, real_results[2], label="Ai")
plt.plot(x_, real_results[3], label="Bi")
plt.plot(x_, aprox_results[2], label="Ai_a")
plt.plot(x_, aprox_results[3], label="Bi_a")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.subplot(height, width, 4)
plt.title("Absolutina napaka")
plt.plot(x_, error[2], label="Ai napaka")
plt.plot(x_, error[3], label="Bi napaka")
plt.legend()
plt.xlabel("x")
plt.ylabel("log(|y-y_a|)")
plt.yscale("log")

plt.subplot(height, width, 6)
plt.title("Relativna napaka")
plt.plot(x_, rel_error[2], label="Ai napaka")
plt.plot(x_, rel_error[3], label="Bi napaka")
plt.legend()
plt.xlabel("x")
plt.ylabel("log(|(y-y_a)/y|)")
plt.yscale("log")

plt.savefig("aprox_z_asimp_vrsto.pdf")
plt.clf()

# =============================================================================
# Zlepek
# =============================================================================

def Airy(x):
    if x < divide_neg:
        return Airy_a_neg(x, 10)
    elif x > divide_poz:
        return Airy_a_poz(x)
    else:
        return Airy_m(x, 15)


# x_all = np.concatenate((x_, x_m, x))
# Ai_all, _, Bi_all, _ = sc.airy(x_all)
# Ai_a_all, Bi_a_all = np.concatenate((aprox_results[2], Ai_m, aprox_results[0])), np.concatenate((aprox_results[3], Bi_m, aprox_results[1]))

divide_neg, divide_poz = 0,0
x_all = np.linspace(-15, 15, 100)
Ai_all, _, Bi_all, _ = sc.airy(x_all)
Ai_a_all, Bi_a_all = [], []
Ai_p_all, Bi_p_all = [], []
for x in x_all:
    m,n = Airy(x)
    o,p = Airy_m(x, 15)
    Ai_a_all.append(m)
    Bi_a_all.append(n)
    Ai_p_all.append(o)
    Bi_p_all.append(p)
#Ai_all, Bi_all = np.array(Ai_all), np.array(Bi_all)
Ai_a_all, Bi_a_all = np.array(Ai_a_all), np.array(Bi_a_all)
Ai_p_all, Bi_p_all = np.array(Ai_p_all), np.array(Bi_p_all)
error = np.array([np.abs(Ai_all-Ai_a_all), np.abs(Bi_all-Bi_a_all)])
rel_error = np.array([np.abs((Ai_all-Ai_a_all)/Ai_all), np.abs((Bi_all-Bi_a_all)/Bi_all)])
error_p = np.array([np.abs(Ai_all-Ai_p_all), np.abs(Bi_all-Bi_p_all)])
rel_error_p = np.array([np.abs((Ai_all-Ai_p_all)/Ai_all), np.abs((Bi_all-Bi_p_all)/Bi_all)])

print("Zlepek aproksimacij Airyjevih funkcij                        SUCCESSFUL")

height, width = 1,2
plt.figure(figsize=(10,4))
plt.suptitle("Zlepek aproksimacij Airyjevih funkcij")

plt.subplot(height, width, 1)
mask = (x_all < 15)*(x_all > -15)
plt.title("Absolutna napaka")
plt.plot(x_all[mask], error[0][mask], label="Ai_a napaka")
plt.plot(x_all[mask], error[1][mask], label="Bi_a napaka")
plt.plot(x_all[mask], error_p[0][mask], label="Ai_m napaka")
plt.plot(x_all[mask], error_p[1][mask], label="Bi_m napaka")
plt.xlabel("x")
plt.ylabel("log(|y-y_a|)")
plt.yscale("log")
plt.legend()

plt.subplot(height, width, 2)
mask = (x_all < 8)*(x_all > -8)
plt.title("Relativna napaka")
plt.plot(x_all[mask], rel_error[0][mask], label="Ai_m napaka")
plt.plot(x_all[mask], rel_error[1][mask], label="Bi_m napaka")
plt.plot(x_all[mask], rel_error_p[0][mask], label="Ai_a napaka")
plt.plot(x_all[mask], rel_error_p[1][mask], label="Bi_a napaka")
plt.xlabel("x")
plt.ylabel("log(|(y-y_a)/y|)")
plt.yscale("log")
plt.legend()

plt.savefig("zlepek.pdf")
plt.clf()

# =============================================================================
# Ničle
# =============================================================================

def a_s(s):
    return -f(3*np.pi*(4*s-1)/8)

def b_s(s):
    return -f(3*np.pi*(4*s-3)/8)

def f(z):
    return z**(2/3)*(1+5/48/z**2-5/36/z**4+77125/82944/z**6-108056875/6967296/z**8)

divide_neg, divide_poz = -6, 7
# Get an initial guess for roots
x = np.linspace(-1, -61, 10000)
Ai, Bi = [], []
for i in x:
    res = Airy(i)
    Ai.append(res[0])
    Bi.append(res[1])
Ai = np.array(Ai)
Bi = np.array(Bi)
mask = (-1e-1 < Ai)*(Ai < 1e-1)
mask2 = (-1e-1 < Bi)*(Bi < 1e-1)
Ai_zeros = x[mask]
Bi_zeros = x[mask2]

# Generate more accurate roots
def Airy_Ai(x):
    return Airy(x)[0]
Ai_zeros_better = []
for zero in Ai_zeros:
    sol = op.fsolve(Airy_Ai, zero)
    Ai_zeros_better.append(*sol)
Ai_zeros2 =  np.unique(np.round(Ai_zeros_better, 8))[::-1]
Ai_zeros2 = Ai_zeros2[:100]

def Airy_Bi(x):
    return Airy(x)[1]
Bi_zeros_better = []
for zero in Bi_zeros:
    sol = op.fsolve(Airy_Bi, zero)
    Bi_zeros_better.append(*sol)
Bi_zeros2 =  np.unique(np.round(Bi_zeros_better, 8))[::-1]
Bi_zeros2 = Bi_zeros2[:100]

# Generate roots using the given formula
real_zeros = []
for s in range(1, 101):
    real_zeros.append([a_s(s), b_s(s)])
real_zeros = np.array(real_zeros).T

print("Ničle Airyjevih funkcij                                      SUCCESSFUL")

height, width = 1,2
plt.figure(figsize=(10,4))

plt.subplot(height, width, 1)
plt.title("Ničle Airyjevih funkcij")
plt.plot(np.linspace(1, len(real_zeros[0]), len(real_zeros[0])), real_zeros[0], label="Ai ničle")
plt.plot(np.linspace(1, len(real_zeros[1]), len(real_zeros[1])), real_zeros[1], label="Bi ničle")
plt.plot(np.linspace(1, len(Ai_zeros2), len(Ai_zeros2)), Ai_zeros2, label = "Ai_a ničle")
plt.plot(np.linspace(1, len(Ai_zeros2), len(Bi_zeros2)), Bi_zeros2, label = "Bi_a ničle")
plt.xlabel("n")
plt.ylabel("x_0")
plt.legend()

plt.subplot(height, width, 2)
plt.title("Absolutna napaka izračuna ničel")
plt.plot(np.linspace(1, len(real_zeros[0]), len(real_zeros[0])), np.abs(real_zeros[0]-Ai_zeros2), label="Ai napaka")
plt.plot(np.linspace(1, len(real_zeros[1]), len(real_zeros[1])), np.abs(real_zeros[1]-Bi_zeros2), label="Bi napaka")
plt.legend()
plt.xlabel("n")
plt.ylabel("log(|x_0-x_a0|)")
plt.yscale("log")

plt.savefig("nicle_Airyjevih_funkcij.pdf")
plt.clf()