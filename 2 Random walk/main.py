# =============================================================================
# Naključni sprehod
# Avtor: Miha Pompe
# =============================================================================

import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import time
import pandas as pd
from scipy import optimize
np.random.seed(int(time.time()))

# x = [[x, y, t],...]
# x_ = [[x, ...], [y, ...], [t, ...]]

# =============================================================================
# Core functions
# =============================================================================

def random_l(num_steps, mi):
    l = np.random.pareto(mi-1, num_steps)+1
    return l

def random_random(num_steps, mi):
    l = []
    for i in range(num_steps):
        l.append((1-random.random())**(-1/(mi-1)))
    l = np.array(l)
    return l

def interpolate_x(x):
    out = []
    out.append(x[0])
    for i in range(1, len(x)):
        dt = int(x[i][2]-out[-1][2])
        dx = (x[i][0]-out[-1][0])/dt
        dy = (x[i][1]-out[-1][1])/dt
        one = np.cumsum(np.ones(dt))
        res = np.array([dx*one+out[-1][0], dy*one+out[-1][1], one+out[-1][2]]).T
        for t in res:
            out.append(t)
    x = np.array(out)
    return x

def random_walk(num_steps, mi=4, ni = 1.5, walk_type="flight", sticking_time=False):
    #l = np.random.rand(num_steps)       # change distribution
    l = random_l(num_steps, mi)
    if walk_type == "flight":
        t = np.linspace(1, num_steps, num_steps)
    elif walk_type == "walk":
        t = np.cumsum(l)
    if sticking_time:
        sticking_t = np.cumsum(random_l(num_steps, ni))
        t += sticking_t
    fi = 2*np.pi*np.random.rand(num_steps)
    step = np.cumsum(l*[np.cos(fi), np.sin(fi)], axis=1)
    x = np.array([step[0], step[1], t]).T
    x = np.concatenate(([np.zeros(3)], x), axis=0)
    if walk_type == "walk":
        x = interpolate_x(x)
    return x

def trim_walks(walks):
    min = 1e10
    for walk in walks:
        ln = len(walk)
        if min > ln:
            min = ln
    out = []
    for walk in walks:
        out.append(walk[:min])
    return out

def walk_distribution(n, num_steps, mi=4, ni = 1.5, walk_type="flight", sticking_time=False):
    # Returns final coordinates of n walks
    # n - number of walks
    walks = []
    means = []
    for i in range(n):
        x = random_walk(num_steps, mi, ni, walk_type, sticking_time)
        walks.append(x)
    print("Generated all random walks                               SUCCESSFUL")
    if walk_type == "walk" or sticking_time == True:
        walks = trim_walks(walks)
    walks = np.array(walks)
    x_finals = walks[:,-1]
    for t in range(len(walks[0,:,-1])):
        x_mean = (sc.median_abs_deviation(walks[:,t,0])*1.4826)**2
        y_mean = (sc.median_abs_deviation(walks[:,t,1])*1.4826)**2
        means.append([x_mean, y_mean])
    means = np.array(means)
    return x_finals, means

# =============================================================================
# Visualization
# =============================================================================

def draw_walk(x):
    plt.title("Naključni sprehod")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x[:,0], x[:,1], "o-")
    plt.show()
    return

def draw_distribution(x_finals):
    plt.subplot(1,2,1)
    plt.hist(x_finals[0], )
    plt.title("Distribucija x-os")
    plt.subplot(1,2,2)
    plt.hist(x_finals[1])
    plt.title("Distribucija y-os")
    plt.show()
    return

def draw_means(means):
    plt.plot(means[:,0])
    plt.xlabel("t")
    plt.ylabel("MAD")
    return

def exercise_plot_walks():
    plt.figure(figsize=(11,10))
    plt.suptitle("Naključni sprehod")

    plt.subplot(2,2,1)
    x = random_walk(10)
    plt.title("Število korakov: 10")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x[:,0], x[:,1])

    plt.subplot(2,2,2)
    x = random_walk(100)
    plt.title("Število korakov: 100")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x[:,0], x[:,1])

    plt.subplot(2,2,3)
    x = random_walk(1000)
    plt.title("Število korakov: 1000")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x[:,0], x[:,1])

    plt.subplot(2,2,4)
    x = random_walk(10000)
    plt.title("Število korakov: 10000")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x[:,0], x[:,1])
    title = "nakljucni_sprehodi"
    plt.savefig(f"{title}.pdf")
    #plt.show()
    print(f"Saved {title}.pdf                                       SUCCESSFUL")
    return

def exercise_plot_gamma(walk_type, n_sim, num_steps, sticking_time = False, ni = 1.5):
    plt.figure(figsize=(11,10))
    suptitle = f"Levy {walk_type}"
    if sticking_time: suptitle += ", sticking time"
    plt.suptitle(suptitle)
    for i in range(4):
        mi = 2+i/2 
        _, means = walk_distribution(n_sim, num_steps, mi=mi, ni=ni, walk_type=walk_type, sticking_time=sticking_time)
        x_mean = means[:,0]
        x = np.cumsum(x_mean**0)

        def fitting_function(x, k, n):
            if walk_type == "flight":
                if 1 < mi < 3: gama = 2/(mi-1)
                elif mi >= 3: gama = 1
            elif walk_type == "walk":
                if 1 < mi <= 2: gama = 2
                elif 2 < mi < 3: gama = 4-mi
                elif mi == 3: return k*x*np.log(x)+n
                elif mi > 3: gama = 1
            return k*x**gama+n

        pars, cov = optimize.curve_fit(fitting_function, x, x_mean)
        k, n = pars
        plt.subplot(2,2,i+1)
        plt.xlabel("t")
        plt.ylabel("MAD")
        title = f"$\mi = {mi} $"
        if sticking_time:
            title += f"$, \ni = {ni}$"
        title += f", število simulacij {n_sim}"
        plt.title(title)
        plt.plot(x, x_mean, label="simulacija")
        plt.plot(x, [fitting_function(i, k, n) for i in x], label="fit")
        plt.legend()
    plt.savefig(f"{suptitle}.pdf")
    print("Analiza MAD vrednosti                                    SUCCESSFUL")
    print(f"Saved {suptitle}.pdf                 SUCCESSFUL")
    #plt.show()
    return

def primerjava_generatorjev():
    num_steps = 1000000
    mi = 2
    max = 1000
    x_1 = random_l(num_steps, mi)
    x_2 = random_random(num_steps, mi)
    x = np.linspace(0, max, max)
    y = x**(-mi)
    x_1 = x_1[x_1 < max]
    x_2 = x_2[x_2 < max]

    plt.suptitle("Primerjava generatorjev porazdelitve l\n N = 1000000")
    plt.subplot(1, 2, 1)
    plt.title("numpy.random.pereto")
    plt.hist(x_1, density=True, label="Generirano", bins=20)
    #plt.plot(x_1)
    plt.plot(x, y, label="Fit $x^{-2}$")
    plt.ylabel("$log(N)$")
    plt.xlabel("$x$")
    plt.legend()
    plt.yscale("log")

    plt.subplot(1, 2, 2)
    plt.title("Integral")
    plt.hist(x_2, density=True, label="Generirano", bins=20)
    plt.plot(x, y, label="Fit $x^{-2}$")
    plt.legend()
    plt.ylabel("$log(N)$")
    plt.xlabel("$x$")
    plt.yscale("log")
    plt.savefig("primerjava_gen.pdf")
    plt.show()
    return

def exercise_plot_linear_gamma(walk_type, n_sim, num_steps, sticking_time = False, ni = 1.5 ):
    suptitle = f"Levy {walk_type}"
    if sticking_time: suptitle += ", sticking time"
    plt.title(suptitle)
    for i in range(4):
        mi = 2+i/2 
        _, means = walk_distribution(n_sim, num_steps, mi=mi, ni=ni, walk_type=walk_type, sticking_time=sticking_time)
        x_mean = np.log(np.abs(means[:,0])+0.01)
        x = np.log(np.cumsum(x_mean**0))
        x_mean = x_mean[x > 1]
        x_mean_mean = sc.median_abs_deviation(x_mean)
        x = x[x > 1]

        def gama(mi):
            if walk_type == "flight":
                if 1 < mi < 3: gama = 2/(mi-1)
                elif mi >= 3: gama = 1
            elif walk_type == "walk":
                if 1 < mi <= 2: gama = 2
                elif 2 < mi < 3: gama = 4-mi
                elif mi >= 3: gama = 1
            return round(gama, 2)
        
        def fitting_function(x, k, n):
            return k * x + n

        pars, cov = optimize.curve_fit(fitting_function, x, x_mean)
        k, n = pars
        stdevs = np.sqrt(np.diag(cov))
        print(pars, stdevs)
        plt.xlabel("log(t)")
        plt.ylabel("log(MAD)")
        title = f"mi = {mi} "
        if sticking_time:
            title += f", ni = {ni}"
        title += f", število simulacij {n_sim}"
        plt.plot(x, x_mean, label="sim, $\gamma = $"+str(gama(mi))+ " $\mu = $" + str(mi))
        plt.plot(x, [fitting_function(i, k, n) for i in x], label="fit, $\gamma_{fit} = $"+str(round(k,2)) + " $\pm$ " + str(round(stdevs[0], 4)))
        plt.fill_between(x, x_mean-x_mean_mean/2, x_mean+x_mean_mean/2, alpha=0.2)
    plt.legend()
    plt.savefig(f"lin_{suptitle}.pdf")
    print("Analiza MAD vrednosti                                    SUCCESSFUL")
    print(f"Saved {suptitle}.pdf                 SUCCESSFUL")
    #plt.show()
    return


if __name__ == "__main__":
    # exercise_plot_walks()
    # exercise_plot_gamma("walk", 1000, 1000)
    # exercise_plot_gamma("flight", 1000, 1000)
    # exercise_plot_gamma("flight", 1000, 1000, True, 1.1)
    exercise_plot_linear_gamma("flight", 10000, 10000)
    # exercise_plot_linear_gamma("walk", 1000, 1000)
    # primerjava_generatorjev()
    # exercise_plot_linear_gamma("flight", 1000, 1000, True, 1.5)
    
