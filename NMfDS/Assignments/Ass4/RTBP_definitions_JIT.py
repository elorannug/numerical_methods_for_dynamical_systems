import numpy as np
from numba import jit

# This file is used to define the functions used in the restricted
# three body problem

# r1, r2, OMEGA, ODE_R3BP, Jacobi_first_integral


@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def r1(mu, x, y):
    return np.sqrt((x - mu)**2 + y**2)


@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def r2(mu, x, y):
    return np.sqrt((x-mu+1)**2 + y**2)


@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def OMEGA(mu, x, y):
    return 0.5 * (x**2 + y**2) + (1 - mu) / r1(mu, x, y) + mu / r2(mu, x, y) \
        + 0.5 * (1 - mu) * mu


@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def ODE_R3BP(t, X, mu):
    # ODEs of the restricted three body problem
    return [X[2], X[3], 2*X[3] + X[0] - (1 - mu)*(X[0] - mu) /
            r1(mu, X[0], X[1])**3 - mu * (X[0] - mu + 1) /
            r2(mu, X[0], X[1])**3, -2*X[2] + X[1] - (1 - mu) * X[1]
            / r1(mu, X[0], X[1])**3 - mu * X[1] / r2(mu, X[0], X[1])**3]


@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def Jacobi_first_integral(X, mu):
    x = X[0]
    y = X[1]
    vx = X[2]
    vy = X[3]
    # Function to compute the Jacobi first integral
    # This should be constant for a given value of mu
    return 2*OMEGA(mu, x, y) - (vx**2 + vy**2)


def hi():
    print('hia')
