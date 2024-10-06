import numpy as np
# This file is used to define the functions used in the restricted 
# three body problem

# r1, r2, OMEGA, ODE_R3BP, Jacobi_first_integral

def r1(mu, x, y):
    return np.sqrt((x - mu)**2 + y**2)

def r2(mu, x, y):
    return np.sqrt((x-mu+1)**2 + y**2)

def OMEGA(mu, x, y):
    return 0.5 * (x**2 + y**2) + (1 - mu) / r1(mu,x,y) + mu / r2(mu, x, y) \
           + 0.5 * (1 - mu) * mu

def ODE_R3BP(t, mu, X):
    # ODEs of the restricted three body problem
    return [X[2], X[3], 2*X[3] + X[0] - (1 - mu)*(X[0] - mu) / \
            r1(mu,X[0],X[1])**3 - mu * (X[0] - mu + 1) / \
            r2(mu,X[0],X[1])**3, -2*X[2] + X[1] - (1 - mu) * X[1] \
            / r1(mu,X[0],X[1])**3 - mu * X[1] / r2(mu,X[0],X[1])**3]

def Jacobi_first_integral(mu, x, y, vx, vy):
    # Function to compute the Jacobi first integral
    # This should be constant for a given value of mu
    return 2*OMEGA(mu, x, y) - (vx**2 + vy**2)