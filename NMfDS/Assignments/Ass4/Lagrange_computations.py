import numpy as np
from scipy.optimize import fsolve
from RTBP_definitions import r1, r2, OMEGA, ODE_R3BP, \
                             Jacobi_first_integral

# Compute x coordinate of the Lagrange points L1, L2, L3
# mu is the mass ratio of the two bodies with mass.
def compute_L1(mu):
    """
    Compute the L1 Lagrange point for the given mass ratio mu.
    For all the mathematics, see L13.
    """
    def polynomial(x):
        return x**5 - (3 - mu)*x**4 + (3 - 2*mu)*x**3 - mu*x**2 + 2*mu*x - mu
    # Estimation for eps_0
    eps = (mu / (3 * (1- mu)))**(1/3) # See L13
    L1 = fsolve(polynomial, eps)
    return mu-1+L1[0] # See L13

def compute_L2(mu):
    """
    Compute the L2 Lagrange point for the given mass ratio mu.
    For all the mathematics, see L13.
    """
    def polynomial(x):
        return x**5 + (3 - mu)*x**4 + (3 - 2*mu)*x**3 - mu*x**2 - 2*mu*x - mu
    # Estimation for eps_0
    eps_0 = (mu / (3 * (1- mu)))**(1/3) # See L13
    L2 = fsolve(polynomial, eps_0)
    return mu-1-L2[0] # See L13

def compute_L3(mu):
    """
    Compute the L3 Lagrange point for the given mass ratio mu.
    For all the mathematics, see L13.
    """
    def polynomial(x):
        return x**5 + (2 + mu)*x**4 + (1 + 2*mu)*x**3 - (1 - mu)*x**2 - \
               2*(1 - mu)*x - (1 - mu)
    # Estimation for eps_0
    eps_0 = 1 - (7 / 12)*mu # See L13
    L3 = fsolve(polynomial, eps_0)
    return mu + L3[0] # See L13

# Wrapper function
# Compute the position of the Lagrange point L for the given mu.
def compute_Lagrange_pt(mu, L):
    if L == 1:
        return compute_L1(mu)
    elif L == 2:
        return compute_L2(mu)
    elif L == 3:
        return compute_L3(mu)
    else:
        raise ValueError("Lagrange point must be 1, 2 or 3")
    
def compute_jacobi_const_Li(mu, L):
    '''
    Compute the Jacobi constant for the Lagrange point 1, 2 or 3
    '''
    xLi = compute_Lagrange_pt(mu, L)
    # from RTBP_definitions.py:
    C = Jacobi_first_integral(mu, xLi, 0, 0, 0)
    return C 