import sys
import scipy as sp
import numpy as np
sys.path.append('C:/Users/rannu/OneDrive - NTNU/Desktop/VsPython/' +
                'Spain/NMfDS/Assignments/Ass4')
sys.path.append('C:/Users/rannu/OneDrive - NTNU/Desktop/VsPython/' +
                'Spain/NMfDS/Assignments/Ass6')
from RTBP_definitions import r1, r2, OMEGA, ODE_R3BP, \
                            Jacobi_first_integral
from custom_ODE_solver import ODE_solver

def poincare_map_solve_ivp_R3BP(ODE_R3BP, initial_conditions, dir,
                                step, t_span, mu, init_search=100, 
                                init_tol=1e-12, refinement=100, 
                                newton_tol = 1e-15):
    """
    Solve the Poincare map for the Restricted Three-Body Problem (R3BP).

    Args:
        ODE_R3BP (function): The Ordinary Differential Equation (ODE) for the R3BP.
        initial_conditions (list): The initial conditions for the ODE.
        dir (int): The direction of integration (+1 for forward, -1 for backward).
        step (float): The step size for integration.
        t_span (list): The time span for integration.
        mu (float): The mass ratio parameter for the R3BP.
        init_search (int, optional): The number of steps to search for the first crossing. Defaults to 100.
        refinement (int, optional): The number of iterations for Newton's method refinement. Defaults to 100.
        newton_tol (float, optional): The tolerance for Newton's method convergence. Defaults to 1e-15.

    Raises:
        ValueError: Raised when no crossing is found within the specified search range.
        ValueError: Raised when Newton's method does not converge within the specified number of iterations.

    Returns:
        tuple: A tuple containing the new initial conditions and the time duration of the crossing.
    """
    # Procedure to compute when the x-axis is crossed
    # for the first time
    product = 1
    time = 0
    startPoint = np.array(initial_conditions)
    initial_conditions = initial_conditions
    failed_mu = -1

    while product >= 0 and abs(time) < abs(step*init_search):
        solution = ODE_solver(ODE_R3BP, startPoint, t_span[1], 1000,
                              t_min=t_span[0], tol = init_tol,
            hamiltonian=lambda X: Jacobi_first_integral(mu, X[0], X[1],
                                                        X[2], X[3]))
        Y = solution.y.T  # Transposing to match previous structure
        product = Y[1, 1] * Y[-1, 1]  # Check if x-axis is crossed
        startPoint = Y[-1, :]
        t_span = [t_span[0] + dir * step, t_span[1] + dir * step]
        time += step*dir
    # make an error if the x-axis is never crossed
    if abs(time) >= abs(step*init_search):
        raise ValueError("No crossing found, initial search failed" +
                          "\ntime:  " + str(time) +
                         "\nproduct  " + str(product) + "\nstep  " 
                         + str(step) +
                            "\ninit_search  " + str(init_search))
    # Procedure to compute the exact time of the crossing
    for i in range(refinement):
        solution = ODE_solver(ODE_R3BP, initial_conditions, time, 1000,
                              tol = init_tol,
                              hamiltonian=lambda X: \
                                Jacobi_first_integral(mu, X[0], X[1],
                                                        X[2], X[3]))
        Y = solution.y.T
        # One iteration of Newton's method.
        difference = Y[-1, 1] / ODE_R3BP(0, Y[-1, :])[1]
        time -= difference

        if i == refinement - 1:
            # raise ValueError("No convergence, refinement failed" + 
            #                  "\ntime:  " + str(time) +
            #                  "\ndifference  " + str(difference))
            #print("No convergence, refinement failed " + "mu: " + str(mu))
            failed_mu = mu
        if abs(difference) < newton_tol or abs(Y[-1, 1]) < newton_tol:
            # print("Convergence after", i, "iterations")
            break

    TimeDuration = time
    newInitial = Y[-1, :]
    return newInitial, TimeDuration, failed_mu
