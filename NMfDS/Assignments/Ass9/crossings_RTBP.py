import numpy as np
import sys
dir_string = 'C:/Users/rannu/OneDrive - NTNU/Desktop/VsPython/'+\
                'Spain/NMfDS/Assignments/'
sys.path.append(dir_string + 'Ass4')
sys.path.append(dir_string + 'Ass6')
sys.path.append(dir_string + 'Ass7')
sys.path.append(dir_string + 'Ass8')
#from RTBP_definitions import r1, r2, OMEGA, ODE_R3BP, Jacobi_first_integral
from Lagrange_computations import compute_Lagrange_pt, compute_jacobi_const_Li
#from custom_ODE_solver import ODE_solver
from PoincareR3BP import poincare_map_solve_ivp_R3BP
from variational_equation_RTBP import variational_eq

def crossings_R3BP(no_crossings, ODE_R3BP, initial_conditions, dir,
                                step, t_span, mu, init_search=100, 
                                init_tol=1e-12, 
                                refinement=100, newton_tol = 1e-15):
    # Procedure to compute when the x-axis is crossed
    # for no_crossings times
    # returns an array of the crossing times
    # and a 2d array of the initial conditions at all the crossings
    crossing_times = np.zeros(no_crossings)
    crossing_initials = np.zeros((no_crossings, 4))
    mu_fails = np.zeros(no_crossings)
    new_initial, time_duration, mu_fail = poincare_map_solve_ivp_R3BP\
                                                (ODE_R3BP, initial_conditions, dir,
                                                step, t_span, mu, init_search, 
                                                init_tol,
                                                refinement, newton_tol)
    crossing_times[0] = time_duration
    crossing_initials[0] = new_initial
    mu_fails[0] = mu_fail

    for i in range(1, no_crossings):
        new_initial, time_duration, mu_fail = poincare_map_solve_ivp_R3BP\
                                        (ODE_R3BP, crossing_initials[i-1], dir,
                                        step, t_span, mu, init_search, 
                                        init_tol,
                                        refinement, newton_tol)
        crossing_times[i] = time_duration
        crossing_initials[i] = new_initial
        mu_fails[i] = mu_fail
    
    return crossing_times, crossing_initials, mu_fails

def crossings_R3BP_by_mu(no_crossings, L123, ODE_R3BP, dir,
                                step, t_span, mu, 
                                init_search=100, init_tol=1e-12, 
                                refinement=100, newton_tol = 1e-15,
                                start_cond_tol = 10**-6):
    Li = [compute_Lagrange_pt(mu, L123), 0, 0, 0] 
    Li.extend([1, 0, 0, 0, 
               0, 1, 0, 0, 
               0, 0, 1, 0, 
               0, 0, 0, 1])  # initial conditions and identity matrix
    time_span = 0

    # compute the Jacobian matrix of the RTBP at Li.
    # The eigenvalues of this matrix are the frequencies of the periodic orbit

    A = variational_eq(t_span[0], Li, mu, 1)[4:20].reshape(4,4)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    lambda_pos = eigenvalues[3].real
    lambda_neg = eigenvalues[2].real
    eigvec_pos = eigenvectors[:,3].real
    eigvec_neg = eigenvectors[:,2].real
    if lambda_pos < lambda_neg:
        print("Warning: eigenvalues are not ordered")
        temp = eigvec_pos
        eigvec_pos = eigvec_neg
        eigvec_neg = temp
    if dir == 1:
        if eigvec_pos[1] > 0:
            eigvec_pos = -eigvec_pos
            #eigvec_pos[0] = -eigvec_pos[0]
        v = eigvec_pos
    elif dir == -1:
        v = eigvec_neg
    else:
        raise ValueError("Direction must be 1 or -1")
    init_cond = Li[0:4] + v*start_cond_tol

    crossing_times, crossing_initials, mu_fails = \
        crossings_R3BP(no_crossings, ODE_R3BP, init_cond, dir, step, t_span, 
                       mu, init_search, init_tol, refinement, newton_tol)
    return crossing_times, crossing_initials, mu_fails