import numpy as np
import scipy as sp

def ODE_solver(func, x0, t_max, eval_pts, tol=1e-12, t_min=0, method='DOP853', hamiltonian=0):
    # Variable explanation:
    # func: the function f in the system of ODEs
    # hamiltonian: the hamiltonian of the system
    # x0: the initial condition, a vector of size n
    # t_max: the maximum time
    # eval_pts: number of points at which the solution is evaluated
    # tol: tolerance for checking if the hamiltonian is constant
    # t_min: the minimum time
    # method: the method used to solve the system of ODEs

    # RETURNS: the solution of the system of ODEs as a scipy.integrate.solve_ivp object

    t_eval = np.linspace(t_min, t_max, eval_pts)

    sol = sp.integrate.solve_ivp(
        func, [t_min, t_max], x0, method=method, 
        t_eval=t_eval, atol=tol, rtol=tol)

    # Check if the hamiltonian remains constant for all points, by checking the error. break iff error > tol
    # Also chek the determinant of the matrix [[x3, x4], [x5, x6]]
    # if hamiltonian == 0:
    #     print("Warning: No hamiltonian given. The hamiltonian will not be checked for constancy.")
    # else:
    #     for i in range(len(sol.t)):
    #         if abs(hamiltonian(sol.y[:, i]) - hamiltonian(x0)) > tol:
    #             print("Hamiltonian is not constant for all points. Error = ", abs(
    #                 hamiltonian(sol.y[:, i]) - hamiltonian(x0)))
    #             print("Hamiltonian at x0 = ", hamiltonian(x0))
    #             print("Hamiltonian at x = ", hamiltonian(sol.y[:, i]))
    #             break
    return sol