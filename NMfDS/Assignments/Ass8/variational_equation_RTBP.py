import numpy as np
import sys
dir_string = 'C:/Users/rannu/OneDrive - NTNU/Desktop/VsPython/'+\
                'Spain/NMfDS/Assignments/'
sys.path.append(dir_string + 'Ass4')
from RTBP_definitions import r1, r2


# Define the variation equation as a function of time, x, mu and direction
def variational_eq(t, x, mu, dir):
    var_eq = np.zeros(20)
    # Defining these as variables so solve_ivp can handle them
    r1_val = r1(mu, x[0], x[1])
    r2_val = r2(mu, x[0], x[1])

    var_eq[0] = x[2]
    var_eq[1] = x[3]
    var_eq[2] = 2 * x[3] + x[0] - (1 - mu) * (x[0] - mu) / r1_val**3 \
                - mu * (x[0] - mu + 1) / r2_val**3
    var_eq[3] = -2 * x[2] + x[1] * (1 - (1 - mu) / r1_val**3 - mu / r2_val**3)

    Omegaxx = 1 - (1 - mu) / r1_val**3 \
            + 3 * (1 - mu) * (x[0] - mu)**2 / r1_val**5 - mu / r2_val**3 \
            + 3 * mu * (x[0] - mu + 1)**2 / r2_val**5
    Omegayy = 1 - (1 - mu) / r1_val**3 - mu / r2_val**3 \
            + (3 * (1 - mu) * x[1]**2) / r1_val**5 \
            + (3 * mu * x[1]**2) / r2_val**5
    Omegaxy = 3 * (1 - mu) * x[1] * (x[0] - mu) / r1_val**5 \
            + 3 * mu * x[1] * (x[0] - mu + 1) / r2_val**5
    
    for i in range(4, 12):
        var_eq[i] = x[i + 8]

    for i in range(12, 16):
        var_eq[i] = Omegaxx * x[i - 8] + Omegaxy * x[i - 4] + 2 * x[i + 4]

    for i in range(16, 20):
        var_eq[i] = Omegaxy * x[i - 12] + Omegayy * x[i - 8] - 2 * x[i - 4]

    # Flip the sign of the variation equation if we are looking at the
    # stable manifold
    if dir == -1:
        var_eq = -var_eq

    return var_eq
