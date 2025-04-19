"""
Simulation demonstrating the different types of pointing control for 400 seconds 

Each simulation assumes that the spacecraft attempts to point immediately
Gains are calculate using the provided requirements of a maximum 120s decay
time and damping ratio of xi=1.0
"""
import numpy as np

from spacecraft_dynamics.utils import initial_conditions as IC
from spacecraft_dynamics.core import (
    RotationalEquationsOfMotion, 
    RungeKutta
)
from spacecraft_dynamics.guidance import (
    SunPointingControl,
    NadirPointingControl,
    GmoPointingControl
)
from spacecraft_dynamics.analysis import plots as plots


if __name__ == "__main__":
    
    # Define the initial state
    sigma_0 = IC.LMO_SIGMA_BN_0.as_array()
    B_omega_BN_0 = IC.LMO_B_OMEGA_BN_0
    state_0 = np.array([sigma_0, B_omega_BN_0])
    I_b = IC.LMO_INERTIA

    # Define final simulation time
    t_final = 500

    # Determine gains for PD control (used in each simulation)
    # We want the slowest decay response time to be 120 so use the largest inertia value when calculating P
    decay_time = 120    # [s]
    I_max = np.max(I_b)
    P1 = 2*I_max/decay_time
    P = np.diag([P1, P1, P1])

    # At least one mode must be critically damped xi = 1
    # We want other modes to have damping rations <= 1 so use min inertia to calculate xi
    xi = 1.0
    I_min = np.min(np.diag(I_b))
    K1 = P1**2/(xi**2 * I_min)
    K = np.diag([K1, K1, K1])

    gains = (K,P)
    print("[PointingSimulations] \n> Gains: \n> K: {} \n> P: {}".format(gains[0], gains[1]))

    ####################################################
    # Conduct the sun pointing simulation
    ####################################################
    solution_state_sunpt = RungeKutta(init_state=state_0, 
                                t_max=t_final, 
                                t_step=1.0, 
                                diff_eq=RotationalEquationsOfMotion, 
                                args=(I_b, ), 
                                ctrl=SunPointingControl,
                                ctrl_args = gains)

    # Extract the log data
    sigma_BN_list_sunpt = solution_state_sunpt["MRP"]
    B_omega_BN_list_sunpt = solution_state_sunpt["omega_B_N"]
    t_list_sunpt = solution_state_sunpt["time"]
    sigma_BR = solution_state_sunpt["sigma_BR"]
    B_omega_BR = solution_state_sunpt["B_omega_BR"]

    # Print the values we're interested in
    print("{Sun Pointing Simulation Results}")
    for idx in range(len(t_list_sunpt)):
        sigma_BN = sigma_BN_list_sunpt[idx]
        B_omega_BN = B_omega_BN_list_sunpt[idx]
        t = t_list_sunpt[idx]
        if t == 15.0 or t == 100.0 or t == 200.0 or t == 400:
            print("> Time: {} \n> sigma_BN: {}".format(t, sigma_BN))

    plots.PlotMrpAndOmegaNorms(sigma_BR, B_omega_BR, t_list_sunpt, title = 'Evolution of $|\sigma_{B/R}|$ and $|\omega_{B/R}|$ in Sun Pointing Scenario')
    plots.PlotMrpAndOmegaComponents(sigma_BR, B_omega_BR, t_list_sunpt, title='Evolution of Reference Tracking Errors in Sun Pointing Scenario')
    ####################################################
    # Conduct the nadir pointing simulation
    ####################################################
    solution_state_nadir = RungeKutta(init_state=state_0, 
                                t_max=t_final, 
                                t_step=1.0, 
                                diff_eq=RotationalEquationsOfMotion, 
                                args=(I_b, ), 
                                ctrl=NadirPointingControl,
                                ctrl_args = gains)

    # Extract the log data
    sigma_BN_list_nadir = solution_state_nadir["MRP"]
    B_omega_BN_list_nadir = solution_state_nadir["omega_B_N"]
    t_list_nadir = solution_state_nadir["time"]
    sigma_BR = solution_state_nadir["sigma_BR"]
    B_omega_BR = solution_state_nadir["B_omega_BR"]

    # Print the values we're interested in
    print("{Nadir Pointing Simulation Results}")
    for idx in range(len(t_list_nadir)):
        sigma_BN = sigma_BN_list_nadir[idx]
        B_omega_BN = B_omega_BN_list_nadir[idx]
        t = t_list_nadir[idx]
        if t == 15.0 or t == 100.0 or t == 200.0 or t == 400:
            print("> Time: {} \n> sigma_BN: {}".format(t, sigma_BN))


    plots.PlotMrpAndOmegaNorms(sigma_BR, B_omega_BR, t_list_nadir, title = 'Evolution of $|\sigma_{B/R}|$ and $|\omega_{B/R}|$ in Nadir Pointing Scenario')
            

    ####################################################
    # Conduct the GMO pointing simulation
    ####################################################
    solution_state_gmopt = RungeKutta(init_state=state_0, 
                                t_max=t_final, 
                                t_step=1.0, 
                                diff_eq=RotationalEquationsOfMotion, 
                                args=(I_b, ), 
                                ctrl=GmoPointingControl,
                                ctrl_args = gains)

    # Extract the log data
    sigma_BN_list_gmopt = solution_state_gmopt["MRP"]
    B_omega_BN_list_gmopt = solution_state_gmopt["omega_B_N"]
    t_list_gmopt = solution_state_gmopt["time"]
    sigma_BR_list_gmopt = solution_state_gmopt["sigma_BR"]
    B_omega_BR_list_gmopt = solution_state_gmopt["B_omega_BR"]

    # Print the values we're interested in
    print("{GMO Pointing Simulation Results}")
    for idx in range(len(t_list_gmopt)):
        sigma_BN = sigma_BN_list_gmopt[idx]
        B_omega_BN = B_omega_BN_list_gmopt[idx]
        sigma_BR = sigma_BR_list_gmopt[idx]
        B_omega_BR = B_omega_BR_list_gmopt[idx]

        t = t_list_nadir[idx]
        if t == 15.0 or t == 100.0 or t == 200.0 or t == 400:
            print("> Time: {} \n> sigma_BN: {}, \n> sigma_BR: {} \n> B_omega_BR: {} ".format(t, sigma_BN, sigma_BR, B_omega_BR))            

    plots.PlotMrpAndOmegaNorms(sigma_BR_list_gmopt, B_omega_BR_list_gmopt, t_list_gmopt, title = 'Evolution of $|\sigma_{B/R}|$ and $|\omega_{B/R}|$ in GMO Pointing Scenario')
