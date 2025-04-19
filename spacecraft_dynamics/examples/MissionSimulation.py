"""
Script to execute the full mission scenario in which all pointing modes are demonstrated
and transitions between them are automatic. The results of the simulation are 
plotted.

This simulation encompases all the code for the capstone project of 
ASEN 5010 - Spacecraft Attitude Dynamics and Control at CU Boulder

The supporting documentation can be found under spacecraft_dynamics/docs
"""
import numpy as np

from spacecraft_dynamics.utils import initial_conditions as IC
from spacecraft_dynamics.core import (
    RotationalEquationsOfMotion, 
    RungeKutta
)
from spacecraft_dynamics.guidance import MissionPointingControl
from spacecraft_dynamics.analysis import plots as plots

if __name__ == "__main__":
    
    # Define the initial state of the LMO spacecraft
    sigma_0 = IC.LMO_SIGMA_BN_0.as_array()
    B_omega_BN_0 = IC.LMO_B_OMEGA_BN_0
    state_0 = np.array([sigma_0, B_omega_BN_0])
    I_b = IC.LMO_INERTIA

    # Define final simulation time
    t_final = 6500

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


    ####################################################
    # Conduct the mission pointing simulation
    ####################################################
    solution_state_mission = RungeKutta(init_state=state_0, 
                                t_max=t_final, 
                                t_step=1.0, 
                                diff_eq=RotationalEquationsOfMotion, 
                                args=(I_b, ), 
                                ctrl=MissionPointingControl,
                                ctrl_args = gains)

    # Extract the log data
    sigma_BN_list_mission = solution_state_mission["MRP"]
    B_omega_BN_list_mission = solution_state_mission["omega_B_N"]
    t_list_mission = solution_state_mission["time"]
    mode_list_mission = solution_state_mission["mode_value"]
    sigma_BR_list_mission = solution_state_mission["sigma_BR"]
    B_omega_BR_list_mission = solution_state_mission["B_omega_BR"]
    B_u_list_mission = solution_state_mission["control"]


    # Print the values we're interested in
    print("{Mission Pointing Simulation Results}")
    for idx in range(len(t_list_mission)):
        sigma_BN = sigma_BN_list_mission[idx]
        B_omega_BN = B_omega_BN_list_mission[idx]
        t = t_list_mission[idx]
        if t == 300.0 or t == 2100.0 or t == 3400.0 or t == 4400.0 or t == 5600.0:
            print("> Time: {} \n> sigma_BN: {}".format(t, sigma_BN))

    plots.PlotMode(mode_list_mission, t_list_mission, title='Pointing Mode in Full Mission Simulation')
    plots.PlotMrpAndOmegaComponents(sigma_BN_list_mission, B_omega_BN_list_mission, t_list_mission, title='Attitude and Angular Velocity in Full Mission Simulation')
    plots.PlotMrpAndOmegaNorms(sigma_BR_list_mission, B_omega_BR_list_mission, t_list_mission, title = 'Evolution of $|\sigma_{B/R}|$ and $|\omega_{B/R}|$ in Full Mission Simulation')
    plots.PlotTorqueComponents(B_u_list_mission, t_list_mission, title='Control Torque Components in Full Mission Simulation')
