"""
Example script creating custom guidance and control functions
"""
import numpy as np

from spacecraft_dynamics.core import RungeKutta, RotationalEquationsOfMotion

from spacecraft_dynamics.utils import MRP
import spacecraft_dynamics.utils.rigid_body_kinematics as RBK
from spacecraft_dynamics.control import CalculateAttitudeError
from spacecraft_dynamics.guidance import Mode

import spacecraft_dynamics.analysis.plots as plots


def ReferenceAttitude(t:float, f:float = 0.03) -> np.array:

    sigma_RN = MRP(0.1 * np.sin(f * t), 
                   0.2 * np.cos(f * t), 
                   -0.3 * np.sin(2.0 * f * t))
    dcm_R_N = RBK.MRP2C(sigma_RN.as_array())

    return dcm_R_N

def ReferenceAngularVelocity(t:float, dt:float=0.0001) -> np.array:
    # Approximate [NRc_dot] numerically by applying a small time step and evaluating the change in [NRc]

    dcm_R_N_0 = ReferenceAttitude(t - dt)
    dcm_R_N_1 = ReferenceAttitude(t)

    dcm_N_R_0 = np.transpose(dcm_R_N_0)
    dcm_N_R_1 = np.transpose(dcm_R_N_1)
    
    dcm_N_R_dt = dcm_N_R_1 - dcm_N_R_0

    dcm_N_R_dot = 1/dt * dcm_N_R_dt
    
    # Kinematic differential equation [C_dot] = -[omega_tilde][C] --> omega_tilde = -[C_dot] * [C]^T
    N_omega_NR_tilde = np.matmul(-dcm_N_R_dot, np.transpose(dcm_N_R_1))

    # We have the angular velocity of N wrt Rc but we need angular velocity of Rc wrt N
    N_omega_RN_tilde = np.transpose(N_omega_NR_tilde)

    # Extract angular velocity from the tilde matrix
    w1 = -N_omega_RN_tilde[1,2]
    w2 = N_omega_RN_tilde[0,2]
    w3 = -N_omega_RN_tilde[0,1]
    N_omega_RN = np.array([w1, w2, w3])
    
    return N_omega_RN

def ReferenceAngularAcceleration(t:float, dt:float=0.0001) -> np.array:
    '''Function for approximating N_omega_RN_dot'''

    N_omega_RN_0 = ReferenceAngularVelocity(t-dt)
    N_omega_RN_1 = ReferenceAngularVelocity(t)

    N_omega_RN_dot = 1/dt*(N_omega_RN_1 - N_omega_RN_0)

    return N_omega_RN_dot

def ControlFunction(t, state, gains=(np.eye(3),np.eye(3)), inertia=np.eye(3)):
    """
    Attitude reference given as a function of time and angular velocity reference is estimated numerically
    
    This function implements a slightly more complicated control law that uses the gyroscopic terms of the reference
    angular velocity
    """
    pointing_mode = Mode.INVALID

    sigma_BN : np.array = state[0]
    B_omega_BN : np.array = state[1]
    K = gains[0]
    P = gains[1]

    # Calculate reference values to be tracked
    dcm_R_N = ReferenceAttitude(t)
    N_omega_RN = ReferenceAngularVelocity(t)
    N_omega_RN_dot = ReferenceAngularAcceleration(t)

    # Use the current attitude to determine the attitude and ang vel tracking error
    sigma_BR, B_omega_BR = CalculateAttitudeError(sigma_BN, B_omega_BN, dcm_R_N, N_omega_RN)

    # if sigma_BR.norm() > 1.0:
    #     sigma_BR = sigma_BR.convert_to_shadow_set()

    # Convert ang vel and acceleration reference to body frame
    dcm_B_N = RBK.MRP2C(sigma_BN)
    B_omega_RN = np.matmul(dcm_B_N, N_omega_RN)
    B_omega_RN_dot = np.matmul(dcm_B_N, N_omega_RN_dot)

    # Calculate tilde matrix of omega_BN
    B_omega_BN_tilde = RBK.v3Tilde(B_omega_BN)

    # Calculate control torques
    u = np.matmul(-K, sigma_BR.as_array()) \
        - np.matmul(P, B_omega_BR) \
            + np.matmul(inertia, (B_omega_RN_dot - np.matmul(B_omega_BN_tilde, B_omega_RN))) \
                + np.matmul(B_omega_BN_tilde, np.matmul(inertia, B_omega_BN)) 

    return u, pointing_mode, sigma_BR, B_omega_BR


if __name__ == "__main__":
    
    # Define the initial state of the LMO spacecraft
    sigma_0 = MRP(0.1, 0.2, -0.1).as_array()
    B_omega_BN_0 = np.deg2rad(np.array([30.0, 10.0, -20.0]))
    state_0 = np.array([sigma_0, B_omega_BN_0])
    I_b = np.diag([100, 75, 80])

    # Define final simulation time
    t_final = 50

    # Define gains
    P1 = 10
    P = np.diag([P1, P1, P1])

    K1 = 5
    K = np.diag([K1, K1, K1])

    gains = (K,P)


    ####################################################
    # Conduct the mission pointing simulation
    ####################################################
    solution_state_mission = RungeKutta(init_state=state_0, 
                                t_max=t_final, 
                                t_step=0.01, 
                                diff_eq=RotationalEquationsOfMotion, 
                                args=(I_b, ), 
                                ctrl=ControlFunction,
                                ctrl_args = gains,
                                inertia=I_b)

    # Extract the log data
    sigma_BN_list_mission = solution_state_mission["MRP"]
    B_omega_BN_list_mission = solution_state_mission["omega_B_N"]
    t_list_mission = solution_state_mission["time"]
    mode_list_mission = solution_state_mission["mode_value"]
    sigma_BR_list_mission = solution_state_mission["sigma_BR"]
    B_omega_BR_list_mission = solution_state_mission["B_omega_BR"]
    B_u_list_mission = solution_state_mission["control"]


    # Print the values we're interested in
    print("{Simulation Results}")
    for idx in range(len(t_list_mission)):
        sigma_BN = sigma_BN_list_mission[idx]
        B_omega_BN = B_omega_BN_list_mission[idx]
        sigma_BR = sigma_BR_list_mission[idx]
        t = t_list_mission[idx]
        # print("> Time: {} sigma_BR: {}".format(t, sigma_BR))
        if (abs(t-15.0) < 1e-6) or (abs(t-40.0) < 1e-6):
            print("> Time: {} sigma_BR: {}".format(t, sigma_BR))
    
    plots.PlotMrpAndOmegaComponents(sigma_BN_list_mission, B_omega_BN_list_mission, t_list_mission, title='Attitude and Angular Velocity in Simulation')
    plots.PlotMrpAndOmegaComponents(sigma_BR_list_mission, B_omega_BR_list_mission, t_list_mission, title='Attitude and Angular Velocity Error in Simulation')
    plots.PlotMrpAndOmegaNorms(sigma_BR_list_mission, B_omega_BR_list_mission, t_list_mission, title = 'Evolution of $|\sigma_{B/R}|$ and $|\omega_{B/R}|$ in Full Mission Simulation')
    plots.PlotTorqueComponents(B_u_list_mission, t_list_mission, title='Control Torque Components in Full Mission Simulation')
