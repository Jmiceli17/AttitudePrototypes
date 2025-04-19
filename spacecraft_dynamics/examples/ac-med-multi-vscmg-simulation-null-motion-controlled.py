import numpy as np
import math

from spacecraft_dynamics.actuators import Vscmg
from spacecraft_dynamics.core import (
    Spacecraft, 
    ControlGains,
    ControlledSpacecraftDynamics
)
from spacecraft_dynamics.state import (
    SpacecraftState, 
    VscmgState
)
from spacecraft_dynamics.utils import MRP
import spacecraft_dynamics.utils.rigid_body_kinematics as RBK
from spacecraft_dynamics.control import CalculateAttitudeError
from spacecraft_dynamics.guidance import Mode

import spacecraft_dynamics.analysis.plots as plots


def ReferenceAttitude(t:float, f:float = 0.03) -> np.array:
    """
    In this scenario, the reference attitude is held at the initial attitude
    """
    sigma_RN =  MRP(0.1, 0.2, 0.3)
    dcm_R_N = RBK.MRP2C(sigma_RN.as_array())

    return dcm_R_N

def ReferenceAngularVelocity(t:float, dt:float=0.0001) -> np.array:
    """
    Approximate [NRc_dot] numerically by applying a small time step and evaluating the change in [NRc]
    """

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
    """
    Function for approximating N_omega_RN_dot
    """

    N_omega_RN_0 = ReferenceAngularVelocity(t-dt)
    N_omega_RN_1 = ReferenceAngularVelocity(t)

    N_omega_RN_dot = 1/dt*(N_omega_RN_1 - N_omega_RN_0)

    return N_omega_RN_dot

def ControlFunction(t, spacecraft:Spacecraft):
    """
    Attitude reference given as a function of time and angular velocity reference is estimated numerically
    
    This function implements a slightly more complicated control law that uses the gyroscopic terms of the reference
    angular velocity
    """
    pointing_mode = Mode.INVALID

    sigma_BN : np.array = spacecraft.state.sigma_BN.as_array()
    B_omega_BN : np.array = spacecraft.state.B_omega_BN
    # inertia = spacecraft.total_inertia

    K = spacecraft.control_gains.proportional
    P = spacecraft.control_gains.derivative

    # Calculate reference values to be tracked
    dcm_R_N = ReferenceAttitude(t)
    N_omega_RN = ReferenceAngularVelocity(t)
    N_omega_RN_dot = ReferenceAngularAcceleration(t)

    # Use the current attitude to determine the attitude and ang vel tracking error
    sigma_BR, B_omega_BR = CalculateAttitudeError(sigma_BN, B_omega_BN, dcm_R_N, N_omega_RN)

    # Convert ang vel and acceleration reference to body frame
    dcm_B_N = RBK.MRP2C(sigma_BN)
    B_omega_RN = np.matmul(dcm_B_N, N_omega_RN)
    B_omega_RN_dot = np.matmul(dcm_B_N, N_omega_RN_dot)

    # Calculate tilde matrix of omega_BN
    B_omega_BN_tilde = RBK.v3Tilde(B_omega_BN)

    inertia = spacecraft.total_inertia

    # Calculate control torque vector in body frame
    term1 = np.matmul(-K, sigma_BR.as_array())
    term2 = np.matmul(-P, B_omega_BR)
    term3 = np.matmul(inertia, (B_omega_RN_dot - np.matmul(B_omega_BN_tilde, B_omega_RN)))
    term4 = np.matmul(B_omega_BN_tilde, (np.matmul(inertia, B_omega_BN)))
    term5 = 0.0

    # Add inertia from each actuator and calculate wheel momentum
    for actuator, actuator_state in zip(spacecraft.actuators, spacecraft.state.actuator_states):
        if not isinstance(actuator, Vscmg):
            raise ValueError("Only VSCMGs are supported for this control law")

        wheel_speed = actuator_state.wheel_speed

        I_Ws = actuator.I_Ws

        dcm_BG = actuator._compute_gimbal_frame(actuator_state)
        B_ghat_s = dcm_BG[:,0]
        B_ghat_t = dcm_BG[:,1]
        B_ghat_g = dcm_BG[:,2]

        ws, wt, wg = actuator._compute_angular_velocity_gimbal_frame_projection(B_omega_BN=B_omega_BN, dcm_BG=dcm_BG)

        term5 += I_Ws * wheel_speed * (wg * B_ghat_t - wt * B_ghat_g)


    u = term1 + term2 + term3 + term4 + term5

    return u, pointing_mode, sigma_BR, B_omega_BR


if __name__ == "__main__":


    #######################
    # VSCMG configuration
    #######################
    # Define properties of all VSCMGs
    # Moment of inertia of gimbal + wheel system expressed in G frame (same for all VSCMGs)
    G_J = np.diag([0.13, 0.04, 0.03]) # [kgm^2]

    # Moment of inertia of wheel about spin axis (same for all VSCMGs)
    I_Ws = 0.1 # [kgm^2]

    # Initial wheel and gimbal rates for all VSCMGs
    wheel_speed_init = 14.4 # [rad/s]
    gimbal_rate_init = 0.0 # [rad/s]


    theta = math.radians(54.75)

    # VSCMG 1
    B_ghat_s_init_1 = np.array([0.,1.,0.])
    B_ghat_g_init_1 = np.array([math.cos(theta), 0., math.sin(theta)])
    B_ghat_t_init_1 = np.cross(B_ghat_g_init_1, B_ghat_s_init_1)
    dcm_GB_init_1 = np.array([B_ghat_s_init_1,
                            B_ghat_t_init_1,
                            B_ghat_g_init_1,])
    dcm_BG_init_1 = np.transpose(dcm_GB_init_1)

    gimbal_angle_init_1 = math.radians(0)    # [rad]
    init_state_vscmg_1 = VscmgState(wheel_speed=wheel_speed_init,
                                  gimbal_angle=gimbal_angle_init_1,
                                  gimbal_rate=gimbal_rate_init)

    # Create the VSCMG model
    sim_vscmg_1 = Vscmg(G_J=G_J,
                      I_Ws=I_Ws,
                      init_state=init_state_vscmg_1,
                      dcm_BG_init=dcm_BG_init_1,
                      gimbal_angle_init=gimbal_angle_init_1)
    
    # VSCMG 2
    B_ghat_s_init_2 = np.array([0.,-1.,0.])
    B_ghat_g_init_2 = np.array([-math.cos(theta), 0., math.sin(theta)])
    B_ghat_t_init_2 = np.cross(B_ghat_g_init_2, B_ghat_s_init_2)
    dcm_GB_init_2 = np.array([B_ghat_s_init_2,
                            B_ghat_t_init_2,
                            B_ghat_g_init_2,])
    dcm_BG_init_2 = np.transpose(dcm_GB_init_2)

    gimbal_angle_init_2 = math.radians(0)    # [rad]
    init_state_vscmg_2 = VscmgState(wheel_speed=wheel_speed_init,
                                  gimbal_angle=gimbal_angle_init_2,
                                  gimbal_rate=gimbal_rate_init)

    # Create the VSCMG model
    sim_vscmg_2 = Vscmg(G_J=G_J,
                      I_Ws=I_Ws,
                      init_state=init_state_vscmg_2,
                      dcm_BG_init=dcm_BG_init_2,
                      gimbal_angle_init=gimbal_angle_init_2)

    # VSCMG 3
    B_ghat_s_init_3 = np.array([1.,0.,0.])
    B_ghat_g_init_3 = np.array([0, math.cos(theta), math.sin(theta)])
    B_ghat_t_init_3 = np.cross(B_ghat_g_init_3, B_ghat_s_init_3)
    dcm_GB_init_3 = np.array([B_ghat_s_init_3,
                            B_ghat_t_init_3,
                            B_ghat_g_init_3,])
    dcm_BG_init_3 = np.transpose(dcm_GB_init_3)

    gimbal_angle_init_3 = math.radians(90)    # [rad]
    init_state_vscmg_3 = VscmgState(wheel_speed=wheel_speed_init,
                                  gimbal_angle=gimbal_angle_init_3,
                                  gimbal_rate=gimbal_rate_init)

    # Create the VSCMG model
    sim_vscmg_3 = Vscmg(G_J=G_J,
                      I_Ws=I_Ws,
                      init_state=init_state_vscmg_3,
                      dcm_BG_init=dcm_BG_init_3,
                      gimbal_angle_init=gimbal_angle_init_3)

    # VSCMG 4
    B_ghat_s_init_4 = np.array([-1.,0.,0.])
    B_ghat_g_init_4 = np.array([0, -math.cos(theta), math.sin(theta)])
    B_ghat_t_init_4 = np.cross(B_ghat_g_init_4, B_ghat_s_init_4)
    dcm_GB_init_4 = np.array([B_ghat_s_init_4,
                            B_ghat_t_init_4,
                            B_ghat_g_init_4,])
    dcm_BG_init_4 = np.transpose(dcm_GB_init_4)

    gimbal_angle_init_4 = math.radians(-90)    # [rad]
    init_state_vscmg_4 = VscmgState(wheel_speed=wheel_speed_init,
                                  gimbal_angle=gimbal_angle_init_4,
                                  gimbal_rate=gimbal_rate_init)

    # Create the VSCMG model
    sim_vscmg_4 = Vscmg(G_J=G_J,
                      I_Ws=I_Ws,
                      init_state=init_state_vscmg_4,
                      dcm_BG_init=dcm_BG_init_4,
                      gimbal_angle_init=gimbal_angle_init_4)


    vscmg_init_states = [init_state_vscmg_1, init_state_vscmg_2, init_state_vscmg_3, init_state_vscmg_4]
    sim_vscmgs = [sim_vscmg_1, sim_vscmg_2, sim_vscmg_3, sim_vscmg_4]


    ############################
    # Spacecraft configuration
    ############################
    # Moment of inertia of spacecraft hub expressed in body frame 
    B_Is = np.diag([86., 85., 113.]) # [kgm^2]

    # Initial attitude states
    sigma_BN_init = MRP(0.1, 0.2, 0.3)
    B_omega_BN_init = np.array([0.0, 0.0, 0.0]) # [rad/s]

    init_state_sc = SpacecraftState(sigma_BN=sigma_BN_init,
                            B_omega_BN=B_omega_BN_init,
                            actuator_states=vscmg_init_states
                            )

    # Define control gains
    K = np.diag([5., 5., 5.])  # Proportional gain
    P = np.diag([15., 15., 15.])  # Derivative gain
    control_gains = ControlGains(proportional=K, derivative=P)

    sim_spacecraft = Spacecraft(B_Is=B_Is,
                            init_state=init_state_sc,
                            actuators=sim_vscmgs,
                            control_gains=control_gains,
                            use_vscmg_null_motion=True,
                            null_motion_correction_gain=0.1)


    # Define integrator and integration properties
    dt = 0.1 # [s]
    t_init = 0 # [s]
    t_final = 600 # [s]

    # Define zero external torque function
    def zero_external_torque(t:float, spacecraft:Spacecraft) -> np.array:
        return np.array([0.0, 0.0, 0.0])


    # Define equations of motion
    eom = ControlledSpacecraftDynamics(spacecraft=sim_spacecraft,
                                      control_law=ControlFunction)


    # Run simulation
    solution = eom.simulate(t_init=t_init,
                                t_max=t_final,
                                t_step=dt,
                                torque_eq=zero_external_torque)

    # Plot results
    # Extract the log data
    sigma_BN_list = solution["MRP"]
    B_omega_BN_list = solution["omega_B_N"]

    wheel_speed_dict = {}
    gimbal_angle_dict = {}
    gimbal_rate_dict = {}
    gimbal_torque_dict = {}
    wheel_torque_dict = {}
    for vidx, actuator in enumerate(sim_spacecraft.actuators):
        wheel_speed_dict.update({vidx: solution[f"wheel_speed_{vidx}"]})
        if isinstance(actuator, Vscmg):
            gimbal_angle_dict.update({vidx: solution[f"gimbal_angle_{vidx}"]})
            gimbal_rate_dict.update({vidx: solution[f"gimbal_rate_{vidx}"]})
            gimbal_torque_dict.update({vidx: solution[f"gimbal_torque_{vidx}"]})
            wheel_torque_dict.update({vidx: solution[f"wheel_torque_{vidx}"]})

    t_list = solution["time"]
    energy_list = solution["total_energy"]
    power_list = solution["total_power"]
    H_list = solution["N_H_total"]

    # Print the values we're interested in
    print("{Simulation Results}")
    for idx in range(len(t_list)):
        sigma_BN = sigma_BN_list[idx]
        B_omega_BN = B_omega_BN_list[idx]
        t = t_list[idx]
        energy = energy_list[idx]
        power = power_list[idx]
        H_total = H_list[idx]

        wheel_speeds = [wheel_speed_dict[vidx][idx] for vidx in range(len(sim_spacecraft.actuators))]
        if isinstance(actuator, Vscmg):
            gimbal_angles = [gimbal_angle_dict[vidx][idx] for vidx in range(len(sim_spacecraft.actuators))]
            gimbal_rates = [gimbal_rate_dict[vidx][idx] for vidx in range(len(sim_spacecraft.actuators))]

        if (abs(t-0.0) < 1e-6) or (abs(t-10.0) < 1e-6) or (abs(t-30.0) < 1e-6) or (abs(t-t_final) < 1e-6):
            print(f"> Time: {t}")
            print(f"  > energy: {energy}")
            print(f"  > power: {power}")
            print(f"  > N_H: {H_total}")
            print(f"  > sigma_BN: {sigma_BN}")
            print(f"  > B_omega_BN: {B_omega_BN}")
            print(f"  > Omegas: {wheel_speeds}")
            if isinstance(actuator, Vscmg):
                print(f"  > gammas: {gimbal_angles}")
                print(f"  > gamma_dot: {gimbal_rates}")
            
    
    plots.PlotMrpAndOmegaComponents(sigma_BN_list, 
                                    B_omega_BN_list, 
                                    t_list, 
                                    title='Attitude and Angular Velocity in Simulation')
    
    plots.PlotWheelSpeeds(wheel_speed_dict, 
                            t_list, 
                            title='Wheel Speeds During Simulation')
    plots.PlotGimbalAngles(gimbal_angle_dict, 
                            t_list, 
                            title='Gimbal Angles During Simulation')
    plots.PlotGimbalTorques(gimbal_torque_dict,
                            t_list,
                            title='Gimbal Torques During Simulation')
    plots.PlotWheelTorques(wheel_torque_dict,
                            t_list,
                            title='Wheel Torques During Simulation')