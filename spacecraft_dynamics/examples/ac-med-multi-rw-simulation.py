import numpy as np

from spacecraft_dynamics.actuators import (
    Vscmg,
    ReactionWheel
)
from spacecraft_dynamics.core import (
    Spacecraft, 
    ControlGains,
    ControlledSpacecraftDynamics
)
from spacecraft_dynamics.state import (
    SpacecraftState, 
    ReactionWheelState
)
from spacecraft_dynamics.utils import MRP
import spacecraft_dynamics.utils.rigid_body_kinematics as RBK
from spacecraft_dynamics.control import CalculateAttitudeError
from spacecraft_dynamics.guidance import Mode

import spacecraft_dynamics.analysis.plots as plots

def ReferenceAttitude(t:float, f:float = 0.03) -> np.array:
    """
    Calcualte reference attitude [RN] given as a function of time
    """
    sigma_RN = MRP(0.25 * 0.1 * np.sin(f * t), 
                   0.25 * 0.2 * np.cos(f * t), 
                   0.25 * -0.3 * np.sin(2.0 * f * t))
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

    # if sigma_BR.norm() > 1.0:
    #     sigma_BR = sigma_BR.convert_to_shadow_set()

    # Convert ang vel and acceleration reference to body frame
    dcm_B_N = RBK.MRP2C(sigma_BN)
    B_omega_RN = np.matmul(dcm_B_N, N_omega_RN)
    B_omega_RN_dot = np.matmul(dcm_B_N, N_omega_RN_dot)

    if t == 0.0:
        print(f"> t: {t}")
        print(f"> sigma_RN: {RBK.C2MRP(dcm_R_N)}")
        print(f"> B_omega_RN: {B_omega_RN}")

    # Calculate tilde matrix of omega_BN
    B_omega_BN_tilde = RBK.v3Tilde(B_omega_BN)

    # Construct the [Gs] matrix where each column is the spin axis for each wheel
    num_wheels = len(spacecraft.actuators)
    Gs = np.zeros((3, num_wheels))
        
    for i, actuator in enumerate(spacecraft.actuators):
        if not isinstance(actuator, ReactionWheel):
            raise ValueError("Only reaction wheels are supported for this control law")
        
        Gs[:, i] = actuator.spin_axis

    # Construct wheel momentum vector
    h_s = np.zeros((num_wheels, 1))

    inertia = spacecraft.B_Is

    # Add inertia from each actuator and calculate wheel momentum
    for i, (actuator, actuator_state) in enumerate(zip(spacecraft.actuators, spacecraft.state.actuator_states)):
        if not isinstance(actuator, ReactionWheel):
            raise ValueError("Only reaction wheels are supported for this control law")

        dcm_BG = actuator._compute_gimbal_frame(actuator_state)
        B_ghat_s = dcm_BG[:,0]
        B_ghat_t = dcm_BG[:,1]
        B_ghat_g = dcm_BG[:,2]

        Jt = actuator.G_J[1,1]
        Jg = actuator.G_J[2,2]

        inertia += Jt * np.outer(B_ghat_t, B_ghat_t) + Jg * np.outer(B_ghat_g, B_ghat_g)

        # Compute angular velocity projection onto wheel spin axis
        ws = np.dot(B_ghat_s, B_omega_BN)

        h_s[i] = actuator.I_Ws * (ws + actuator.state.wheel_speed)

    # Calculate control torque vector in body frame
    term1 = np.matmul(-K, sigma_BR.as_array())
    term2 = np.matmul(-P, B_omega_BR)
    term3 = np.matmul(inertia, (B_omega_RN_dot - np.matmul(B_omega_BN_tilde, B_omega_RN)))
    term4 = np.matmul(B_omega_BN_tilde, (np.matmul(inertia, B_omega_BN) + np.matmul(Gs, h_s.flatten())))
    
    u = term1 + term2 + term3 + term4

    return u, pointing_mode, sigma_BR, B_omega_BR



if __name__ == "__main__":
    # Reaction Wheel configuration
    # Moment of inertia of wheel about spin axis (same for all wheels)
    I_Ws = 0.1  # [kgm^2]

    # In this scenario, the inertia is given as [I_RW], the inertia of the body plus the 
    # inertia of the reaction wheels so the inertia of the wheel assembly is left nulled
    G_J = np.diag([0., 0., 0.])  # [kgm^2]

    # Initial wheel speeds
    wheel_speed_init = 0.0  # [rad/s]

    # Define the spin axes for each reaction wheel
    spin_axis_1 = np.array([1., 0., 0.])
    spin_axis_2 = np.array([0., 1., 0.])
    spin_axis_3 = np.array([0., 0., 1.])
    spin_axis_4 = np.array([1., 1., 1.]) / np.sqrt(3)

    # Create reaction wheel states and objects
    init_state_rw_1 = ReactionWheelState(wheel_speed=wheel_speed_init)
    init_state_rw_2 = ReactionWheelState(wheel_speed=wheel_speed_init)
    init_state_rw_3 = ReactionWheelState(wheel_speed=wheel_speed_init)
    init_state_rw_4 = ReactionWheelState(wheel_speed=wheel_speed_init)

    # Create the reaction wheel models
    sim_rw_1 = ReactionWheel(I_Ws=I_Ws, init_state=init_state_rw_1, spin_axis=spin_axis_1, G_J=G_J)
    sim_rw_2 = ReactionWheel(I_Ws=I_Ws, init_state=init_state_rw_2, spin_axis=spin_axis_2, G_J=G_J)
    sim_rw_3 = ReactionWheel(I_Ws=I_Ws, init_state=init_state_rw_3, spin_axis=spin_axis_3, G_J=G_J)
    sim_rw_4 = ReactionWheel(I_Ws=I_Ws, init_state=init_state_rw_4, spin_axis=spin_axis_4, G_J=G_J)

    rw_init_states = [init_state_rw_1, init_state_rw_2, init_state_rw_3, init_state_rw_4]
    sim_rws = [sim_rw_1, sim_rw_2, sim_rw_3, sim_rw_4]

    # Spacecraft configuration
    # Moment of inertia of spacecraft hub + reaction wheels expressed in body frame 
    B_Is = np.diag([86., 85., 113.])  # [kgm^2]

    # Initial attitude states
    sigma_BN_init = MRP(0.1, 0.2, 0.3)
    B_omega_BN_init = np.array([0.01, -0.01, 0.005])  # [rad/s]

    init_state_sc = SpacecraftState(sigma_BN=sigma_BN_init,
                                  B_omega_BN=B_omega_BN_init,
                                  actuator_states=rw_init_states)

    # Define control gains
    K = np.diag([5., 5., 5.])  # Proportional gain
    P = np.diag([15., 15., 15.])  # Derivative gain
    control_gains = ControlGains(proportional=K, derivative=P)

    sim_spacecraft = Spacecraft(B_Is=B_Is,
                              init_state=init_state_sc,
                              actuators=sim_rws,
                              control_gains=control_gains)

    # Define zero external torque function
    def zero_external_torque(t:float, spacecraft:Spacecraft) -> np.array:
        return np.array([0.0, 0.0, 0.0])

    # Define equations of motion
    eom = ControlledSpacecraftDynamics(spacecraft=sim_spacecraft,
                                      control_law=ControlFunction)

    # Define integrator and integration properties
    dt = 0.1 # [s]
    t_init = 0.0 # [s]
    t_final = 120 # [s]

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
    for vidx, actuator in enumerate(sim_spacecraft.actuators):
        wheel_speed_dict.update({vidx: solution[f"wheel_speed_{vidx}"]})
        if isinstance(actuator, Vscmg):
            gimbal_angle_dict.update({vidx: solution[f"gimbal_angle_{vidx}"]})
            gimbal_rate_dict.update({vidx: solution[f"gimbal_rate_{vidx}"]})

    t_list = solution["time"]
    energy_list = solution["total_energy"]
    power_list = solution["total_power"]
    H_list = solution["N_H_total"]
    control_torque_list = solution["control_torque"]
    sigma_BR_list = solution["sigma_BR"]
    B_omega_BR_list = solution["B_omega_BR"]
    
    # Print the values we're interested in
    print("{Simulation Results}")
    for idx in range(len(t_list)):
        sigma_BN = sigma_BN_list[idx]
        B_omega_BN = B_omega_BN_list[idx]
        t = t_list[idx]
        energy = energy_list[idx]
        power = power_list[idx]
        H_total = H_list[idx]
        control_torque = control_torque_list[idx]
        wheel_speeds = [wheel_speed_dict[vidx][idx] for vidx in range(len(sim_spacecraft.actuators))]
        if isinstance(actuator, Vscmg):
            gimbal_angles = [gimbal_angle_dict[vidx][idx] for vidx in range(len(sim_spacecraft.actuators))]
            gimbal_rates = [gimbal_rate_dict[vidx][idx] for vidx in range(len(sim_spacecraft.actuators))]

        if abs((t-0.0) < 1e-6) or (abs(t-10.0) < 1e-6) or (abs(t-30.0) < 1e-6) or (abs(t-120.0) < 1e-6):
            print(f"> Time: {t}")
            print(f"  > energy: {energy}")
            print(f"  > power: {power}")
            print(f"  > N_H: {H_total}")
            print(f"  > sigma_BN: {sigma_BN}")
            print(f"  > B_omega_BN: {B_omega_BN}")
            print(f"  > Omegas: {wheel_speeds}")
            print(f"  > control_torque: {control_torque}")
            if isinstance(actuator, Vscmg):
                print(f"  > gammas: {gimbal_angles}")
            
    
    plots.PlotMrpAndOmegaComponents(sigma_BN_list, 
                                    B_omega_BN_list, 
                                    t_list, 
                                    title='Attitude and Angular Velocity in Simulation')
    plots.PlotMrpAndOmegaComponents(sigma_BR_list, B_omega_BR_list, t_list, title='Attitude and Angular Velocity Error in Simulation')
    plots.PlotMrpAndOmegaNorms(sigma_BR_list, B_omega_BR_list, t_list, title = 'Evolution of $|\sigma_{B/R}|$ and $|\omega_{B/R}|$ in Full Mission Simulation')
    plots.PlotTorqueComponents(control_torque_list, t_list, title='Control Torque Components in Full Mission Simulation')