import numpy as np
import math

from spacecraft_dynamics.actuators import Vscmg
from spacecraft_dynamics.core import (
    Spacecraft, 
    TorqueFreeSpacecraftDynamics
)
from spacecraft_dynamics.state import (
    SpacecraftState, 
    VscmgState
)
from spacecraft_dynamics.utils import MRP

import spacecraft_dynamics.analysis.plots as plots

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
    B_omega_BN_init = np.array([0.01, -0.01, 0.005]) # [rad/s]

    init_state_sc = SpacecraftState(sigma_BN=sigma_BN_init,
                            B_omega_BN=B_omega_BN_init,
                            actuator_states=vscmg_init_states
                            )

    sim_spacecraft = Spacecraft(B_Is=B_Is,
                            init_state=init_state_sc,
                            actuators=sim_vscmgs)
    


    # Define equations of motion
    vscmg_eom = TorqueFreeSpacecraftDynamics(spacecraft=sim_spacecraft)

    # Define integrator and integration properties
    dt = 0.1 # [s]
    t_init = 0.0 # [s]
    t_final = 40 # [s]

    # TODO: move ext torque to Spacecraft class?
    def constant_external_torque(t:float,
                                spacecraft:Spacecraft) -> np.array:
        """
        Calculate a constant external torque acting on a spacecraft

        Arguments:
            t (float): Simulation time
            spacecraft (Spacecraft): Spacecraft that the torque is acting upon

        Retuns:
            ndarray: Torque vector acting on the spacecraft (in spacecraft body frame) [Nms]
        """

        B_L = np.array([0.1, 0.1, -0.1])
        return B_L


    # Run simulation
    solution = vscmg_eom.simulate(t_init=t_init,
                                t_max=t_final,
                                t_step=dt,
                                torque_eq=constant_external_torque)

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

        if (abs(t-10.0) < 1e-6) or (abs(t-30.0) < 1e-6) or (abs(t-40.0) < 1e-6):
            print(f"> Time: {t}")
            print(f"  > energy: {energy}")
            print(f"  > power: {power}")
            print(f"  > N_H: {H_total}")
            print(f"  > sigma_BN: {sigma_BN}")
            print(f"  > B_omega_BN: {B_omega_BN}")
            print(f"  > Omegas: {wheel_speeds}")
            if isinstance(actuator, Vscmg):
                print(f"  > gammas: {gimbal_angles}")
            
    
    plots.PlotMrpAndOmegaComponents(sigma_BN_list, 
                                    B_omega_BN_list, 
                                    t_list, 
                                    title='Attitude and Angular Velocity in Simulation')