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

    # Define properties
    # Moment of inertia of spacecraft hub expressed in body frame 
    B_Is = np.diag([86., 85., 113.]) # [kgm^2]

    # Moment of inertia of gimbal + wheel system expressed in G frame
    G_J = np.diag([0.13, 0.04, 0.03]) # [kgm^2]

    # Moment of inertia of wheel about spin axis
    I_Ws = 0.1 # [kgm^2]

    # Define initial state
    # Initial BG DCM
    theta = math.radians(54.75)
    B_ghat_s_init = np.array([0.,1.,0.])
    B_ghat_g_init = np.array([math.cos(theta), 0., math.sin(theta)])
    B_ghat_t_init = np.cross(B_ghat_g_init, B_ghat_s_init)
    dcm_GB_init = np.array([B_ghat_s_init,
                            B_ghat_t_init,
                            B_ghat_g_init,])
    dcm_BG_init = np.transpose(dcm_GB_init)

    # Initial wheel and gimbal states
    wheel_speed_init = 14.4 # [rad/s]
    gimbal_angle_init = 0.0 # [rad]
    gimbal_rate_init = 0.0 # [rad/s]

    init_state_vscmg = VscmgState(wheel_speed=wheel_speed_init,
                                  gimbal_angle=gimbal_angle_init,
                                  gimbal_rate=gimbal_rate_init)

    sim_vscmg = Vscmg(G_J=G_J,
                      I_Ws=I_Ws,
                      init_state=init_state_vscmg,
                      dcm_BG_init=dcm_BG_init,
                      gimbal_angle_init=gimbal_angle_init)

    # Initial attitude states
    sigma_BN_init = MRP(0.1, 0.2, 0.3)
    B_omega_BN_init = np.array([0.01, -0.01, 0.005]) # [rad/s]

    init_state_sc = SpacecraftState(sigma_BN=sigma_BN_init,
                            B_omega_BN=B_omega_BN_init,
                            actuator_states=[init_state_vscmg]
                            )

    sim_spacecraft = Spacecraft(B_Is=B_Is,
                            init_state=init_state_sc,
                            actuators=[sim_vscmg])

    # Define equations of motion
    vscmg_eom = TorqueFreeSpacecraftDynamics(spacecraft=sim_spacecraft)

    # Define integrator and integration properties
    dt = 0.1 # [s]
    t_init = 0.0 # [s]
    t_final = 30 # [s]

    # Run simulation
    solution = vscmg_eom.simulate(t_init=t_init,
                                t_max=t_final,
                                t_step=dt)

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
    H_list = solution["N_H_total"]

    # Print the values we're interested in
    print("{Simulation Results}")
    for idx in range(len(t_list)):
        sigma_BN = sigma_BN_list[idx]
        B_omega_BN = B_omega_BN_list[idx]
        t = t_list[idx]
        energy = energy_list[idx]
        H_total = H_list[idx]

        wheel_speeds = [wheel_speed_dict[vidx][idx] for vidx in range(len(sim_spacecraft.actuators))]
        if isinstance(actuator, Vscmg):
            gimbal_angles = [gimbal_angle_dict[vidx][idx] for vidx in range(len(sim_spacecraft.actuators))]
            gimbal_rates = [gimbal_rate_dict[vidx][idx] for vidx in range(len(sim_spacecraft.actuators))]

        if (abs(t-10.0) < 1e-6) or (abs(t-30.0) < 1e-6):
            print(f"> Time: {t}")
            print(f"  > energy: {energy}")
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