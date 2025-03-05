import numpy as np
import math

from State import StateVscmg
from ModifiedRodriguesParameters import MRP
from EquationsOfMotion import VscmgDynamics # RungeKutta

import Plots

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

    # Initial attitude states
    sigma_BN_init = MRP(0.1, 0.2, 0.3)
    B_omega_BN_init = np.array([0.01, -0.01, 0.005]) # [rad/s]

    state_init = StateVscmg(sigma_BN=sigma_BN_init,
                            B_omega_BN=B_omega_BN_init,
                            wheel_speed=wheel_speed_init,
                            gimbal_angle=gimbal_angle_init,
                            gimbal_rate=gimbal_rate_init
                            )

    # Define equations of motion
    vscmg_eom = VscmgDynamics(B_Is=B_Is,
                            G_J=G_J,
                            I_Ws=I_Ws,
                            dcm_BG_init=dcm_BG_init,
                            gimbal_angle_init=gimbal_angle_init)

    # Define integrator and integration properties
    dt = 0.1 # [s]
    t_init = 0.0 # [s]
    t_final = 30 # [s]

    # Run simulation
    solution = vscmg_eom.simulate(init_state=state_init,
                        t_init=t_init,
                        t_max=t_final,
                        t_step=dt)

    # Plot results
    # Extract the log data
    sigma_BN_list = solution["MRP"]
    B_omega_BN_list = solution["omega_B_N"]
    wheel_speed_list = solution["wheel_speed"]
    gimbal_angle_list = solution["gimbal_angle"]
    gimbal_rate_list = solution["gimbal_rate"]
    t_list = solution["time"]
    energy_list = solution["total_energy"]
    H_list = solution["N_H_total"]

    # Print the values we're interested in
    print("{Simulation Results}")
    for idx in range(len(t_list)):
        sigma_BN = sigma_BN_list[idx]
        B_omega_BN = B_omega_BN_list[idx]
        wheel_speed = wheel_speed_list[idx]
        gimbal_angle = gimbal_angle_list[idx]
        gimbal_rate = gimbal_rate_list[idx]
        t = t_list[idx]
        energy = energy_list[idx]
        H_total = H_list[idx]

        if (abs(t-10.0) < 1e-6) or (abs(t-30.0) < 1e-6):
            print(f"> Time: {t}")
            print(f"  > energy: {energy}")
            print(f"  > N_H: {H_total}")
            print(f"  > sigma_BN: {sigma_BN}")
            print(f"  > B_omega_BN: {B_omega_BN}")
            print(f"  > Omega: {wheel_speed}")
            print(f"  > gamma: {gimbal_angle}")
            
    
    Plots.PlotMrpAndOmegaComponents(sigma_BN_list, 
                                    B_omega_BN_list, 
                                    t_list, 
                                    title='Attitude and Angular Velocity in Simulation')