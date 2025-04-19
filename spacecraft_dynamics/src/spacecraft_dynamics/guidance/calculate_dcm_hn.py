import numpy as np
import math

from ..utils import constants as constants
from ..orbit import InertialPositionVelocity


def CalculateDcmHN(t):
    """
    Calculate the DCM from inertial frame to Hill frame at a given point in time for the 
    LMO spacecraft. Note this function is only meant to be used for the LMO spacecraft 
    as it relies on the initial conditions of the spacecraft

    :param t: elapsed time from the initial conditions of the LMO spacecraft [s]
    :return dcm_H_N: numpy array representing the DCM that describes the rotation
                    from the inertial frame to the Hill frame of the LMO spacecraft
                    at time t
    """

    # LMO initial parameters
    Om_LMO = np.deg2rad(20.0)
    inc_LMO = np.deg2rad(30.0)
    theta_LMO_0 = np.deg2rad(60.0)
    theta_dot_LMO = 0.000884797 #[rad/s]
    h_LMO = 400.0*1000  # [m]

    # Integrate theta to time t
    theta_LMO = theta_LMO_0 + t*theta_dot_LMO
    theta_LMO = theta_LMO%(2*math.pi)

    # Set R and 3-1-3 angles
    r_LMO = h_LMO + constants.R_MARS
    angles313_LMO = np.array([Om_LMO, inc_LMO, theta_LMO])

    # Calculate the inertial position and velocity in inertial frame components [m, m/s]
    N_pos_LMO, N_vel_LMO = InertialPositionVelocity(r_LMO, angles313_LMO)

    # Calcualte the basis vectors of the Hill frame in inertial frame components
    N_i_r_hat = N_pos_LMO/np.linalg.norm(N_pos_LMO)

    N_i_h_hat = np.cross(N_pos_LMO, N_vel_LMO)/np.linalg.norm(np.cross(N_pos_LMO, N_vel_LMO))

    N_i_t_hat = np.cross(N_i_h_hat, N_i_r_hat)

    # Construct the [HN] DCM
    # Note that [NH] = {N_ir, N_it, N_ih} (column vectors) so the transpose [NH]
    # can be created with row vectors
    dcm_H_N = np.array([N_i_r_hat,
                        N_i_t_hat,
                        N_i_h_hat])
    
    return dcm_H_N


if __name__ == "__main__":

    ##### TASK 2
    # Test function by evaluating at t=300s

    dcm_H_N = CalculateDcmHN(300)
    print("dcm_H_N: \n{}".format(dcm_H_N))