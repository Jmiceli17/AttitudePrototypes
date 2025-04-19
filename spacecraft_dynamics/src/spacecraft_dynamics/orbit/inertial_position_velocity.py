import numpy as np
import math

from ..utils import rigid_body_kinematics as RBK
from ..utils import constants as constants

def InertialPositionVelocity(r:float, angles_313):    
    """
    Using 3-1-3 euler angles and a hill-frame position r ihat_r,
    calculate the corresponding inertial components of position and 
    velocity 

    Note: this function assumes the spacecraft is in a circular orbit around Mars

    :param r: radius of orbit (m)
    :param angles_313: 3-1-3 Euler angles describing orientation of Hill frame wrt N frame (rad)
    :return N_position: Inertial position of spacecraft in inertial frame components
    :return N_velocity: Inertial velocity of spacecraft in inertial frame components
    """
    # # Standard gravitational param for mars [m^3/s^2]
    # MU_MARS = 4.28283e13 

    # Position vector in Hill frame components [m]
    H_position = np.array([r, 0.0, 0.0])

    # Orbit rate for circular orbit around mars [rad/s]
    theta_dot = np.sqrt(constants.MU_MARS/(r**3))

    # Inertial frame relative derivative of position expressed in Hill frame components
    H_velocity = np.array([0.0, r*theta_dot, 0.0])


    # Get the DCM from inertial to Hill frame using the provided 3-1-3 Euler angles
    dcm_H_N = RBK.euler3132C(angles_313)
    dcm_N_H = np.transpose(dcm_H_N)

    # Position vector in inertial frame components [m]
    N_position = np.matmul(dcm_N_H, H_position)

    # Velocity vector in inertial frame components [m]
    N_velocity = np.matmul(dcm_N_H, H_velocity)


    return N_position, N_velocity

if __name__ == "__main__":

    ##### TASK 1
    # Test function using values from project

    # R_MARS = 3396.19*1000  # [m]

    # LMO initial parameters
    Om_LMO = np.deg2rad(20.0)
    inc_LMO = np.deg2rad(30.0)
    theta_LMO = np.deg2rad(60.0)
    # theta_dot_LMO = 0.000884797 #[rad/s]
    theta_dot_LMO = constants.ORBIT_RATE_LMO
    h_LMO = 400.0*1000  # [m]

    # Integrate theta to 450s
    theta_LMO = theta_LMO + 450*theta_dot_LMO
    theta_LMO = theta_LMO%(2*math.pi)

    # Set R and 3-1-3 angles
    r_LMO = h_LMO + constants.R_MARS
    angles313_LMO = np.array([Om_LMO, inc_LMO, theta_LMO])

    N_pos_LMO, N_vel_LMO = InertialPositionVelocity(r_LMO, angles313_LMO)
    
    # Convert to km
    N_pos_LMO = N_pos_LMO/1000
    N_vel_LMO = N_vel_LMO/1000

    print("N_pos_LMO: {} [km]".format(N_pos_LMO))
    print("N_vel_LMO: {} [km/s]".format(N_vel_LMO))

    # GMO initial parameters
    Om_GMO = np.deg2rad(0.0)
    inc_GMO = np.deg2rad(0.0)
    theta_GMO = np.deg2rad(250.0)
    theta_dot_GMO = 0.0000709003
    h_GMO = 17028.01*1000  # [m]

    # Integrate theta to 1150s
    theta_GMO = theta_GMO + 1150*theta_dot_GMO
    theta_GMO = theta_GMO%(2*math.pi)   # Wrap to 2pi

    # Set R and 3-1-3 angles
    r_GMO = h_GMO + constants.R_MARS
    angles313_GMO = np.array([Om_GMO, inc_GMO, theta_GMO])

    N_pos_GMO, N_vel_GMO = InertialPositionVelocity(r_GMO, angles313_GMO)
    
    # Convert to km
    N_pos_GMO = N_pos_GMO/1000
    N_vel_GMO = N_vel_GMO/1000
    
    print("N_pos_GMO: {} [km]".format(N_pos_GMO))
    print("N_vel_GMO: {} [km/s]".format(N_vel_GMO))