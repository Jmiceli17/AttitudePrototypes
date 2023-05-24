import numpy as np
from enum import Enum
import math

from ReferenceAttitudes import (GetSunPointingReferenceAttitude,
                                GetSunPointingReferenceAngularVelocity,
                                GetNadirPointingReferenceAttitude,
                                GetNadirPointingReferenceAngularVelocity,
                                GetGmoPointingReferenceAttitude,
                                GetGmoPointingReferenceAngularVelocity)
from AttitudeError import CalculateAttitudeError
from PDControl import PDControl
from InertialPositionVelocity import InertialPositionVelocity
import InitialConditions as IC
import constants

class Mode(Enum):
    INVALID = -1 
    SUN_POINTING = 0
    NADIR_POINTING = 1
    GMO_POINTING = 2

def SunPointingControl(t, state, gains = (np.eye(3),np.eye(3))):
    """
    Sun pointing guidance function to calculate control torques using the sun pointing
    reference attitude and angular velocity

    :param t: time
    :param state: 2x3 array containing the sigma_BN and B_omega_BN of the spacecraft
    :param gains: Tuple containing K and P gains for PD controller
    :return u: control torque vector in body frame components [Nm]
    :return pointing_mode: Enum defining the current pointing mode
    :return sigma_BR: Reference attitude tracking error MRP
    :return B_omega_BR: Reference angular velocity tracking error in body frame components [rad/s]
    """
    pointing_mode = Mode.SUN_POINTING

    sigma_BN = state[0]
    B_omega_BN = state[1]
    K = gains[0]
    P = gains[1]

    # Get reference attitude and angular velocity
    dcm_Rs_N = GetSunPointingReferenceAttitude(t)
    N_omega_RN = GetSunPointingReferenceAngularVelocity(t)

    # Use the current attitude to determine the attitude and ang vel tracking error
    sigma_BR, B_omega_BR = CalculateAttitudeError(sigma_BN, B_omega_BN, dcm_Rs_N, N_omega_RN)

    # print("[SunPointingControl] \n> sigma_BR: {}".format(sigma_BR))

    # Calculate control torques using tracking error
    u = PDControl(sigma_BR, B_omega_BR, K, P)

    return u, pointing_mode, sigma_BR, B_omega_BR

def NadirPointingControl(t, state, gains = (np.eye(3),np.eye(3))):
    """
    Nadir pointing guidance function to calculate control torques using the nadir pointing
    reference attitude and angular velocity

    :param t: time
    :param state: 2x3 array containing the sigma_BN and B_omega_BN of the spacecraft
    :param gains: Tuple containing K and P gains for PD controller
    :return u: control torque vector in body frame components [Nm]
    :return pointing_mode: Enum defining the current pointing mode
    :return sigma_BR: Reference attitude tracking error MRP
    :return B_omega_BR: Reference angular velocity tracking error in body frame components [rad/s]
    """
    pointing_mode = Mode.NADIR_POINTING

    sigma_BN = state[0]
    B_omega_BN = state[1]
    K = gains[0]
    P = gains[1]

    # Get reference attitude and angular velocity
    dcm_Rn_N = GetNadirPointingReferenceAttitude(t)
    N_omega_RN = GetNadirPointingReferenceAngularVelocity(t)

    # Use the current attitude to determine the attitude and ang vel tracking error
    sigma_BR, B_omega_BR = CalculateAttitudeError(sigma_BN, B_omega_BN, dcm_Rn_N, N_omega_RN)

    # Calculate control torques using tracking error
    u = PDControl(sigma_BR, B_omega_BR, K, P)

    return u, pointing_mode, sigma_BR, B_omega_BR

def GmoPointingControl(t, state, gains = (np.eye(3),np.eye(3))):
    """
    GMO pointing guidance function to calculate control torques using the GMO pointing
    reference attitude and angular velocity

    :param t: time
    :param state: 2x3 array containing the sigma_BN and B_omega_BN of the spacecraft
    :param gains: Tuple containing K and P gains for PD controller
    :return u: control torque vector in body frame components [Nm]
    :return pointing_mode: Enum defining the current pointing mode
    :return sigma_BR: Reference attitude tracking error MRP
    :return B_omega_BR: Reference angular velocity tracking error in body frame components [rad/s]
    """
    pointing_mode = Mode.GMO_POINTING

    sigma_BN = state[0]
    B_omega_BN = state[1]
    K = gains[0]
    P = gains[1]

    # Get reference attitude and angular velocity
    dcm_Rg_N = GetGmoPointingReferenceAttitude(t)
    N_omega_RN = GetGmoPointingReferenceAngularVelocity(t)

    # Use the current attitude to determine the attitude and ang vel tracking error
    sigma_BR, B_omega_BR = CalculateAttitudeError(sigma_BN, B_omega_BN, dcm_Rg_N, N_omega_RN)

    # Calculate control torques using tracking error
    u = PDControl(sigma_BR, B_omega_BR, K, P)

    return u, pointing_mode, sigma_BR, B_omega_BR
        

def MissionPointingControl(t, state, gains = (np.eye(3),np.eye(3))):
    """
    Mission pointing guidance function to calculate control torques using the 
    reference attitude and angular velocity calculated from whatever the current pointing mode is
    This function essentially combines all other pointing control functions and adds logic to switch 
    between them according to mission requirements

    :param t: time
    :param state: 2x3 array containing the sigma_BN and B_omega_BN of the spacecraft
    :param gains: Tuple containing K and P gains for PD controller
    :return u: control torque vector in body frame components [Nm]
    :return pointing_mode: Enum defining the current pointing mode
    :return sigma_BR: Reference attitude tracking error MRP
    :return B_omega_BR: Reference angular velocity tracking error in body frame components [rad/s]
    """

    # The reference attitude and angular velocity depends on the position of 
    # the LMO spacecraft and the GMO spacecraft so the first step is to determine
    # those vectors
    
    # Determine inertial position of LMO spacecaft
    # LMO initial parameters
    Om_LMO = IC.LMO_OMEGA_0_RAD
    inc_LMO = IC.LMO_INC_0_RAD
    theta_LMO = IC.LMO_THETA_0_RAD
    theta_dot_LMO = IC.LMO_ORBIT_RATE
    h_LMO = IC.LMO_ALT_M

    # Integrate theta to current time
    theta_LMO = theta_LMO + t*theta_dot_LMO
    theta_LMO = theta_LMO%(2*math.pi)

    # Set R and 3-1-3 angles
    r_LMO = h_LMO + constants.R_MARS
    angles313_LMO = np.array([Om_LMO, inc_LMO, theta_LMO])

    # Determine inertial position of LMO spacecaft
    N_pos_LMO, N_vel_LMO = InertialPositionVelocity(r_LMO, angles313_LMO)
    
    # GMO initial parameters
    Om_GMO = IC.GMO_OMEGA_0_RAD
    inc_GMO = IC.GMO_INC_0_RAD
    theta_GMO = IC.GMO_THETA_0_RAD
    theta_dot_GMO = IC.GMO_ORBIT_RATE
    h_GMO = IC.GMO_ALT_M

    # Integrate theta to current time
    theta_GMO = theta_GMO + t*theta_dot_GMO
    theta_GMO = theta_GMO%(2*math.pi)   # Wrap to 2pi

    # Set R and 3-1-3 angles
    r_GMO = h_GMO + constants.R_MARS
    angles313_GMO = np.array([Om_GMO, inc_GMO, theta_GMO])
    
    # Determine inertial position of GMO spacecaft
    N_pos_GMO, N_vel_GMO = InertialPositionVelocity(r_GMO, angles313_GMO)

    
    # Determine if LMO spacecraft is on the sunlit side of mars at this time
    # Recall that the sun is assumed to be in the +n2 direction
    if (N_pos_LMO[1] > 0):
        u, mode, sigma_BR, B_omega_BR = SunPointingControl(t, state, gains)

    else:

        # If LMO spacecraft is not on the sunlit side of Mars, need to determine if 
        # the GMO spacecraft is visible
        N_pos_hat_LMO = N_pos_LMO / np.linalg.norm(N_pos_LMO)
        N_pos_hat_GMO = N_pos_GMO / np.linalg.norm(N_pos_GMO)
        dot_product = np.dot(N_pos_hat_LMO, N_pos_hat_GMO)
        angle_rad = np.arccos(dot_product)
        if np.rad2deg(angle_rad) < 35.0:
            # Angular difference is less than 35 degrees so the GMO spacecraft is visible
            u, mode, sigma_BR, B_omega_BR = GmoPointingControl(t, state, gains)

        else:
            # The LMO spacecraft is not on the sunlit side of Mars and cannot see the GMO spacecraft
            # so just revert to nadir pointing
            u, mode, sigma_BR, B_omega_BR = NadirPointingControl(t, state, gains)

    return u, mode, sigma_BR, B_omega_BR