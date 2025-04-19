import numpy as np
from enum import Enum
import math

from .reference_attitudes import (
    GetSunPointingReferenceAttitude,
    GetSunPointingReferenceAngularVelocity,
    GetNadirPointingReferenceAttitude,
    GetNadirPointingReferenceAngularVelocity,
    GetGmoPointingReferenceAttitude,
    GetGmoPointingReferenceAngularVelocity,
    GetHomework6ReferenceAttitude,
    GetHomework6ReferenceAngularVelocity,
    GetHomework6ReferenceAngularAcceleration)
from ..control import (
    CalculateAttitudeError,
    PDControl
)
from ..orbit import InertialPositionVelocity
from ..utils import initial_conditions as IC
from ..utils import constants as constants
from ..utils import rigid_body_kinematics as RBK

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


def Homework6_CC1_Control(t, state, gains = (np.eye(3),np.eye(3))):
    """
    Attitude reference is the inertial frame
    Angular velocity reference is 0s
    
    This function implements a slightly more complicated control law that uses the gyroscopic terms of the reference
    angular velocity
    """
    pointing_mode = Mode.INVALID

    sigma_BN = state[0]
    B_omega_BN = state[1]
    K = gains[0]
    P = gains[1]
    # print("[Homework6_CC1_Control] \n> Gains: \n> K: {} \n> P: {}".format(gains[0], gains[1]))
    I = np.array([[100.0, 0, 0],
                [0, 75.0, 0],
                [0, 0, 80.0]])


    dcm_R_N = np.eye(3) # Desired frame is identity
    N_omega_RN = np.zeros(3)    

    # Use the current attitude to determine the attitude and ang vel tracking error
    sigma_BR, B_omega_BR = CalculateAttitudeError(sigma_BN, B_omega_BN, dcm_R_N, N_omega_RN)

    # print("[Homework6_CC1_Control] \n> sigma_BR: {}".format(sigma_BR))

    # Convert ang vel reference to body frame
    dcm_B_N = RBK.MRP2C(sigma_BN)
    B_omega_RN = np.matmul(dcm_B_N, N_omega_RN)

    # Calculate tilde matrix of omega_BN
    B_omega_BN_tilde = RBK.v3Tilde(B_omega_BN)
    

    # Calculate control torques
    u = np.matmul(-K, sigma_BR.as_array()) - np.matmul(P, B_omega_BR) + np.matmul(I, (-np.cross(B_omega_BN, B_omega_RN))) + np.matmul(B_omega_BN_tilde, np.matmul(I, B_omega_BN)) 

    return u, pointing_mode, sigma_BR, B_omega_BR
     

def Homework6_CC1_AttitudeTrackingControl(t, state, gains = (np.eye(3),np.eye(3))):
    pointing_mode = Mode.INVALID

    sigma_BN = state[0]
    B_omega_BN = state[1]
    K = gains[0]
    P = gains[1]
    # print("[Homework6_CC1_Control] \n> Gains: \n> K: {} \n> P: {}".format(gains[0], gains[1]))
    I = np.array([[100.0, 0, 0],
                [0, 75.0, 0],
                [0, 0, 80.0]])


    dcm_R_N = GetHomework6ReferenceAttitude(t)
    N_omega_RN = GetHomework6ReferenceAngularVelocity(t)

    # Use the current attitude to determine the attitude and ang vel tracking error
    sigma_BR, B_omega_BR = CalculateAttitudeError(sigma_BN, B_omega_BN, dcm_R_N, N_omega_RN)

    # print("[Homework6_CC1_Control] \n> sigma_BR: {}".format(sigma_BR))

    # Convert ang vel reference to body frame
    dcm_B_N = RBK.MRP2C(sigma_BN)
    B_omega_RN = np.matmul(dcm_B_N, N_omega_RN)

    # Calculate tilde matrix of omega_BN and omega_RN
    B_omega_BN_tilde = RBK.v3Tilde(B_omega_BN)
    B_omega_RN_tilde = RBK.v3Tilde(B_omega_RN)
    

    # # # Calculate control torques
    # # # This doesn't seem to work
    # # u = np.matmul(-K, sigma_BR.as_array()) \
    # #     - np.matmul(P, B_omega_BR) \
    # #     - np.matmul(B_omega_RN_tilde, np.matmul(I, B_omega_RN))\
    # #     - np.matmul(I, (np.cross(B_omega_BN, B_omega_RN))) \
    # #     + np.matmul(B_omega_BN_tilde, np.matmul(I, B_omega_BN)) 

    # Approximate the reference angular acceleration
    N_omega_RN_dot = GetHomework6ReferenceAngularAcceleration(t)

    # Convert to body frame components
    B_omega_RN_dot = np.matmul(dcm_B_N, N_omega_RN_dot)

    # Calculate control torques
    u = np.matmul(-K, sigma_BR.as_array()) \
        - np.matmul(P, B_omega_BR) \
        + np.matmul(I, (B_omega_RN_dot - np.cross(B_omega_BN, B_omega_RN))) \
        + np.matmul(B_omega_BN_tilde, np.matmul(I, B_omega_BN)) 

    return u, pointing_mode, sigma_BR, B_omega_BR

def Homework6_CC2_AttitudeTrackingControl(t, state, gains = (np.eye(3),np.eye(3))):
    """
    No external torque fed forward in the controller
    """
    pointing_mode = Mode.INVALID

    sigma_BN = state[0]
    B_omega_BN = state[1]
    K = gains[0]
    P = gains[1]

    # Same principal inertias as CC1
    I = np.array([[100.0, 0, 0],
                [0, 75.0, 0],
                [0, 0, 80.0]])


    # Same reference attitudes and angular velocity as CC1
    dcm_R_N = GetHomework6ReferenceAttitude(t)
    N_omega_RN = GetHomework6ReferenceAngularVelocity(t)

    # Use the current attitude to determine the attitude and ang vel tracking error
    sigma_BR, B_omega_BR = CalculateAttitudeError(sigma_BN, B_omega_BN, dcm_R_N, N_omega_RN)

    # Convert ang vel reference to body frame
    dcm_B_N = RBK.MRP2C(sigma_BN)
    B_omega_RN = np.matmul(dcm_B_N, N_omega_RN)

    # Calculate tilde matrix of omega_BN and omega_RN
    B_omega_BN_tilde = RBK.v3Tilde(B_omega_BN)
    B_omega_RN_tilde = RBK.v3Tilde(B_omega_RN)
    

    # Calculate control torques
    u = np.matmul(-K, sigma_BR.as_array()) - np.matmul(P, B_omega_BR)

    return u, pointing_mode, sigma_BR, B_omega_BR    


def Homework6_CC2_AttitudeTrackingControl(t, state, gains = (np.eye(3),np.eye(3))):
    """
    External torque included in controller
    """

    pointing_mode = Mode.INVALID

    sigma_BN = state[0]
    B_omega_BN = state[1]
    K = gains[0]
    P = gains[1]
    # print("[Homework6_CC1_Control] \n> Gains: \n> K: {} \n> P: {}".format(gains[0], gains[1]))
    I = np.array([[100.0, 0, 0],
                [0, 75.0, 0],
                [0, 0, 80.0]])


    dcm_R_N = GetHomework6ReferenceAttitude(t)
    N_omega_RN = GetHomework6ReferenceAngularVelocity(t)

    # Use the current attitude to determine the attitude and ang vel tracking error
    sigma_BR, B_omega_BR = CalculateAttitudeError(sigma_BN, B_omega_BN, dcm_R_N, N_omega_RN)

    # print("[Homework6_CC1_Control] \n> sigma_BR: {}".format(sigma_BR))

    # Convert ang vel reference to body frame
    dcm_B_N = RBK.MRP2C(sigma_BN)
    B_omega_RN = np.matmul(dcm_B_N, N_omega_RN)

    # Calculate tilde matrix of omega_BN and omega_RN
    B_omega_BN_tilde = RBK.v3Tilde(B_omega_BN)
    B_omega_RN_tilde = RBK.v3Tilde(B_omega_RN)
    
    # Approximate the reference angular acceleration
    N_omega_RN_dot = GetHomework6ReferenceAngularAcceleration(t)

    # Convert to body frame components
    B_omega_RN_dot = np.matmul(dcm_B_N, N_omega_RN_dot)

    # External torque (numbers from Homework 6 CC2 Problem 6/7)
    L = np.array([0.5, -0.3, 0.2])

    # Calculate control torques
    u = np.matmul(-K, sigma_BR.as_array()) \
        - np.matmul(P, B_omega_BR) \
        + np.matmul(I, (B_omega_RN_dot - np.cross(B_omega_BN, B_omega_RN))) \
        + np.matmul(B_omega_BN_tilde, np.matmul(I, B_omega_BN)) \
        - L

    return u, pointing_mode, sigma_BR, B_omega_BR


def Homework6_CC1_AttitudeTrackingConstrainedControl(t, state, gains = (np.eye(3),np.eye(3))):
    pointing_mode = Mode.INVALID

    sigma_BN = state[0]
    B_omega_BN = state[1]
    K = gains[0]
    P = gains[1]
    # print("[Homework6_CC1_Control] \n> Gains: \n> K: {} \n> P: {}".format(gains[0], gains[1]))
    I = np.array([[100.0, 0, 0],
                [0, 75.0, 0],
                [0, 0, 80.0]])


    dcm_R_N = GetHomework6ReferenceAttitude(t)
    N_omega_RN = GetHomework6ReferenceAngularVelocity(t)

    # Use the current attitude to determine the attitude and ang vel tracking error
    sigma_BR, B_omega_BR = CalculateAttitudeError(sigma_BN, B_omega_BN, dcm_R_N, N_omega_RN)

    # print("[Homework6_CC1_Control] \n> sigma_BR: {}".format(sigma_BR))

    # Convert ang vel reference to body frame
    dcm_B_N = RBK.MRP2C(sigma_BN)
    B_omega_RN = np.matmul(dcm_B_N, N_omega_RN)

    # Calculate tilde matrix of omega_BN and omega_RN
    B_omega_BN_tilde = RBK.v3Tilde(B_omega_BN)

    # Approximate the reference angular acceleration
    N_omega_RN_dot = GetHomework6ReferenceAngularAcceleration(t)

    # Convert to body frame components
    B_omega_RN_dot = np.matmul(dcm_B_N, N_omega_RN_dot)

    # Calculate control torques
    u = np.matmul(-K, sigma_BR.as_array()) \
        - np.matmul(P, B_omega_BR) \
        + np.matmul(I, (B_omega_RN_dot - np.cross(B_omega_BN, B_omega_RN))) \
        + np.matmul(B_omega_BN_tilde, np.matmul(I, B_omega_BN)) 

    for idx in range(len(u)):
        if np.abs(u[idx]) < 1.0: # max torque [Nm]
            u[idx] = 1.0

    return u, pointing_mode, sigma_BR, B_omega_BR


