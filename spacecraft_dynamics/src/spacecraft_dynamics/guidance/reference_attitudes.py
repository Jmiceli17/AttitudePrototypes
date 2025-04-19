import numpy as np
import math

from .calculate_dcm_hn import CalculateDcmHN
from ..orbit import InertialPositionVelocity
from ..utils import MRP
from ..utils import initial_conditions as IC
from ..utils import constants as constants
from ..utils import rigid_body_kinematics as RBK



def GetSunPointingReferenceAttitude(t):
    """
    Calculate the DCM from inertial frame to sun-pointing reference frame at a given point in time
    The sun pointing reference frame is defined as r3 aligned with sun (the solar panels are assumed
    to be mounted with a normal vector equal to b3 in the body frame), r1 = -n1, and r2 = r3 x r1
    As defined, this function is time-invariant but the function accepts t as an input in case
    that changes in the future

    :param t: elapsed time from the initial conditions of the LMO spacecraft [s]
    :return dcm_Rs_N: numpy array representing the DCM that describes the rotation
                    from the inertial frame to the sun-pointing reference frame
    """

    # Define the basis vector of the Rs frame
    # r1 is aligned with -n1
    N_r1_hat = np.array([-1,0,0])
    # This is the direction we want aligned with the sun, this scenario assumes the sun is in the n2 direction
    N_r3_hat = np.array([0,1,0])    
    # Finish the right-handed coordinate system
    N_r2_hat = np.cross(N_r3_hat, N_r1_hat)
    
    # Construct the [RsN] DCM
    # Note that [NRs] = {N_r1, N_r2, N_r3} (column vectors) so the transpose [NRs]
    # can be created with row vectors
    dcm_Rs_N = np.array([N_r1_hat,
                        N_r2_hat,
                        N_r3_hat])
    
    return dcm_Rs_N

def GetSunPointingReferenceAngularVelocity(t):
    """
    There is no angular velocity between the sun pointing reference frame and the inertial frame

    """
    N_omega_R_N = np.zeros((3,))

    return N_omega_R_N

def GetNadirPointingReferenceAttitude(t):
    """
    Calculate the DCM from inertial frame to nadir-pointing reference frame at a given point in time
    The nadir pointing reference frame is defined as r1 pointing toward the center of the planet, r2 
    aligned with the velocity vector i_theta, and r3 completes the coordinate system

    :param t: elapsed time from the initial conditions of the LMO spacecraft [s]
    :return dcm_Rn_N: numpy array representing the DCM that describes the rotation
                    from the inertial frame to the nadir-pointing reference frame
    """

    # Define the Hill frame vectors with which we want to align the nadir pointing reference frame
    H_i_r_hat = np.array([-1,0,0])  
    H_i_t_hat = np.array([0,1,0])    

    # Get dcm_N_H for the current time for the LMO spacecraft
    dcm_H_N = CalculateDcmHN(t)    
    dcm_N_H = np.transpose(dcm_H_N)

    # Define the basis vector of the Rn frame
    # Align r1 with -i_r
    N_r1_hat = np.matmul(dcm_N_H, H_i_r_hat)
    # Align r2 with velocity direction i_theta
    N_r2_hat = np.matmul(dcm_N_H, H_i_t_hat)
    # Complete the right handed coordinate system
    N_r3_hat = np.cross(N_r1_hat, N_r2_hat)


    # Construct the [RnN] DCM
    # Note that [NRn] = {N_r1, N_r2, N_r3} (column vectors) so the transpose [RnN]
    # can be created with row vectors
    dcm_Rn_N = np.array([N_r1_hat,
                        N_r2_hat,
                        N_r3_hat])

    return dcm_Rn_N


def GetNadirPointingReferenceAngularVelocity(t):
    """
    Calculate the angular velocity of the Rn frame wrt the inertial frame at a given point in time

    :param t: elapsed time from the initial conditions of the LMO spacecraft [s]
    :return N_omega_RN: numpy array representing the angular velocity of the
                    nadir-pointing reference frame wrt the inertial frame
    """

    # Define the angular velocity omega_RN = orbit_rate * i_h or -orbit_rate * r_3
    # Here we use the Hill frame to express the vector
    H_omega_RN = np.array([0, 0, IC.LMO_ORBIT_RATE])

    # Get dcm_N_H for the current time for the LMO spacecraft
    dcm_H_N = CalculateDcmHN(t)    
    dcm_N_H = np.transpose(dcm_H_N)

    # Convert to inertial frame components
    N_omega_RN = np.matmul(dcm_N_H, H_omega_RN)
    
    return N_omega_RN


def GetGmoPointingReferenceAttitude(t):
    """
    Calculate the DCM from inertial frame to GMO-pointing reference frame at a given point in time
    The GMO pointing reference frame is defined as -r1 pointing toward the GMO spacecraft (r1 aligned with -dr), 
    r2 = dr x n3, and r3 completes the coordinate system

    :param t: elapsed time from the initial conditions of the LMO spacecraft [s]
    :return dcm_Rc_N: numpy array representing the DCM that describes the rotation
                    from the inertial frame to the GMO pointing reference frame
    """

    # Get the current position and velocity of the LMO and GMO spacecraft in inertial frame components
    # LMO initial parameters
    Om_LMO = IC.LMO_OMEGA_0_RAD
    inc_LMO = IC.LMO_INC_0_RAD
    theta_LMO = IC.LMO_THETA_0_RAD
    theta_dot_LMO = IC.LMO_ORBIT_RATE
    h_LMO = IC.LMO_ALT_M

    # Integrate theta to t
    theta_LMO = theta_LMO + t*theta_dot_LMO
    theta_LMO = theta_LMO%(2*math.pi)

    # Set R and 3-1-3 angles
    r_LMO = h_LMO + constants.R_MARS
    angles313_LMO = np.array([Om_LMO, inc_LMO, theta_LMO])

    N_pos_LMO, N_vel_LMO = InertialPositionVelocity(r_LMO, angles313_LMO)

    # GMO initial parameters
    # TODO: move to constants
    Om_GMO = np.deg2rad(0.0)
    inc_GMO = np.deg2rad(0.0)
    theta_GMO = np.deg2rad(250.0)
    theta_dot_GMO = IC.GMO_ORBIT_RATE
    h_GMO = 17028.01*1000  # [m]    

    # Integrate theta to t
    theta_GMO = theta_GMO + t*theta_dot_GMO
    theta_GMO = theta_GMO%(2*math.pi)   # Wrap to 2pi

    # Set R and 3-1-3 angles
    r_GMO = h_GMO + constants.R_MARS
    angles313_GMO = np.array([Om_GMO, inc_GMO, theta_GMO])

    N_pos_GMO, N_vel_GMO = InertialPositionVelocity(r_GMO, angles313_GMO)

    # Determine direction from LMO to GMO spacecraft and express in inertial frame components [m]
    N_r_GL = N_pos_GMO - N_pos_LMO

    # n3 expressed in inertial frame components (for constructing Rc basis vectors)
    N_n3_hat = np.array([0,0,1])

    # Define the basis vector of the Rc frame
    N_r1_hat = -1.0*N_r_GL/np.linalg.norm(-1.0*N_r_GL)

    N_r2_hat = np.cross(N_r_GL, N_n3_hat)/np.linalg.norm(np.cross(N_r_GL, N_n3_hat))

    N_r3_hat = np.cross(N_r1_hat, N_r2_hat)

    # Construct the [RcN] DCM
    # Note that [NRc] = {N_r1, N_r2, N_r3} (column vectors) so the transpose [RcN]
    # can be created with row vectors
    dcm_Rc_N = np.array([N_r1_hat,
                        N_r2_hat,
                        N_r3_hat])

    return dcm_Rc_N


def GetGmoPointingReferenceAngularVelocity(t):
    """
    Calculate the angular velocity of the Rc frame wrt the inertial frame at a given point in time
    note t must be greater than 0.1

    :param t: elapsed time from the initial conditions of the LMO spacecraft [s]
    :return N_omega_RcN: numpy array representing the angular velocity of the
                    GMO-pointing reference frame wrt the inertial frame
    """
    # if t < 0.1:
    #     print(" > [GetGmoPointingReferenceAngularVelocity][ERROR] t must be greater than 0.1")
    #     return

    # In this case, we'll leverage the kinematic differential equation
    # [NRc_dot] = -[N_omega_NR_tilde]*[NR]

    # Approximate [NRc_dot] numerically by applying a small time step and evaluating the change in [NRc]
    dt = 0.1
    dcm_Rc_N_0 = GetGmoPointingReferenceAttitude(t - dt)
    dcm_Rc_N_1 = GetGmoPointingReferenceAttitude(t)

    dcm_N_Rc_0 = np.transpose(dcm_Rc_N_0)
    dcm_N_Rc_1 = np.transpose(dcm_Rc_N_1)
    
    dcm_N_Rc_dt = dcm_N_Rc_1 - dcm_N_Rc_0

    dcm_N_Rc_dot = 1/dt * dcm_N_Rc_dt
    
    N_omega_NRc_tilde = np.matmul(-dcm_N_Rc_dot, np.transpose(dcm_N_Rc_1))

    # We have the angular velocity of N wrt Rc but we need angular velocity of Rc wrt N
    N_omega_RcN_tilde = np.transpose(N_omega_NRc_tilde)

    # Extract angular velocity from the tilde matrix
    w1 = -N_omega_RcN_tilde[1,2]
    w2 = N_omega_RcN_tilde[0,2]
    w3 = -N_omega_RcN_tilde[0,1]
    N_omega_RcN = np.array([w1, w2, w3])
    
    return N_omega_RcN

def GetHomework6ReferenceAttitude(t):

    f = 0.05 # [rad/s]
    sigma_RN = MRP(0.2*np.sin(f*t), 0.3*np.cos(f*t), -0.3*np.sin(f*t))

    dcm_R_N = RBK.MRP2C(sigma_RN.as_array())

    return dcm_R_N

def GetHomework6ReferenceAngularVelocity(t):

    # Approximate [NRc_dot] numerically by applying a small time step and evaluating the change in [NRc]
    dt = 0.1
    dcm_R_N_0 = GetHomework6ReferenceAttitude(t - dt)
    dcm_R_N_1 = GetHomework6ReferenceAttitude(t)

    dcm_N_R_0 = np.transpose(dcm_R_N_0)
    dcm_N_R_1 = np.transpose(dcm_R_N_1)
    
    dcm_N_R_dt = dcm_N_R_1 - dcm_N_R_0

    dcm_N_R_dot = 1/dt * dcm_N_R_dt
    
    N_omega_NR_tilde = np.matmul(-dcm_N_R_dot, np.transpose(dcm_N_R_1))

    # We have the angular velocity of N wrt Rc but we need angular velocity of Rc wrt N
    N_omega_RN_tilde = np.transpose(N_omega_NR_tilde)

    # Extract angular velocity from the tilde matrix
    w1 = -N_omega_RN_tilde[1,2]
    w2 = N_omega_RN_tilde[0,2]
    w3 = -N_omega_RN_tilde[0,1]
    N_omega_RN = np.array([w1, w2, w3])
    
    return N_omega_RN


def GetHomework6ReferenceAngularAcceleration(t):

    dt = 0.1
    N_omega_RN_0 = GetHomework6ReferenceAngularVelocity(t-dt)
    N_omega_RN_1 = GetHomework6ReferenceAngularVelocity(t)

    N_omega_RN_dot = 1/dt*(N_omega_RN_1 - N_omega_RN_0)

    return N_omega_RN_dot

if __name__ == "__main__":

    ##### TASK 3
    # Test [RsN] by evaluating at t=0
    dcm_Rs_N = GetSunPointingReferenceAttitude(0)
    print("dcm_Rs_N: \n{}".format(dcm_Rs_N))


    ##### TASK 4
    # Evaluate [RnN] at t=330
    dcm_Rn_N = GetNadirPointingReferenceAttitude(330)
    print("dcm_Rn_N: \n{}".format(dcm_Rn_N))

    # Evaluate N_omega_RN  at t=330
    N_omega_RN = GetNadirPointingReferenceAngularVelocity(330)
    print("N_omega_RN: \n{}".format(N_omega_RN))


    ##### TASK 5
    # Evaluate at t = 330
    dcm_Rc_N = GetGmoPointingReferenceAttitude(330)
    print("dcm_Rc_N: \n{}".format(dcm_Rc_N))
    
    N_omega_RcN = GetGmoPointingReferenceAngularVelocity(330)
    print("N_omega_RcN: \n{}".format(N_omega_RcN))