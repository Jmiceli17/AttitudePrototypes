import numpy as np

from ..utils import rigid_body_kinematics as RBK
from ..utils import MRP
from ..utils import initial_conditions as IC
from ..guidance import reference_attitudes as reference_attitudes 


def CalculateAttitudeError(sigma_BN, B_omega_BN, dcm_R_N, N_omega_RN):
    """
    Function for calculating the tracking error between the rigid body's current attitude and angular velocity 
    and the reference attitude and angular velocity

    :param sigma_BN: MRP set describing the current orientation of the body wrt the inertial frame (as an np array)
    :param B_omega_BN: The angular velocity of the body wrt the inertial frame expressed in body frame [rad/s]
    :param dcm_R_N: The DCM describing the rotation from the inertial frame the the desired reference frame
    :param N_omega_RN: The reference (i.e. desired) angular velocity of the body wrt the inertial frame 
            expressed in inertial frame components [rad/s] 

    :return sigma_BR:  MRP representing the rotation from the reference (i.e. desired) frame to the body frame
    :return B_omega_BR: Angular velocity of B frame wrt the reference (desired) frame expressed in body frame components
    """

    # First calculate attitude tracking error as the DCM from the R frame to the B frame
    # Convert current attitude from MRP to DCM
    dcm_B_N = RBK.MRP2C(sigma_BN)
    dcm_N_R = np.transpose(dcm_R_N)

    # Attitude error in DCM form
    dcm_B_R = np.matmul(dcm_B_N, dcm_N_R)

    # Convert to MRP
    sigma_BR_array = RBK.C2MRP(dcm_B_R)
    s1 = sigma_BR_array[0]
    s2 = sigma_BR_array[1]
    s3 = sigma_BR_array[2]
    sigma_BR = MRP(s1,s2,s3)
    # Now calculate angular velocity tracking error
    # convert omega_RN to body frame components 
    B_omega_RN = np.matmul(dcm_B_N, N_omega_RN)

    # Angular velocity tracking error in body frame components
    B_omega_BR = B_omega_BN - B_omega_RN

    return sigma_BR, B_omega_BR


if __name__ == "__main__":

    ##### TASK 6
    # Evaluate attitude and ang vel tracking errors for each reference frame type at t=0
    t = 0.0
    sigma_BN = IC.LMO_SIGMA_BN_0.as_array()
    B_omega_BN = IC.LMO_B_OMEGA_BN_0

    # Sun pointing
    dcm_Rs_N = reference_attitudes.GetSunPointingReferenceAttitude(t)
    N_omega_RsN = reference_attitudes.GetSunPointingReferenceAngularVelocity(t)

    sigma_BRs, B_omega_BRs = CalculateAttitudeError(sigma_BN, B_omega_BN, dcm_Rs_N, N_omega_RsN)
    print("sigma_BRs: \n{} ".format(sigma_BRs.as_array()))
    print("B_omega_BRs: \n{} ".format(B_omega_BRs))

    # Nadir pointing
    dcm_Rn_N = reference_attitudes.GetNadirPointingReferenceAttitude(t)
    N_omega_RnN = reference_attitudes.GetNadirPointingReferenceAngularVelocity(t)

    sigma_BRn, B_omega_BRn = CalculateAttitudeError(sigma_BN, B_omega_BN, dcm_Rn_N, N_omega_RnN)
    print("sigma_BRn: \n{} ".format(sigma_BRn.as_array()))
    print("B_omega_BRn: \n{} ".format(B_omega_BRn))

    # GMO Pointing
    dcm_Rg_N = reference_attitudes.GetGmoPointingReferenceAttitude(t)
    N_omega_RgN = reference_attitudes.GetGmoPointingReferenceAngularVelocity(t)

    sigma_BRg, B_omega_BRg = CalculateAttitudeError(sigma_BN, B_omega_BN, dcm_Rg_N, N_omega_RgN)
    print("sigma_BRg: \n{} ".format(sigma_BRg.as_array()))
    print("B_omega_BRg: \n{} ".format(B_omega_BRg))
