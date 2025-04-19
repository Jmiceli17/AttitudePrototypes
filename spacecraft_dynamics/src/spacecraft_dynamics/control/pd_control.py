import numpy as np

from ..utils import MRP

def PDControl(sigma_BR:MRP, B_omega_BR, K, P):
    """
    Function for calculating control torques based on current attitude and angular velocity 
    tracking errors

    :param sigma_BR: Attitude tracking error between body frame and reference frame expressed as MRP
    :param B_omega_BR: Angular velocity tracking error between body frame and reference frame expressed in Body frame
    :param K: Proportional gain matrix (diagonal, 3x3)
    :param P: Deriviative gain matrix (diagonal, 3x3)

    :return u: Control torque vector in body frame [N*m]
    """

    # Convert input MRP to array 
    sigma_BR = sigma_BR.as_array()

    u = -np.matmul(K, sigma_BR) - np.matmul(P, B_omega_BR)

    return u