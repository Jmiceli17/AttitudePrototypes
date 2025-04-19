import numpy as np
from abc import ABC, abstractmethod

from ..state import ActuatorState

class WheelBase(ABC):
    def __init__(self,
                 G_J: np.array,
                 I_Ws: np.array,
                 init_state: ActuatorState,
                 wheel_torque: float = 0.0):
        self.G_J = G_J
        self.I_Ws = I_Ws
        self.state = init_state
        self.wheel_torque = wheel_torque

    @abstractmethod
    def _compute_gimbal_frame(self, state: 'ActuatorState') -> np.array:
        """Returns the DCM from gimbal to body frame"""
        pass

    def _compute_angular_velocity_gimbal_frame_projection(self, B_omega_BN: np.array, dcm_BG: np.array) -> tuple[float, float, float]:
        """
        Compute the gimbal frame angular velocity projection components ws, wt, wg (i.e the components of angular velocity projected on the gimbal frame
        basis vectors g_hat_s, g_hat_t, g_hat_g)

        Args:
            B_omega_BN (ndarray): The angular velocity of the body wrt the inertial frame expressed in body frame components [rad/s]
            dcm_BG (ndarray): DCM from gimbal frame to body frame (does not have to be the current gimbal frame)

        Returns:
            Three angular velocity components ws, wt, wg all expressed in [rad/s]
        """
        
        B_ghat_s = dcm_BG[:,0]
        B_ghat_t = dcm_BG[:,1]
        B_ghat_g = dcm_BG[:,2]

        ws = np.dot(B_ghat_s, B_omega_BN)
        wt = np.dot(B_ghat_t, B_omega_BN)
        wg = np.dot(B_ghat_g, B_omega_BN)

        return ws, wt, wg