import numpy as np
import math

from .wheel_base import WheelBase
from ..state import VscmgState


class Vscmg(WheelBase):
    def __init__(self,
                 G_J: np.array,
                 I_Ws: np.array,
                 init_state: VscmgState,
                 dcm_BG_init: np.array = None,
                 gimbal_angle_init: float = None,
                 wheel_torque: float = 0.0,
                 gimbal_torque: float = 0.0,
                 K_gimbal:float = 1.0):
        super().__init__(G_J, I_Ws, init_state, wheel_torque)
        self.dcm_BG_init = dcm_BG_init
        self.gimbal_angle_init = gimbal_angle_init
        self.gimbal_torque = gimbal_torque
        # Proportional gain for VSCMG acceleration based steering law
        self.K_gimbal = K_gimbal

    def _compute_gimbal_frame(self, state: VscmgState) -> np.array:
        """
        Compute the current gimbal frame for this actuator
        For reaction wheels, this frame is fixed and does not change with state

        Args:
            state (VscmgState): State of the actuator to use for evaluation
        
        Returns:
            3x3 DCM [BG] describing the orienation of the gimbal frame wrt the body frame
        """
        if not isinstance(state, VscmgState):
            raise TypeError(f"Expected VscmgState, got {type(state)}")
            
        # Get the current gimbal angle for this VSCMG
        gimbal_angle = state.gimbal_angle
        gamma0 = self.gimbal_angle_init
        dcm_BG_init = self.dcm_BG_init
        
        B_ghat_s_init = dcm_BG_init[:,0]
        B_ghat_t_init = dcm_BG_init[:,1]
        B_ghat_g_init = dcm_BG_init[:,2]

        B_ghat_s = ((math.cos(gimbal_angle - gamma0) * B_ghat_s_init) + 
                    (math.sin(gimbal_angle - gamma0) * B_ghat_t_init))
        B_ghat_t = ((-math.sin(gimbal_angle - gamma0) * B_ghat_s_init) + 
                    (math.cos(gimbal_angle - gamma0) * B_ghat_t_init))
        B_ghat_g = B_ghat_g_init

        dcm_GB = np.array([B_ghat_s,
                          B_ghat_t,
                          B_ghat_g])
        dcm_BG = np.transpose(dcm_GB)

        return dcm_BG