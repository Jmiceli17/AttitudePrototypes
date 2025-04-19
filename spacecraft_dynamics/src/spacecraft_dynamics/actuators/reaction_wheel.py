import numpy as np

from .wheel_base import WheelBase
from ..state import ReactionWheelState

class ReactionWheel(WheelBase):
    def __init__(self,
                 G_J: np.array,
                 I_Ws: np.array,
                 init_state: ReactionWheelState,
                 spin_axis: np.array,
                 wheel_torque: float = 0.0):
        super().__init__(G_J, I_Ws, init_state, wheel_torque)
        self.spin_axis = spin_axis / np.linalg.norm(spin_axis)  # Normalize the axis

    def _compute_gimbal_frame(self, state: ReactionWheelState) -> np.array:
        """
        Compute the current gimbal frame for this actuator
        For reaction wheels, this frame is fixed and does not change with state

        Args:
            state (ReactionWheelState): State of the actuator to use for evaluation
        
        Returns:
            3x3 DCM [BG] describing the orienation of the gimbal frame wrt the body frame
        """
        # First column is the spin axis
        s = self.spin_axis
        # Choose any vector not parallel to s for temporary vector
        if abs(np.dot(s, [1,0,0])) < 0.9:
            temp = np.array([1,0,0])
        else:
            temp = np.array([0,1,0])
        # Construct orthogonal vectors
        t = np.cross(s, temp)
        t = t / np.linalg.norm(t)
        g = np.cross(s, t)
        # Return DCM
        return np.column_stack((s, t, g))