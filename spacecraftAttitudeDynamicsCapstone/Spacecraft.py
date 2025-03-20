import numpy as np
import math
from abc import ABC, abstractmethod
from State import SpacecraftState, ActuatorState, VscmgState, ReactionWheelState

class ControlGains:
    def __init__(self, proportional: np.array, derivative: np.array):
        """
        Control gains for the spacecraft

        Args:
            proportional (np.array): 3x3 Proportional gain matrix
            derivative (np.array): 3x3 Derivative gain matrix
        """
        self.proportional = proportional
        self.derivative = derivative


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
        Compute the gimbal frame angular velocity projection components ws, wt, wg and a desired DCM [BG] (does not have to be the current DCM)
        
        Args:
            B_omega_BN (ndarray): The angular velocity of the body wrt the inertial frame expressed in body frame
            dcm_BG (ndarray): DCM from gimbal frame to body frame
        Returns:    
            tuple: Three floats ws, wt, wg
        """
        
        B_ghat_s = dcm_BG[:,0]
        B_ghat_t = dcm_BG[:,1]
        B_ghat_g = dcm_BG[:,2]

        ws = np.dot(B_ghat_s, B_omega_BN)
        wt = np.dot(B_ghat_t, B_omega_BN)
        wg = np.dot(B_ghat_g, B_omega_BN)

        return ws, wt, wg
        
class Vscmg(WheelBase):
    def __init__(self,
                 G_J: np.array,
                 I_Ws: np.array,
                 init_state: VscmgState,
                 dcm_BG_init: np.array = None,
                 gimbal_angle_init: float = None,
                 wheel_torque: float = 0.0,
                 gimbal_torque: float = 0.0):
        super().__init__(G_J, I_Ws, init_state, wheel_torque)
        self.dcm_BG_init = dcm_BG_init
        self.gimbal_angle_init = gimbal_angle_init
        self.gimbal_torque = gimbal_torque

    def _compute_gimbal_frame(self, state: VscmgState) -> np.array:
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
        For reaction wheels, we ignore the state parameter since the frame is fixed
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

class Spacecraft:
    def __init__(self, 
                 B_Is: np.array, 
                 init_state: SpacecraftState, 
                 actuators: list[WheelBase],
                 control_gains: ControlGains):
        self.B_Is = B_Is
        self.state = init_state
        self.actuators = actuators
        self.control_gains = control_gains
        self.total_inertia = self.B_Is  # TODO: should this be initialized with actuator inertias?

    def update_state(self, state: SpacecraftState) -> None:
        """
        Updates the spacecraft's state and all attached actuator states.
        
        Arguments:
            state (SpacecraftState): The state to update the spacecraft with
        """
        if not isinstance(state, SpacecraftState):
            raise TypeError(f"Expected SpacecraftState, got {type(state)}")

        # Check if the number of actuator states matches the number of attached actuators
        if len(state.actuator_states) != len(self.actuators):
            raise ValueError(
                f"Mismatch in number of actuators: {len(state.actuator_states)} states provided, "
                f"but spacecraft has {len(self.actuators)} actuators."
            )

        # Verify actuator state types match
        for actuator, new_state in zip(self.actuators, state.actuator_states):
            if not isinstance(new_state, type(actuator.state)):
                raise TypeError(
                    f"Actuator state type mismatch. Expected {type(actuator.state)}, "
                    f"got {type(new_state)}"
                )

        # Update spacecraft state
        self.state = state

        # Update each actuator's state
        for actuator, new_state in zip(self.actuators, state.actuator_states):
            actuator.state = new_state

        # Update the current total spacecraft + actuator inertia
        B_I = self.B_Is.copy()

        # Add inertia from each actuator
        for actuator, actuator_state in zip(self.actuators, state.actuator_states):
            dcm_BG = actuator._compute_gimbal_frame(actuator_state)
            B_J = np.matmul(dcm_BG, np.matmul(actuator.G_J, np.transpose(dcm_BG)))
            B_I += B_J

        self.total_inertia = B_I


    def update_control_torque(self, B_L_R: np.array) -> None:
        """
        Maps a desired control torque vector to wheel torque commands using minimum norm pseudoinverse solution.
        Solves the equation [Gs]us = -Lr where:
        - [Gs] is the matrix mapping wheel torques to body torques
        - us is the vector of wheel torque commands
        - Lr is the desired control torque vector

        Args:
            B_L_R (np.array): 3x1 Desired control torque in body frame [Nm]
        """
        # First verify we only have reaction wheels
        for actuator in self.actuators:
            if not isinstance(actuator, ReactionWheel):
                raise TypeError("This function only supports reaction wheel actuators")

        # Construct the [Gs] matrix where each column is the spin axis for each wheel
        num_wheels = len(self.actuators)
        Gs = np.zeros((3, num_wheels))
        
        for i, wheel in enumerate(self.actuators):
            Gs[:, i] = wheel.spin_axis

        # Calculate minimum norm pseudoinverse solution
        # us = -pinv(Gs) * B_L_R
        Gs_pinv = np.linalg.pinv(Gs)
        wheel_torques = -np.matmul(Gs_pinv, B_L_R)

        # Update each wheel's torque command
        for wheel, torque in zip(self.actuators, wheel_torques):
            wheel.wheel_torque = torque

    # def CalculateTotalEnergy(self) -> float:
    #     """
    #     Calculate the total rotational kinetic energy of the system (see eq. 4.144 in Schaub, Junkins)

    #     Args:
    #         state (SpacecraftState): Current state of the entire system
    #     """
        
    #     B_omega_BN = self.state.B_omega_BN


    #     # Energy of spacecraft
    #     T_sc = 0.5 * np.dot(B_omega_BN, np.matmul(self.B_Is, B_omega_BN))

    #     # Total energy will be the kinetic energy of the spacecraft plus all the VSCMGs
    #     T_total = T_sc

    #     for vscmg in self.vscmgs:

    #         # Extract inertia variables
    #         Js = vscmg.G_J[0,0]
    #         Jt = vscmg.G_J[1,1]
    #         Jg = vscmg.G_J[2,2]
    #         I_Ws = vscmg.I_Ws
    #         I_Gs = Js - I_Ws

    #         gimbal_angle = vscmg.state.gimbal_angle
    #         wheel_speed = vscmg.state.wheel_speed
    #         gimbal_rate = vscmg.state.gimbal_rate

    #         dcm_BG = vscmg._compute_gimbal_frame(vscmg_state=vscmg.state)
    #         ws, wt, wg = self._compute_angular_velocity_gimbal_frame_projection(B_omega_BN=B_omega_BN,
    #                                                                             dcm_BG=dcm_BG)

    #         # Energy of gimbal
    #         T_gimbal = 0.5 * ((I_Gs * ws**2) + (Jt * wt**2) + (Jg * (wg + gimbal_rate)**2 ))

    #         # Energy of wheel
    #         T_wheel = 0.5 * (I_Ws * (wheel_speed + ws)**2)

    #         # Add to total
    #         T_total += T_gimbal + T_wheel

    #     return T_total 
