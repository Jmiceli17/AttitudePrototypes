from abc import ABC, abstractmethod
from State import SpacecraftState, VscmgState, ActuatorState
import RigidBodyKinematics as RBK 
from PointingControl import Mode
from ModifiedRodriguesParameters import MRP
from Spacecraft import Spacecraft, Vscmg, WheelBase

import numpy as np


class BaseSpacecraftDynamics(ABC):
    """Abstract base class for spacecraft dynamics"""
    
    def __init__(self, spacecraft: Spacecraft):
        """Initialize the dynamics model
        
        Args:
            spacecraft (Spacecraft): The spacecraft object being modeled
        """
        self.spacecraft = spacecraft

    @abstractmethod
    def compute_state_derivatives(self, t: float, state_array: np.array, **kwargs) -> np.array:
        """Compute state derivatives based on current state and inputs
        
        Args:
            t (float): Current time
            state_array (np.array): Current state vector
            **kwargs: Additional arguments specific to dynamics type
            
        Returns:
            np.array: State derivatives
        """
        pass

    @abstractmethod
    def simulate(self, t_init: float, t_max: float, t_step: float = 0.1, **kwargs) -> dict:
        """Simulate the system forward in time
        
        Args:
            t_init (float): Initial time
            t_max (float): End time
            t_step (float): Time step for integration
            **kwargs: Additional simulation parameters
            
        Returns:
            dict: Solution dictionary containing time histories
        """
        pass

    def CalculateTotalPower(self, spacecraft: Spacecraft, external_torque: np.array) -> float:
        """
        Calculate the total rate of change of rotational kinetic energy of the system 
        given the current state of the spacecraft (see eq. 4.145 in Schaub, Junkins)

        Args:
            spacecraft (Spacecraft): Current state of the entire system

        Returns:
            float: Current rate of change of inertial kinetic energy of the entire system
        """
        B_omega_BN = spacecraft.state.B_omega_BN
        T_dot = np.dot(B_omega_BN, external_torque)

        # TODO: add power terms from VSCMGs...

        return T_dot
    
    def CalculateTotalEnergy(self, spacecraft: Spacecraft) -> float:
        """
        TODO: Move this to Spacecraft class
        Calculate the total rotational kinetic energy of the system given the current state
        of the spacecraft (see eq. 4.144 in Schaub, Junkins)

        Args:
            spacecraft (Spacecraft): Current state of the entire system

        Returns:
            float: Current inertial kinetic energy of the entire system
        """
        
        B_omega_BN = spacecraft.state.B_omega_BN

        # Energy of spacecraft
        T_sc = 0.5 * np.dot(B_omega_BN, np.matmul(spacecraft.B_Is, B_omega_BN))

        # Total energy will be the kinetic energy of the spacecraft plus all the VSCMGs
        T_total = T_sc

        for actuator in spacecraft.actuators:

            # Extract inertia variables
            Js = actuator.G_J[0,0]
            Jt = actuator.G_J[1,1]
            Jg = actuator.G_J[2,2]
            I_Ws = actuator.I_Ws
            I_Gs = abs(Js - I_Ws)

            wheel_speed = actuator.state.wheel_speed

            if isinstance(actuator, Vscmg):
                gimbal_angle = actuator.state.gimbal_angle
                gimbal_rate = actuator.state.gimbal_rate
            else:
                gimbal_angle = 0.0
                gimbal_rate = 0.0

            dcm_BG = actuator._compute_gimbal_frame(actuator.state)
            ws, wt, wg = actuator._compute_angular_velocity_gimbal_frame_projection(B_omega_BN=B_omega_BN,
                                                                                dcm_BG=dcm_BG)

            # Energy of gimbal
            T_gimbal = 0.5 * ((I_Gs * ws**2) + (Jt * wt**2) + (Jg * (wg + gimbal_rate)**2 ))

            # Energy of wheel
            T_wheel = 0.5 * (I_Ws * (wheel_speed + ws)**2)

            # Add to total
            T_total += T_gimbal + T_wheel

        return T_total 
    
    def CalculateTotalInertialAngularMomentum(self, spacecraft:Spacecraft) -> np.array:
        """
        TODO: Move to Spacecraft class
        Calculate the total angular momentum of the system in inertial frame components 
        (derived from eq 4.113 in Schuab, Junkins)

        Args:
            state (SpacecraftState): Current state of the entire system
        
        Reutrns:
            ndarray: Angular momentum vector of the total system expressed in the inertial frame [Nms]
        """

        spacecraft_state = spacecraft.state
        sigma_BN = spacecraft_state.sigma_BN
        B_omega_BN = spacecraft_state.B_omega_BN

        # Inertial to body DCM
        dcm_BN = sigma_BN.MRP2C()

        # Body to inertial DCM
        dcm_NB = np.transpose(dcm_BN)

        # Angular momentum of the body/spacecraft
        B_H_B = np.matmul(spacecraft.B_Is, B_omega_BN)
        N_H_B = np.matmul(dcm_NB, B_H_B)

        # Total inertial angular momentum will be the body + all the VSCMGs
        N_H_total = N_H_B

        for actuator in spacecraft.actuators:

            wheel_speed = actuator.state.wheel_speed

            # Get current state info for this VSCMG
            if isinstance(actuator, Vscmg):
                gimbal_angle = actuator.state.gimbal_angle
                gimbal_rate = actuator.state.gimbal_rate
            else:
                gimbal_angle = 0.0
                gimbal_rate = 0.0
            
            # Extract inertia variables
            Js = actuator.G_J[0,0]
            Jt = actuator.G_J[1,1]
            Jg = actuator.G_J[2,2]
            I_Ws = actuator.I_Ws

            dcm_BG = actuator._compute_gimbal_frame(actuator.state)
            ws, wt, wg = actuator._compute_angular_velocity_gimbal_frame_projection(B_omega_BN=B_omega_BN,
                                                                                dcm_BG=dcm_BG)

            # Angular momentum of the gimbal + wheel in gimbal frame
            G_H_vscmg = np.array([Js * ws + I_Ws * wheel_speed ,
                                Jt * wt,
                                Jg * (wg + gimbal_rate)**2])

            # Convert to body frame
            B_H_vscmg = np.matmul(dcm_BG, G_H_vscmg)

            # Convert to inertial frame
            N_H_vscmg = np.matmul(dcm_NB, B_H_vscmg)

            # Add to total angular momentum
            N_H_total += N_H_vscmg

        return N_H_total
    

    def _compute_sigma_dot(self, sigma:np.array, omega:np.array) -> np.array:
        """
        Compute the derivative of the MRPs.
        
        Args:
            sigma (ndarray): Modified Rodrigues Parameters (assumed frame A wrt B)
            omega (ndarray): Angular velocity (assumed frame A wrt B expressed in A frame) [rad/s]
            
        Returns:
            ndarray: Time derivative of MRPs
        """
        if isinstance(sigma, MRP):
            sigma = sigma.as_array()

        # MRP kinematics: σ̇ = 1/4 [(1 - σ²)I + 2[σ×] + 2σσᵀ]ω
        Bmat = RBK.BmatMRP(sigma) 
        sigma_dot = 0.25 * np.matmul(Bmat, omega)

        return sigma_dot

class TorqueFreeSpacecraftDynamics(BaseSpacecraftDynamics):
    """
    Spacecraft dynamics with no control inputs
    """
    def __init__(self, 
                 spacecraft:Spacecraft):
        """
        Initialize the dynamics model
        
        Args:
            spacecraft (Spacecraft): The spacecraft object that is being modeled

        """
        self.spacecraft = spacecraft

    # def CalculateTotalPower(self,
    #                         spacecraft:Spacecraft,
    #                         external_torque:np.array) -> float:
    #     """
    #     Calculate the total rate of change of rotational kinetic energy of the system 
    #     given the current state of the spacecraft (see eq. 4.145 in Schaub, Junkins)

    #     Args:
    #         spacecraft (Spacecraft): Current state of the entire system

    #     Returns:
    #         float: Current rate of change of inertial kinetic energy of the entire system
    #     """
    #     B_omega_BN = spacecraft.state.B_omega_BN
    #     T_dot = np.dot(B_omega_BN, external_torque)

    #     # TODO: add power terms from VSCMGs...

    #     return T_dot


    # def CalculateTotalEnergy(self,
    #                          spacecraft:Spacecraft) -> float:
    #     """
    #     TODO: Move this to Spacecraft class
    #     Calculate the total rotational kinetic energy of the system given the current state
    #     of the spacecraft (see eq. 4.144 in Schaub, Junkins)

    #     Args:
    #         spacecraft (Spacecraft): Current state of the entire system

    #     Returns:
    #         float: Current inertial kinetic energy of the entire system
    #     """
        
    #     B_omega_BN = spacecraft.state.B_omega_BN

    #     # Energy of spacecraft
    #     T_sc = 0.5 * np.dot(B_omega_BN, np.matmul(spacecraft.B_Is, B_omega_BN))

    #     # Total energy will be the kinetic energy of the spacecraft plus all the VSCMGs
    #     T_total = T_sc

    #     for actuator in spacecraft.actuators:

    #         # Extract inertia variables
    #         Js = actuator.G_J[0,0]
    #         Jt = actuator.G_J[1,1]
    #         Jg = actuator.G_J[2,2]
    #         I_Ws = actuator.I_Ws
    #         I_Gs = Js - I_Ws

    #         gimbal_angle = actuator.state.gimbal_angle
    #         wheel_speed = actuator.state.wheel_speed
    #         gimbal_rate = actuator.state.gimbal_rate

    #         dcm_BG = actuator._compute_gimbal_frame(actuator.state)
    #         ws, wt, wg = self._compute_angular_velocity_gimbal_frame_projection(B_omega_BN=B_omega_BN,
    #                                                                             dcm_BG=dcm_BG)

    #         # Energy of gimbal
    #         T_gimbal = 0.5 * ((I_Gs * ws**2) + (Jt * wt**2) + (Jg * (wg + gimbal_rate)**2 ))

    #         # Energy of wheel
    #         T_wheel = 0.5 * (I_Ws * (wheel_speed + ws)**2)

    #         # Add to total
    #         T_total += T_gimbal + T_wheel

    #     return T_total 

    # def CalculateTotalInertialAngularMomentum(self, spacecraft:Spacecraft) -> np.array:
    #     """
    #     TODO: Move to Spacecraft class
    #     Calculate the total angular momentum of the system in inertial frame components 
    #     (derived from eq 4.113 in Schuab, Junkins)

    #     Args:
    #         state (SpacecraftState): Current state of the entire system
        
    #     Reutrns:
    #         ndarray: Angular momentum vector of the total system expressed in the inertial frame [Nms]
    #     """

    #     spacecraft_state = spacecraft.state
    #     sigma_BN = spacecraft_state.sigma_BN
    #     B_omega_BN = spacecraft_state.B_omega_BN

    #     # Inertial to body DCM
    #     dcm_BN = sigma_BN.MRP2C()

    #     # Body to inertial DCM
    #     dcm_NB = np.transpose(dcm_BN)

    #     # Angular momentum of the body/spacecraft
    #     B_H_B = np.matmul(spacecraft.B_Is, B_omega_BN)
    #     N_H_B = np.matmul(dcm_NB, B_H_B)

    #     # Total inertial angular momentum will be the body + all the VSCMGs
    #     N_H_total = N_H_B

    #     for actuator in spacecraft.actuators:
            
    #         # Get current state info for this VSCMG
    #         gimbal_angle = actuator.state.gimbal_angle
    #         wheel_speed = actuator.state.wheel_speed
    #         gimbal_rate = actuator.state.gimbal_rate
            
    #         # Extract inertia variables
    #         Js = actuator.G_J[0,0]
    #         Jt = actuator.G_J[1,1]
    #         Jg = actuator.G_J[2,2]
    #         I_Ws = actuator.I_Ws

    #         dcm_BG = actuator._compute_gimbal_frame(actuator.state)
    #         ws, wt, wg = self._compute_angular_velocity_gimbal_frame_projection(B_omega_BN=B_omega_BN,
    #                                                                             dcm_BG=dcm_BG)

    #         # Angular momentum of the gimbal + wheel in gimbal frame
    #         G_H_vscmg = np.array([Js * ws + I_Ws * wheel_speed ,
    #                             Jt * wt,
    #                             Jg * (wg + gimbal_rate)**2])

    #         # Convert to body frame
    #         B_H_vscmg = np.matmul(dcm_BG, G_H_vscmg)

    #         # Convert to inertial frame
    #         N_H_vscmg = np.matmul(dcm_NB, B_H_vscmg)

    #         # Add to total angular momentum
    #         N_H_total += N_H_vscmg

    #     return N_H_total

    def compute_state_derivatives(self, 
                                t:float, 
                                state_array:np.array, 
                                external_torque=np.array([0,0,0])) -> np.array:
        """
        Compute state derivatives for either VSCMG or reaction wheel systems
        """
        # Convert array to state object
        state = SpacecraftState.from_array(
            state_array, 
            spacecraft=self.spacecraft
        )

        # Extract state variables
        sigma_BN = state.sigma_BN
        B_omega_BN = state.B_omega_BN

        # Compute total spacecraft + actuator inertia
        B_I = self.spacecraft.B_Is.copy()

        # Add inertia from each actuator
        for actuator, actuator_state in zip(self.spacecraft.actuators, state.actuator_states):
            dcm_BG = actuator._compute_gimbal_frame(actuator_state)
            B_J = np.matmul(dcm_BG, np.matmul(actuator.G_J, np.transpose(dcm_BG)))
            B_I += B_J

        # Compute state derivatives
        sigma_BN_dot = self._compute_sigma_dot(sigma_BN, B_omega_BN)
        B_omega_BN_dot = self._compute_omega_dot(B_I, state, external_torque)

        # Compute the derivatives for each actuator
        first_actuator = self.spacecraft.actuators[0]

        if isinstance(first_actuator, Vscmg):
            # VSCMG case - compute all derivatives
            wheel_speed_dot = []
            gimbal_angle_dot = []
            gimbal_rate_dot = []
            
            for actuator, actuator_state in zip(self.spacecraft.actuators, state.actuator_states):
                gimbal_angle_dot.append(actuator_state.gimbal_rate)
                
                gimbal_rate_dot_val = self._compute_gimbal_rate_dot(
                    actuator, actuator_state, B_omega_BN
                )
                gimbal_rate_dot.append(gimbal_rate_dot_val)
                
                wheel_speed_dot_val = self._compute_wheel_speed_dot(
                    actuator, actuator_state, B_omega_BN
                )
                wheel_speed_dot.append(wheel_speed_dot_val)

            state_diff = np.concatenate([
                sigma_BN_dot,
                B_omega_BN_dot,
                gimbal_angle_dot,
                gimbal_rate_dot,
                wheel_speed_dot
            ])
            
        else:  # ReactionWheel case
            # Only compute wheel speed derivatives
            wheel_speed_dot = []
            for actuator, actuator_state in zip(self.spacecraft.actuators, state.actuator_states):
                wheel_speed_dot_val = self._compute_wheel_speed_dot(
                    actuator, actuator_state, B_omega_BN
                )
                wheel_speed_dot.append(wheel_speed_dot_val)

            state_diff = np.concatenate([
                sigma_BN_dot,
                B_omega_BN_dot,
                wheel_speed_dot
            ])

        # Calculate state_dot using M matrix
        Mmat = self._compute_M_matrix(B_I, state)
        state_dot = np.linalg.solve(Mmat, state_diff)
        return state_dot

    # def SpacecraftEquationsOfMotion(self, 
    #                                 t:float, 
    #                                 state_array:np.array, 
    #                                 external_torque=np.array([0,0,0]),
    #                                 control_torque=np.array([0,0,0])) -> np.array:
    #     """
    #     Compute state derivatives for either VSCMG or reaction wheel systems
    #     """
    #     # Convert array to state object
    #     state = SpacecraftState.from_array(
    #         state_array, 
    #         spacecraft=self.spacecraft
    #     )

    #     # Extract state variables
    #     sigma_BN = state.sigma_BN
    #     B_omega_BN = state.B_omega_BN

    #     # Compute total spacecraft + actuator inertia
    #     B_I = self.spacecraft.B_Is.copy()

    #     # Add inertia from each actuator
    #     for actuator, actuator_state in zip(self.spacecraft.actuators, state.actuator_states):
    #         # TODO: may need to modify this - unsure if [I_RW] is same as [B_I]
    #         print(f"type: {type(actuator_state)}")
    #         dcm_BG = actuator._compute_gimbal_frame(actuator_state)
    #         B_J = np.matmul(dcm_BG, np.matmul(actuator.G_J, np.transpose(dcm_BG)))
    #         B_I += B_J

    #     # Compute state derivatives
    #     sigma_BN_dot = self._compute_sigma_dot(sigma_BN, B_omega_BN)
    #     B_omega_BN_dot = self._compute_omega_dot(B_I, state, external_torque)   # TODO: This needs to change for this function because we have control torque

    #     # Compute the derivatives for each actuator
    #     first_actuator = self.spacecraft.actuators[0]

    #     if isinstance(first_actuator, Vscmg):
    #         raise NotImplementedError("VSCMG case not implemented")
            
    #     else:
            
        

    # def _compute_sigma_dot(self, sigma:np.array, omega:np.array) -> np.array:
    #     """
    #     Compute the derivative of the MRPs.
        
    #     Args:
    #         sigma (ndarray): Modified Rodrigues Parameters (assumed frame A wrt B)
    #         omega (ndarray): Angular velocity (assumed frame A wrt B expressed in A frame) [rad/s]
            
    #     Returns:
    #         ndarray: Time derivative of MRPs
    #     """
    #     if isinstance(sigma, MRP):
    #         sigma = sigma.as_array()

    #     # MRP kinematics: σ̇ = 1/4 [(1 - σ²)I + 2[σ×] + 2σσᵀ]ω
    #     Bmat = RBK.BmatMRP(sigma) 
    #     sigma_dot = 0.25 * np.matmul(Bmat, omega)

    #     return sigma_dot
    
    def _compute_omega_dot(self, 
                           B_I:np.array,
                           state:SpacecraftState,
                           external_torque:np.array, ) -> np.array:
        """
        Compute the RHS of the derivative of angular velocity based on ex 4.15 from H. Schaub, J. Junkins
        Args:
            B_I (ndarray): Total inertia of the system expressed in body frame components
            state (SpacecraftState): Spacecraft state
            external_torque (ndarray): External torque vector acting on the spacecraft in body frame components [Nm]

        Returns:
            ndarray: Time derivative of angular velocity
        """

        # External torques (in body frame)
        B_L = external_torque

        B_omega_BN = state.B_omega_BN

        # Calculation based on ex 4.15 from Schaub, Junkins (f_omega)
        f_omega = np.cross(-B_omega_BN, np.matmul(B_I, B_omega_BN)) + B_L
        
        # Add the component from each VSCMG
        for (actuator, actuator_state) in zip(self.spacecraft.actuators, state.actuator_states):

            Js = actuator.G_J[0,0]
            Jt = actuator.G_J[1,1]
            Jg = actuator.G_J[2,2]

            I_Ws = actuator.I_Ws

            dcm_BG = actuator._compute_gimbal_frame(state=actuator_state)

            ws, wt, wg = actuator._compute_angular_velocity_gimbal_frame_projection(B_omega_BN, dcm_BG)

            gimbal_rate = actuator_state.gimbal_rate
            wheel_speed = actuator_state.wheel_speed

            # Extract the gimbal frame vectors for this VSCMG (expressed in B frame)
            B_ghat_s = dcm_BG[:,0]
            B_ghat_t = dcm_BG[:,1]
            B_ghat_g = dcm_BG[:,2]
            
            g_hat_s_term = B_ghat_s * (Js * gimbal_rate * wt - (Jt - Jg) * wt * gimbal_rate)
            g_hat_t_term = B_ghat_t * ((Js * ws + I_Ws * wheel_speed) * gimbal_rate - \
                            (Jt + Jg) * ws * gimbal_rate + I_Ws * wheel_speed * wg)
            g_hat_g_term = B_ghat_g * (I_Ws * wheel_speed * wt)

            vscmg_term = g_hat_s_term + g_hat_t_term - g_hat_g_term

            f_omega -= vscmg_term

        return f_omega
    
    def _compute_wheel_speed_dot(self, 
                                vscmg:Vscmg,
                                vscmg_state:VscmgState,
                                B_omega_BN:np.array,) -> float:
        """
        Compute the RHS of the derivative of wheel speed based on ex 4.15 from H. Schaub, J. Junkins
        
        Args:
            vscmg (Vscmg): The VSCMG
            vscmg_state (VscmgState): The state of the vscmg with which to calculate wheel speed dot 
                (not necessarily the current state of the vscmg)
            B_omega_BN (ndarray): Angular velocity of body wrt inertial frame expressed in body 
                frame components [rad/s]

        Returns:
            float: Time derivative of wheel speed
        """
        
        I_Ws = vscmg.I_Ws

        wheel_torque = vscmg.wheel_torque
        gimbal_rate = vscmg_state.gimbal_rate
        dcm_BG = vscmg._compute_gimbal_frame(vscmg_state)

        ws, wt, wg = vscmg._compute_angular_velocity_gimbal_frame_projection(B_omega_BN=B_omega_BN, dcm_BG=dcm_BG)

        f_wheel_speed = wheel_torque - I_Ws * gimbal_rate * wt 

        return f_wheel_speed
    
    def _compute_gimbal_rate_dot(self, 
                                vscmg:Vscmg, 
                                vscmg_state:VscmgState,
                                B_omega_BN, ) -> float:
        """
        Compute the RHS of the derivative of gimbal rate based on ex 4.15 from H. Schaub, J. Junkins
        
        Args:
            vscmg (Vscmg): The VSCMG
            vscmg_state (VscmgState): The state of the vscmg with which to calculate wheel speed dot 
                (not necessarily the current state of the vscmg)
            B_omega_BN (ndarray): Angular velocity of body wrt inertial frame expressed in body 
                frame components [rad/s]

        Returns:
            float: Time derivative of gimbal rate
        """
        # Extract inertia properties
        I_Ws = vscmg.I_Ws
        Js = vscmg.G_J[0,0]
        Jt = vscmg.G_J[1,1]
        Jg = vscmg.G_J[2,2]

        # Extract properties from this VSCMG
        gimbal_torque = vscmg.gimbal_torque

        # Extract state info 
        wheel_speed = vscmg_state.wheel_speed
        dcm_BG = vscmg._compute_gimbal_frame(vscmg_state)

        # Get angular velocity projections
        ws, wt, wg = vscmg._compute_angular_velocity_gimbal_frame_projection(B_omega_BN, dcm_BG)

        # Compute the right hand side of the gimbal rate equations of motion (see ex 4.15 Schaub, Junkins)
        f_gimbal_rate = gimbal_torque + ((Js - Jt) * ws * wt) + (I_Ws * wheel_speed * wt)
        return f_gimbal_rate
    
    # def _compute_angular_velocity_gimbal_frame_projection(self, 
    #                                                       B_omega_BN:np.array, 
    #                                                       dcm_BG:np.array) -> tuple[float, float, float]:
    #     """
    #     Compute the gimbal frame angular velocity projection components ws, wt, wg
        
    #     Args:
    #         B_omega_BN (ndarray): The angular velocity of the body wrt the inertial frame expressed in body frame
    #         dcm_BG (ndarray): DCM from gimbal frame to body frame
    #     Returns:
    #         tuple: Three floats ws, wt, wg
    #     """

    #     B_ghat_s = dcm_BG[:,0]
    #     B_ghat_t = dcm_BG[:,1]
    #     B_ghat_g = dcm_BG[:,2]

    #     ws = np.dot(B_ghat_s, B_omega_BN)
    #     wt = np.dot(B_ghat_t, B_omega_BN)
    #     wg = np.dot(B_ghat_g, B_omega_BN)

    #     return ws, wt, wg


    def _compute_gimbal_frame_matrices(self, state:SpacecraftState) -> tuple[np.array, np.array, np.array]:
        """
        Compute the gimbal frame matrices for all actuators
        """
        GMat_s, GMat_t, GMat_g = [], [], []

        for actuator, actuator_state in zip(self.spacecraft.actuators, state.actuator_states):
            dcm_BG = actuator._compute_gimbal_frame(actuator_state)

            # Extract the gimbal frame vectors (expressed in B frame)
            B_ghat_s = dcm_BG[:,0]
            B_ghat_t = dcm_BG[:,1]
            B_ghat_g = dcm_BG[:,2]

            # Add the columns to the output matrices
            GMat_s.append(B_ghat_s)
            GMat_t.append(B_ghat_t)
            GMat_g.append(B_ghat_g)

        # Convert lists to 3xN matrices (stack columns)
        GMat_s = np.column_stack(GMat_s)
        GMat_t = np.column_stack(GMat_t)
        GMat_g = np.column_stack(GMat_g)

        return GMat_s, GMat_t, GMat_g


    def _compute_M_matrix(self, B_I: np.array, state:SpacecraftState) -> np.array:
        """
        Compute the system matrix for the equations of motion
        """
        num_actuators = len(self.spacecraft.actuators)
        first_actuator = self.spacecraft.actuators[0]
        
        # Create base matrices
        i3 = np.eye(3)
        zero3 = np.zeros((3, 3))
        
        if isinstance(first_actuator, Vscmg):
            # VSCMG case - use existing logic
            iN = np.eye(num_actuators)
            zero3N = np.zeros((3, num_actuators))
            zeroN3 = np.zeros((num_actuators, 3))
            zeroN = np.zeros((num_actuators, num_actuators))

            GMat_s, GMat_t, GMat_g = self._compute_gimbal_frame_matrices(state)

            # Get VSCMG properties
            Js = first_actuator.G_J[0,0]
            Jt = first_actuator.G_J[1,1]
            Jg = first_actuator.G_J[2,2]
            I_Ws = first_actuator.I_Ws

            M = np.block([
                [i3, zero3, zero3N, zero3N, zero3N],
                [zero3, B_I, zero3N, (Jg * GMat_g), (I_Ws * GMat_s)],
                [zeroN3, zeroN3, iN, zeroN, zeroN],
                [zeroN3, (Jg * GMat_g.T), zeroN, (Jg * iN), zeroN],
                [zeroN3, (I_Ws * GMat_s.T), zeroN, zeroN, (I_Ws * iN)],
            ])
            
        else:  # ReactionWheel case
            # Simpler matrix for reaction wheels (no gimbal terms)
            zero3N = np.zeros((3, num_actuators))
            zeroN3 = np.zeros((num_actuators, 3))
            iN = np.eye(num_actuators)
            
            GMat_s, _, _ = self._compute_gimbal_frame_matrices(state)
            I_Ws = first_actuator.I_Ws

            M = np.block([
                [i3, zero3, zero3N],
                [zero3, B_I, (I_Ws * GMat_s)],
                [zeroN3, (I_Ws * GMat_s.T), (I_Ws * iN)],
            ])

        return M



    def simulate(self, 
                t_init:float,
                t_max:float, 
                t_step:float=0.1,
                torque_eq:callable=None) -> dict:
        """
        4th order RK4 integrator 

        Args:
            init_state (SpacecraftState): Initial state 
            t_init (float): Initial time corresponding to initial state
            t_max (float): End time of integrator
            t_step (float): Time step for integration
            torque_eq (callable): Function to calculate external torque being applied to the spacecraft, must be of 
                form f(t, spacecraft)

        Returns:
            (dict) Dictionary mapping variables to lists of the values they had during integration
        """

        # Initialize state and time
        state = self.spacecraft.state
        t = t_init

        external_torque = torque_eq(t=t, 
                                spacecraft=self.spacecraft)

        init_energy = self.CalculateTotalEnergy(spacecraft=self.spacecraft)
        init_H = self.CalculateTotalInertialAngularMomentum(spacecraft=self.spacecraft)
        init_power = self.CalculateTotalPower(spacecraft=self.spacecraft,
                                              external_torque=external_torque)
        # TODO: turn this into class
        # Initialize containers for storing data
        solution_dict = {
            "MRP": [state.sigma_BN.as_array()],
            "omega_B_N": [state.B_omega_BN],
            "total_energy": [init_energy],
            "total_power": [init_power],
            "N_H_total": [init_H],
            "mode_value": [Mode.INVALID.value],
            "time": [t]
        }

        # Initialize actuator-specific state tracking
        for idx, actuator in enumerate(self.spacecraft.actuators):
            actuator_state = actuator.state
            if isinstance(actuator, Vscmg):
                solution_dict[f"wheel_speed_{idx}"] = [actuator_state.wheel_speed]
                solution_dict[f"gimbal_angle_{idx}"] = [actuator_state.gimbal_angle]
                solution_dict[f"gimbal_rate_{idx}"] = [actuator_state.gimbal_rate]
            else:  # ReactionWheel
                solution_dict[f"wheel_speed_{idx}"] = [actuator_state.wheel_speed]

        while t < t_max:

            # Ensure that all torques are zero for this type of dynamics
            for actuator in self.spacecraft.actuators:
                actuator.wheel_torque = 0.0
                if isinstance(actuator, Vscmg):
                    actuator.gimbal_torque = 0.0

            # Make sure the input state is an array
            if isinstance(state, SpacecraftState):
                state = state.to_array()

            # Calculate intermediate values
            k1 = t_step*self.compute_state_derivatives(t, state, external_torque)
            k2 = t_step*self.compute_state_derivatives(t + t_step/2, state + k1/2, external_torque)
            k3 = t_step*self.compute_state_derivatives(t + t_step/2, state + k2/2, external_torque)
            k4 = t_step*self.compute_state_derivatives(t + t_step, state + k3, external_torque)

            # DEBUGGING: print the intermediate derivatives on the first step
            if t == 0.0:
                print(f"state at [t={t}]: \n{state}")
                print(f"k1 array: {k1}\n as state: {SpacecraftState.from_array(k1, spacecraft=self.spacecraft)}")
                print(f"k2 array: {k2}\n as state: {SpacecraftState.from_array(k2, spacecraft=self.spacecraft)}")
                print(f"k3 array: {k3}\n as state: {SpacecraftState.from_array(k3, spacecraft=self.spacecraft)}")
                print(f"k4 array: {k4}\n as state: {SpacecraftState.from_array(k4, spacecraft=self.spacecraft)}")

            # Update state array for next step
            state = state + 1.0/6.0*(k1 + 2*k2 + 2*k3 + k4)

            # Check MRP magnitude and covert to shadow set if necessary
            state = SpacecraftState.from_array(state, spacecraft=self.spacecraft)

            if state.sigma_BN.norm() > 1.0:
                # TODO: this should go inside SpacecraftState
                state.sigma_BN = state.sigma_BN.convert_to_shadow_set()

            # Update the state of the spacecraft object 
            self.spacecraft.update_state(state=state)

            # Increment the time
            t = t + t_step

            # Update torque for the next step
            external_torque = torque_eq(t=t, 
                                        spacecraft=self.spacecraft)

            current_total_energy = self.CalculateTotalEnergy(spacecraft=self.spacecraft)

            N_H_total = self.CalculateTotalInertialAngularMomentum(spacecraft=self.spacecraft)

            current_total_power = self.CalculateTotalPower(spacecraft=self.spacecraft,
                                                            external_torque=external_torque)

            # Save states and controls
            solution_dict["MRP"].append(state.sigma_BN.as_array())
            solution_dict["omega_B_N"].append(state.B_omega_BN)
            solution_dict["total_energy"].append(current_total_energy)
            solution_dict["total_power"].append(current_total_power)
            solution_dict["N_H_total"].append(N_H_total)
            solution_dict["mode_value"].append(Mode.INVALID.value)

            # Update solution dictionary based on actuator type
            for idx, actuator in enumerate(self.spacecraft.actuators):
                actuator_state = actuator.state
                solution_dict[f"wheel_speed_{idx}"].append(actuator_state.wheel_speed)
                if isinstance(actuator, Vscmg):
                    solution_dict[f"gimbal_angle_{idx}"].append(actuator_state.gimbal_angle)
                    solution_dict[f"gimbal_rate_{idx}"].append(actuator_state.gimbal_rate)

            solution_dict["time"].append(t)


        # Convert lists to arrays so they're easier to work with later
        for key in solution_dict.keys():
            solution_dict[key] = np.array(solution_dict[key])

        return solution_dict


class ControlledSpacecraftDynamics(BaseSpacecraftDynamics):
    """Spacecraft dynamics with control inputs"""

    def __init__(self, 
                 spacecraft:Spacecraft,
                 control_law:callable):
        """
        Initialize the dynamics model
        
        Args:
            spacecraft (Spacecraft): The spacecraft object that is being modeled
            control_law (callable): Function to calculate control torque being applied to the spacecraft, must be of 
                form f(t, spacecraft)
        """
        self.spacecraft = spacecraft
        self.control_law = control_law

    def compute_state_derivatives(self, 
                                t: float, 
                                state_array: np.array,
                                external_torque=np.array([0,0,0])) -> np.array:
        """
        Compute state derivatives including control torque
        TODO: move external_torque to spacecraft?
        """
        # Convert array to state object
        state = SpacecraftState.from_array(state_array, spacecraft=self.spacecraft)

        # Extract state variables
        sigma_BN = state.sigma_BN
        B_omega_BN = state.B_omega_BN

        # Compute total spacecraft + actuator inertia
        B_I = self.spacecraft.B_Is.copy()

        # Add inertia from each actuator
        # NOTE: cannot use self.spacecraft.total_inertia because inertia changes in 
        # intermediate calls to Runge Kutta
        for actuator, actuator_state in zip(self.spacecraft.actuators, state.actuator_states):
            dcm_BG = actuator._compute_gimbal_frame(actuator_state)
            B_J = np.matmul(dcm_BG, np.matmul(actuator.G_J, np.transpose(dcm_BG)))
            B_I += B_J

        # Compute state derivatives with both external and control torques
        sigma_BN_dot = self._compute_sigma_dot(sigma_BN, B_omega_BN)
        
        B_omega_BN_dot = self._compute_omega_dot(B_I, state, external_torque)

        # Compute the derivatives for each actuator
        first_actuator = self.spacecraft.actuators[0]

        if isinstance(first_actuator, Vscmg):
            # VSCMG case - compute all derivatives
            wheel_speed_dot = []
            gimbal_angle_dot = []
            gimbal_rate_dot = []
            
            for actuator, actuator_state in zip(self.spacecraft.actuators, state.actuator_states):
                gimbal_angle_dot.append(actuator_state.gimbal_rate)
                
                gimbal_rate_dot_val = self._compute_gimbal_rate_dot(
                    actuator, actuator_state, B_omega_BN
                )
                gimbal_rate_dot.append(gimbal_rate_dot_val)
                
                wheel_speed_dot_val = self._compute_wheel_speed_dot(
                    actuator, actuator_state, B_omega_BN
                )
                wheel_speed_dot.append(wheel_speed_dot_val)

            state_diff = np.concatenate([
                sigma_BN_dot,
                B_omega_BN_dot,
                gimbal_angle_dot,
                gimbal_rate_dot,
                wheel_speed_dot
            ])
            
        else:  # ReactionWheel case
            # Only compute wheel speed derivatives
            wheel_speed_dot = []
            for actuator, actuator_state in zip(self.spacecraft.actuators, state.actuator_states):
                wheel_speed_dot_val = self._compute_wheel_speed_dot(
                    actuator, actuator_state, B_omega_BN
                )
                wheel_speed_dot.append(wheel_speed_dot_val)

            state_diff = np.concatenate([
                sigma_BN_dot,
                B_omega_BN_dot,
                wheel_speed_dot
            ])

        # Calculate state_dot using M matrix
        Mmat = self._compute_M_matrix(B_I, state)
        state_dot = np.linalg.solve(Mmat, state_diff)
        return state_dot

    def _compute_omega_dot(self, 
                           B_I: np.array, 
                           state:SpacecraftState, 
                           external_torque:np.array) -> np.array:
        """
        Compute the RHS of the derivative of angular velocity based on ex 4.15 from H. Schaub, J. Junkins
        Args:
            B_I (ndarray): Total inertia of the system expressed in body frame components
            state (SpacecraftState): Spacecraft state
            external_torque (ndarray): External torque vector acting on the spacecraft in body frame components [Nm]

        Returns:
            ndarray: Time derivative of angular velocity
        """

        # External torques (in body frame)
        B_L = external_torque

        B_omega_BN = state.B_omega_BN

        # Calculation based on ex 4.15 from Schaub, Junkins (f_omega)
        f_omega = np.cross(-B_omega_BN, np.matmul(B_I, B_omega_BN)) + B_L
        
        # Add the component from each VSCMG
        for (actuator, actuator_state) in zip(self.spacecraft.actuators, state.actuator_states):

            Js = actuator.G_J[0,0]
            Jt = actuator.G_J[1,1]
            Jg = actuator.G_J[2,2]

            I_Ws = actuator.I_Ws

            dcm_BG = actuator._compute_gimbal_frame(state=actuator_state)

            ws, wt, wg = actuator._compute_angular_velocity_gimbal_frame_projection(B_omega_BN, dcm_BG)

            wheel_speed = actuator_state.wheel_speed

            if isinstance(actuator_state, VscmgState):
                gimbal_rate = actuator_state.gimbal_rate # TODO: should we just add gimbal_rate to state and set it to 0 for reaction wheels?
            else:
                gimbal_rate = 0

            # Extract the gimbal frame vectors for this VSCMG (expressed in B frame)
            B_ghat_s = dcm_BG[:,0]
            B_ghat_t = dcm_BG[:,1]
            B_ghat_g = dcm_BG[:,2]
            
            g_hat_s_term = B_ghat_s * (Js * gimbal_rate * wt - (Jt - Jg) * wt * gimbal_rate)
            g_hat_t_term = B_ghat_t * ((Js * ws + I_Ws * wheel_speed) * gimbal_rate - \
                            (Jt + Jg) * ws * gimbal_rate + I_Ws * wheel_speed * wg)
            g_hat_g_term = B_ghat_g * (I_Ws * wheel_speed * wt)

            vscmg_term = g_hat_s_term + g_hat_t_term - g_hat_g_term

            f_omega -= vscmg_term

        return f_omega


    def _compute_gimbal_rate_dot(self, 
                                vscmg:Vscmg, 
                                vscmg_state:VscmgState,
                                B_omega_BN, ) -> float:
        """
        Compute the RHS of the derivative of gimbal rate based on ex 4.15 from H. Schaub, J. Junkins
        
        Args:
            vscmg (Vscmg): The VSCMG
            vscmg_state (VscmgState): The state of the vscmg with which to calculate wheel speed dot 
                (not necessarily the current state of the vscmg)
            B_omega_BN (ndarray): Angular velocity of body wrt inertial frame expressed in body 
                frame components [rad/s]

        Returns:
            float: Time derivative of gimbal rate
        """
        # Extract inertia properties
        I_Ws = vscmg.I_Ws
        Js = vscmg.G_J[0,0]
        Jt = vscmg.G_J[1,1]
        Jg = vscmg.G_J[2,2]

        # Extract properties from this VSCMG
        gimbal_torque = vscmg.gimbal_torque

        # Extract state info 
        wheel_speed = vscmg_state.wheel_speed
        dcm_BG = vscmg._compute_gimbal_frame(vscmg_state)

        # Get angular velocity projections
        ws, wt, wg = vscmg._compute_angular_velocity_gimbal_frame_projection(B_omega_BN, dcm_BG)

        # Compute the right hand side of the gimbal rate equations of motion (see ex 4.15 Schaub, Junkins)
        f_gimbal_rate = gimbal_torque + ((Js - Jt) * ws * wt) + (I_Ws * wheel_speed * wt)
        return f_gimbal_rate
    

    def _compute_wheel_speed_dot(self, 
                                actuator:WheelBase,
                                actuator_state:ActuatorState,
                                B_omega_BN:np.array,) -> float:
        """
        Compute the RHS of the derivative of wheel speed based on ex 4.15 from H. Schaub, J. Junkins
        
        Args:
            vscmg (Vscmg): The VSCMG
            vscmg_state (VscmgState): The state of the vscmg with which to calculate wheel speed dot 
                (not necessarily the current state of the vscmg)
            B_omega_BN (ndarray): Angular velocity of body wrt inertial frame expressed in body 
                frame components [rad/s]

        Returns:
            float: Time derivative of wheel speed
        """
        
        I_Ws = actuator.I_Ws

        wheel_torque = actuator.wheel_torque
        if isinstance(actuator_state, VscmgState):
            gimbal_rate = actuator_state.gimbal_rate
        else:
            gimbal_rate = 0
        dcm_BG = actuator._compute_gimbal_frame(actuator_state)

        ws, wt, wg = actuator._compute_angular_velocity_gimbal_frame_projection(B_omega_BN=B_omega_BN, dcm_BG=dcm_BG)

        f_wheel_speed = wheel_torque - I_Ws * gimbal_rate * wt 

        return f_wheel_speed


    # def _compute_angular_velocity_gimbal_frame_projection(self, 
    #                                                       B_omega_BN:np.array, 
    #                                                       dcm_BG:np.array) -> tuple[float, float, float]:
    #     """
    #     TODO: move this to actuator classes
    #     Compute the gimbal frame angular velocity projection components ws, wt, wg
        
    #     Args:
    #         B_omega_BN (ndarray): The angular velocity of the body wrt the inertial frame expressed in body frame
    #         dcm_BG (ndarray): DCM from gimbal frame to body frame
    #     Returns:
    #         tuple: Three floats ws, wt, wg
    #     """

    #     B_ghat_s = dcm_BG[:,0]
    #     B_ghat_t = dcm_BG[:,1]
    #     B_ghat_g = dcm_BG[:,2]

    #     ws = np.dot(B_ghat_s, B_omega_BN)
    #     wt = np.dot(B_ghat_t, B_omega_BN)
    #     wg = np.dot(B_ghat_g, B_omega_BN)

    #     return ws, wt, wg


    def _compute_M_matrix(self, B_I: np.array, state:SpacecraftState) -> np.array:
        """
        Compute the system matrix for the equations of motion
        """
        num_actuators = len(self.spacecraft.actuators)
        first_actuator = self.spacecraft.actuators[0]
        
        # Create base matrices
        i3 = np.eye(3)
        zero3 = np.zeros((3, 3))
        
        if isinstance(first_actuator, Vscmg):
            # VSCMG case - use existing logic
            iN = np.eye(num_actuators)
            zero3N = np.zeros((3, num_actuators))
            zeroN3 = np.zeros((num_actuators, 3))
            zeroN = np.zeros((num_actuators, num_actuators))

            GMat_s, GMat_t, GMat_g = self._compute_gimbal_frame_matrices(state)

            # Get VSCMG properties
            Js = first_actuator.G_J[0,0]
            Jt = first_actuator.G_J[1,1]
            Jg = first_actuator.G_J[2,2]    # TODO: assumes all VSCMGs have same inertia
            I_Ws = first_actuator.I_Ws

            M = np.block([
                [i3, zero3, zero3N, zero3N, zero3N],
                [zero3, B_I, zero3N, (Jg * GMat_g), (I_Ws * GMat_s)],
                [zeroN3, zeroN3, iN, zeroN, zeroN],
                [zeroN3, (Jg * GMat_g.T), zeroN, (Jg * iN), zeroN],
                [zeroN3, (I_Ws * GMat_s.T), zeroN, zeroN, (I_Ws * iN)],
            ])
            
        else:  # ReactionWheel case
            # Simpler matrix for reaction wheels (no gimbal terms)
            zero3N = np.zeros((3, num_actuators))
            zeroN3 = np.zeros((num_actuators, 3))
            iN = np.eye(num_actuators)
            
            GMat_s, _, _ = self._compute_gimbal_frame_matrices(state)
            I_Ws = first_actuator.I_Ws

            M = np.block([
                [i3, zero3, zero3N],
                [zero3, B_I, (I_Ws * GMat_s)],
                [zeroN3, (I_Ws * GMat_s.T), (I_Ws * iN)],
            ])

        return M
    

    def _compute_gimbal_frame_matrices(self, state:SpacecraftState) -> tuple[np.array, np.array, np.array]:
        """
        Compute the gimbal frame matrices for all actuators
        """
        GMat_s, GMat_t, GMat_g = [], [], []

        for actuator, actuator_state in zip(self.spacecraft.actuators, state.actuator_states):
            dcm_BG = actuator._compute_gimbal_frame(actuator_state)

            # Extract the gimbal frame vectors (expressed in B frame)
            B_ghat_s = dcm_BG[:,0]
            B_ghat_t = dcm_BG[:,1]
            B_ghat_g = dcm_BG[:,2]

            # Add the columns to the output matrices
            GMat_s.append(B_ghat_s)
            GMat_t.append(B_ghat_t)
            GMat_g.append(B_ghat_g)

        # Convert lists to 3xN matrices (stack columns)
        GMat_s = np.column_stack(GMat_s)
        GMat_t = np.column_stack(GMat_t)
        GMat_g = np.column_stack(GMat_g)

        return GMat_s, GMat_t, GMat_g

    def simulate(self, 
                t_init:float,
                t_max:float, 
                t_step:float=0.1,
                torque_eq:callable=None) -> dict:
        """
        Args:
            init_state (SpacecraftState): Initial state 
            t_init (float): Initial time corresponding to initial state
            t_max (float): End time of integrator
            t_step (float): Time step for integration
            torque_eq (callable): Function to calculate external torque being applied to the spacecraft, must be of 
                form f(t, spacecraft)

        Returns:
            (dict) Dictionary mapping variables to lists of the values they had during integration
        """

        # Initialize state and time
        state = self.spacecraft.state
        t = t_init

        external_torque = torque_eq(t=t, 
                                spacecraft=self.spacecraft)

        init_energy = self.CalculateTotalEnergy(spacecraft=self.spacecraft)
        init_H = self.CalculateTotalInertialAngularMomentum(spacecraft=self.spacecraft)
        init_power = self.CalculateTotalPower(spacecraft=self.spacecraft,
                                              external_torque=external_torque)

        # Compute required control torque in body frame
        B_L_R, pointing_mode, sigma_BR, B_omega_BR = self.control_law(t, self.spacecraft)

        # TODO: turn this into class
        # Initialize containers for storing data
        solution_dict = {
            "MRP": [state.sigma_BN.as_array()],
            "omega_B_N": [state.B_omega_BN],
            "total_energy": [init_energy],
            "total_power": [init_power],
            "N_H_total": [init_H],
            "mode_value": [Mode.INVALID.value],
            "time": [t],
            "control_torque": [B_L_R],
            "sigma_BR": [sigma_BR.as_array()],
            "B_omega_BR": [B_omega_BR]
        }

        # Initialize actuator-specific state tracking
        for idx, actuator in enumerate(self.spacecraft.actuators):
            actuator_state = actuator.state
            if isinstance(actuator, Vscmg):
                solution_dict[f"wheel_speed_{idx}"] = [actuator_state.wheel_speed]
                solution_dict[f"gimbal_angle_{idx}"] = [actuator_state.gimbal_angle]
                solution_dict[f"gimbal_rate_{idx}"] = [actuator_state.gimbal_rate]
            else:  # ReactionWheel
                solution_dict[f"wheel_speed_{idx}"] = [actuator_state.wheel_speed]


        # Integrate equations of motion
        while t < t_max:

            # Apply control torque to the actuators 
            # TODO: This is treated as constanst during each intermediate time step, should that be the case?
            # If not, it will need to be moved to compute_state_derivatives
            self.spacecraft.update_control_torque(B_L_R)

            # Make sure the input state is an array
            if isinstance(state, SpacecraftState):
                state = state.to_array()

            # Calculate intermediate values
            k1 = t_step*self.compute_state_derivatives(t, state, external_torque)
            k2 = t_step*self.compute_state_derivatives(t + t_step/2, state + k1/2, external_torque)
            k3 = t_step*self.compute_state_derivatives(t + t_step/2, state + k2/2, external_torque)
            k4 = t_step*self.compute_state_derivatives(t + t_step, state + k3, external_torque)

            # DEBUGGING: print the intermediate derivatives on the first step
            if t == 0.0:
                print(f"state at [t={t}]: \n{state}")
                print(f"k1 array: {k1}\n as state: {SpacecraftState.from_array(k1, spacecraft=self.spacecraft)}")
                print(f"k2 array: {k2}\n as state: {SpacecraftState.from_array(k2, spacecraft=self.spacecraft)}")
                print(f"k3 array: {k3}\n as state: {SpacecraftState.from_array(k3, spacecraft=self.spacecraft)}")
                print(f"k4 array: {k4}\n as state: {SpacecraftState.from_array(k4, spacecraft=self.spacecraft)}")

            # Update state array for next step
            state = state + 1.0/6.0*(k1 + 2*k2 + 2*k3 + k4)

            # Check MRP magnitude and covert to shadow set if necessary
            state = SpacecraftState.from_array(state, spacecraft=self.spacecraft)

            if state.sigma_BN.norm() > 1.0:
                # TODO: this should go inside SpacecraftState
                state.sigma_BN = state.sigma_BN.convert_to_shadow_set()

            # Update the state of the spacecraft object 
            self.spacecraft.update_state(state=state)

            # Increment the time
            t = t + t_step

            # Update torque for the next step
            external_torque = torque_eq(t=t, 
                                        spacecraft=self.spacecraft)

            current_total_energy = self.CalculateTotalEnergy(spacecraft=self.spacecraft)

            N_H_total = self.CalculateTotalInertialAngularMomentum(spacecraft=self.spacecraft)

            current_total_power = self.CalculateTotalPower(spacecraft=self.spacecraft,
                                                            external_torque=external_torque)

            # Compute required control torque in body frame
            B_L_R, pointing_mode, sigma_BR, B_omega_BR = self.control_law(t, self.spacecraft)

            # Save states and controls
            solution_dict["MRP"].append(state.sigma_BN.as_array())
            solution_dict["omega_B_N"].append(state.B_omega_BN)
            solution_dict["total_energy"].append(current_total_energy)
            solution_dict["total_power"].append(current_total_power)
            solution_dict["N_H_total"].append(N_H_total)
            solution_dict["mode_value"].append(Mode.INVALID.value)
            solution_dict["control_torque"].append(B_L_R)
            solution_dict["sigma_BR"].append(sigma_BR.as_array())
            solution_dict["B_omega_BR"].append(B_omega_BR)

            # Update solution dictionary based on actuator type
            for idx, actuator in enumerate(self.spacecraft.actuators):
                actuator_state = actuator.state
                solution_dict[f"wheel_speed_{idx}"].append(actuator_state.wheel_speed)
                if isinstance(actuator, Vscmg):
                    solution_dict[f"gimbal_angle_{idx}"].append(actuator_state.gimbal_angle)
                    solution_dict[f"gimbal_rate_{idx}"].append(actuator_state.gimbal_rate)

            solution_dict["time"].append(t)


        # Convert lists to arrays so they're easier to work with later
        for key in solution_dict.keys():
            solution_dict[key] = np.array(solution_dict[key])

        return solution_dict