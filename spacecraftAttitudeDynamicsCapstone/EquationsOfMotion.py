
from State import SpacecraftState, VscmgState
import RigidBodyKinematics as RBK 
from PointingControl import Mode
from ModifiedRodriguesParameters import MRP
from Spacecraft import Spacecraft, Vscmg

import numpy as np
import math

class VscmgDynamics:


    """
    Class to handle the equations of motion for a spacecraft with a single VSCMG
    """
    def __init__(self, 
                 spacecraft:Spacecraft):
        """
        Initialize the dynamics model
        
        Args:
            spacecraft (Spacecraft): The spacecraft object that is being modeled

        """
        self.spacecraft = spacecraft
        # self.B_I_s = B_Is
        # self.G_J = G_J
        # self.I_Ws = I_Ws
        # self.dcm_BG_init = dcm_BG_init

        # # Extract initial body frame components of the gimbal frame basis vectors
        # self.B_ghat_s_init = dcm_BG_init[:,0]
        # self.B_ghat_t_init = dcm_BG_init[:,1]
        # self.B_ghat_g_init = dcm_BG_init[:,2]
        # self.gimbal_angle_init = gimbal_angle_init

    def CalculateTotalEnergy(self,
                             spacecraft:Spacecraft) -> float:
        """
        TODO: Move this to Spacecraft class
        Calculate the total rotational kinetic energy of the system given the current state
        of the spacecraft (see eq. 4.144 in Schaub, Junkins)

        Args:
            spacecraft (Spacecraft): Current state of the entire system
        """
        
        B_omega_BN = spacecraft.state.B_omega_BN


        # Energy of spacecraft
        T_sc = 0.5 * np.dot(B_omega_BN, np.matmul(spacecraft.B_Is, B_omega_BN))

        # Total energy will be the kinetic energy of the spacecraft plus all the VSCMGs
        T_total = T_sc

        for vscmg in spacecraft.vscmgs:

            # Extract inertia variables
            Js = vscmg.G_J[0,0]
            Jt = vscmg.G_J[1,1]
            Jg = vscmg.G_J[2,2]
            I_Ws = vscmg.I_Ws
            I_Gs = Js - I_Ws

            gimbal_angle = vscmg.state.gimbal_angle
            wheel_speed = vscmg.state.wheel_speed
            gimbal_rate = vscmg.state.gimbal_rate

            dcm_BG = vscmg._compute_gimbal_frame(vscmg_state=vscmg.state)
            ws, wt, wg = self._compute_angular_velocity_gimbal_frame_projection(B_omega_BN=B_omega_BN,
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

        for vscmg in spacecraft.vscmgs:
            
            # Get current state info for this VSCMG
            gimbal_angle = vscmg.state.gimbal_angle
            wheel_speed = vscmg.state.wheel_speed
            gimbal_rate = vscmg.state.gimbal_rate
            
            # Extract inertia variables
            Js = vscmg.G_J[0,0]
            Jt = vscmg.G_J[1,1]
            Jg = vscmg.G_J[2,2]
            I_Ws = vscmg.I_Ws

            dcm_BG = vscmg._compute_gimbal_frame(vscmg_state=vscmg.state)
            ws, wt, wg = self._compute_angular_velocity_gimbal_frame_projection(B_omega_BN=B_omega_BN,
                                                                                dcm_BG=dcm_BG)

            # Angular momentum of the gimbal + wheel in gimbal frame
            G_H_vscmg = np.array([Js * ws + I_Ws * wheel_speed ,
                                Jt * wt,
                                Jg * (wg + gimbal_rate)])

            # Convert to body frame
            B_H_vscmg = np.matmul(dcm_BG, G_H_vscmg)

            # Convert to inertial frame
            N_H_vscmg = np.matmul(dcm_NB, B_H_vscmg)

            # Add to total angular momentum
            N_H_total += N_H_vscmg

        return N_H_total

    def TorqueFreeEquationsOfMotion(self, 
                                    t:float, 
                                    state_array:np.array, 
                                    external_torque=np.array([0,0,0]),) -> np.array:
                                    # wheel_torque=0.0, 
                                    # gimbal_torque=0.0) -> np.array:
        """
        Compute the time derivatives for the spacecraft state in the absence of external torques, wheel motor torque, 
        and gimbal motor torque
        
        Args:
            t (float): Current time (for time-dependent systems)
            state (ndarray): Current state of the spacecraft
            external_torque (ndarray, optional): External control torque. Defaults to None.
            wheel_torque (float, optional): Torque applied to the VSCMG wheel. Defaults to None.
            gimbal_torque (float, optional): Torque applied to the VSCMG gimbal. Defaults to None.
            
        Returns:
            np.array: Time derivative of the state
        """

        # TODO: we cannot use spacecraft.state anywhere in this function becuase that is not updated until
        # after the RK4 integration, this function needs to be able to calculate intermediate steps
        state = SpacecraftState.from_array(state_array)
        # print(f"  > state: {state}")

        # Extract state variables
        sigma_BN = state.sigma_BN
        B_omega_BN = state.B_omega_BN

        # Compute the total spacecraft + VSCMG inertia
        B_I = self.spacecraft.B_Is.copy()

        # Add the inertia from each VSCMG
        for (vscmg, vscmg_state) in zip(self.spacecraft.vscmgs, state.vscmg_states):

            # Calculate the gimbal-to-body DCM
            dcm_BG = vscmg._compute_gimbal_frame(vscmg_state=vscmg_state)

            # Convert the gimbal + wheel inertia [J] to body frame
            B_J = np.matmul(dcm_BG, np.matmul(vscmg.G_J, np.transpose(dcm_BG)))

            B_I += B_J

        # Compute right-hand side of each state's equations of motion
        sigma_BN_dot = self._compute_sigma_dot(sigma_BN, B_omega_BN)
        B_omega_BN_dot = self._compute_omega_dot(B_I=B_I,
                                                 state=state,
                                                # B_omega_BN=B_omega_BN,
                                                # dcm_BG=dcm_BG,
                                                # wheel_speed=wheel_speed, 
                                                # gimbal_rate=gimbal_rate, 
                                                external_torque=external_torque, )
                                                # wheel_torque=wheel_torque, 
                                                # gimbal_torque=gimbal_torque)

        # Compute wheel speed dot, gimbal angle dot, and gimbal rate dot for each VSCMG
        wheel_speed_dot, gimbal_angle_dot, gimbal_rate_dot = [], [], []
        for (vscmg, vscmg_state) in zip(self.spacecraft.vscmgs, state.vscmg_states):

            gimbal_angle_dot.append(vscmg_state.gimbal_rate)

            gimbal_rate_dot_vscmg = self._compute_gimbal_rate_dot(vscmg=vscmg,
                                                                  vscmg_state=vscmg_state, 
                                                                  B_omega_BN=B_omega_BN)
            gimbal_rate_dot.append(gimbal_rate_dot_vscmg)

            wheel_speed_dot_vscmg = self._compute_wheel_speed_dot(vscmg=vscmg,
                                                                  vscmg_state=vscmg_state,
                                                                B_omega_BN=B_omega_BN)
            wheel_speed_dot.append(wheel_speed_dot_vscmg)

        # # Combine all derivatives
        # state_dot = np.concatenate([
        #     sigma_BN_dot,
        #     B_omega_BN_dot,
        #     [wheel_speed_dot],
        #     [gimbal_angle_dot],
        #     [gimbal_rate_dot]
        # ])


        Mmat = self._compute_M_matrix(B_I=B_I, state=state)

        state_diff = np.concatenate([
            sigma_BN_dot,
            B_omega_BN_dot,
            gimbal_angle_dot,
            gimbal_rate_dot,
            wheel_speed_dot
        ])# .reshape(-1, 1)   # shape as column vector
        # print(f"    > state_diff: {state_diff}")
        # Simultaneously solve system of equations from the equations of motion
        # state_dot = np.matmul(np.linalg.inv(Mmat), state_diff )
        # print(f"    > Mmat shape: {Mmat.shape} state_diff shape: {state_diff.shape}")
        state_dot = np.linalg.solve(Mmat, state_diff)
        # state_dot = np.matmul(np.linalg.pinv(Mmat), state_diff )
        # print(f"    > state_dot: {state_dot.flatten()}")
        return state_dot

    # def _compute_gimbal_frame(self, vscmg:Vscmg) -> np.array:
    #                         #   gimbal_angle:float) -> np.array:
    #     """
    #     TODO: move this to Vscmg class...
    #     Compute the gimbal to body DCM [BG] based on the current gimbal angle
        
    #     Args:
    #         gimbal_angle (float): Current gimbal angle [rad]
            
    #     Returns:
    #         ndarray: 3x3 DCM describing transformation from gimbal to body frame 
    #     """
    #     # Get the current gimbal angle for this VSCMG
    #     gimbal_angle = vscmg.state.gimbal_angle

    #     # Get the initial gimbal angle and corresponding DCM for this VSCMG
    #     gamma0 = vscmg.gimbal_angle_init

    #     dcm_BG_init = vscmg.dcm_BG_init
    #     B_ghat_s_init = dcm_BG_init[:,0]
    #     B_ghat_t_init = dcm_BG_init[:,1]
    #     B_ghat_g_init = dcm_BG_init[:,2]

    #     B_ghat_s = ((math.cos(gimbal_angle - gamma0) * B_ghat_s_init) + 
    #                 (math.sin(gimbal_angle - gamma0) * B_ghat_t_init))
    #     B_ghat_t = ((-math.sin(gimbal_angle - gamma0) * B_ghat_s_init) + 
    #                 (math.cos(gimbal_angle - gamma0) * B_ghat_t_init))
    #     B_ghat_g = B_ghat_g_init

    #     # Construct the [GB] DCM
    #     # Note that [BG] = {B_g1, B_g2, B_g3} (column vectors) so the transpose [GB]
    #     # can be created with row vectors
    #     dcm_GB = np.array([B_ghat_s,
    #                        B_ghat_t,
    #                        B_ghat_g])
    #     dcm_BG = np.transpose(dcm_GB)

    #     return dcm_BG
    
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
    
    def _compute_omega_dot(self, 
                           B_I:np.array,
                           state:SpacecraftState,
                           #B_omega_BN:np.array,
                           #dcm_BG:np.array, 
                           #wheel_speed:float, 
                           #gimbal_rate:float, 
                           external_torque:np.array, ) -> np.array:
                           #wheel_torque:float, 
                           #gimbal_torque:float) -> np.array:
        """
        Compute the derivative of the angular velocity based on equation 4.137 from H. Schaub, J. Junkins
        Args:
            B_I (ndarray): Total inertia of the system expressed in body frame components
            B_omega_BN (ndarray): Ang vel of entire system wrt inertial frame expressed in body frame components
            dcm_BG (ndarray): 3x3 DCM from G frame to B frame
            wheel_speed (float): Speed of VSCMG wheel [rad/s]
            gimbal_rate (float): Rate of change of gimbal angle [rad/s]
            external_torque (ndarray): Control torque vector in body frame components [Nm]
            wheel_torque (float): Wheel torque [Nm]
            gimbal_torque (float): Gimbal torque [Nm]

        Returns:
            ndarray: Time derivative of angular velocity
        """

        # # Extract inertia variables
        # Js = self.G_J[0,0]
        # Jt = self.G_J[1,1]
        # Jg = self.G_J[2,2]
        # I_Ws = self.I_Ws
        # I_Gs = Js - I_Ws

        # # Extract the gimbal frame vectors (expressed in B frame)
        # B_ghat_s = dcm_BG[:,0]
        # B_ghat_t = dcm_BG[:,1]
        # B_ghat_g = dcm_BG[:,2]

        # # Compute the projection of angular velocity on the gimbal frame
        # ws, wt, wg = self._compute_angular_velocity_gimbal_frame_projection(B_omega_BN, dcm_BG)

        # # Compute the VSCGM gyroscopic terms
        # ghat_s_term = B_ghat_s * (wheel_torque + (I_Gs * gimbal_rate * wt) - 
        #                           ((Jt - Jg) * wt * gimbal_rate))
        # ghat_t_term = B_ghat_t * ((Js * ws + (I_Ws * wheel_speed)) * gimbal_rate - 
        #                           ((Jt + Jg) * ws * gimbal_rate) +
        #                           (I_Ws * wheel_speed * wg))
        # ghat_g_term = B_ghat_g * (gimbal_torque + ((Js - Jt) * ws * wt))

        # External torques (in body frame)
        B_L = external_torque

        # Compute angular momentum derivative (expressed in B frame)
        # This is the RHS of the equation
        # B_H_dot = np.cross(-B_omega_BN, np.matmul(B_I, B_omega_BN)) - ghat_s_term - ghat_t_term - ghat_g_term + B_L

        B_omega_BN = state.B_omega_BN

        # Calculation based on ex 4.15 from Schaub, Junkins
        B_H_dot = np.cross(-B_omega_BN, np.matmul(B_I, B_omega_BN)) + B_L
        
        # Add the component from each VSCMG
        for (vscmg, vscmg_state) in zip(self.spacecraft.vscmgs, state.vscmg_states):

            Js = vscmg.G_J[0,0]
            Jt = vscmg.G_J[1,1]
            Jg = vscmg.G_J[2,2]

            I_Ws = vscmg.I_Ws

            dcm_BG = vscmg._compute_gimbal_frame(vscmg_state=vscmg_state)

            ws, wt, wg = self._compute_angular_velocity_gimbal_frame_projection(B_omega_BN, dcm_BG)

            gimbal_rate = vscmg_state.gimbal_rate
            wheel_speed = vscmg_state.wheel_speed

            # Extract the gimbal frame vectors for this VSCMG (expressed in B frame)
            B_ghat_s = dcm_BG[:,0]
            B_ghat_t = dcm_BG[:,1]
            B_ghat_g = dcm_BG[:,2]
            
            g_hat_s_term = B_ghat_s * (Js * gimbal_rate * wt - (Jt - Jg) * wt * gimbal_rate)
            g_hat_t_term = B_ghat_t * ((Js * ws + I_Ws * wheel_speed) * gimbal_rate - \
                            (Jt + Jg) * ws * gimbal_rate + I_Ws * wheel_speed * wg)
            g_hat_g_term = B_ghat_g * (I_Ws * wheel_speed * wt)

            vscmg_term = g_hat_s_term + g_hat_t_term - g_hat_g_term

            B_H_dot -= vscmg_term

        # Compute the augmented inertia term (on LHS of equation)
        # adjusted_inertia = B_I - (I_Ws * np.outer(B_ghat_s, B_ghat_s)) - (Jg * np.outer(B_ghat_g, B_ghat_g))

        # Isolate omega_dot on the LHS of the equation
        # omega_dot = np.dot(np.linalg.inv(adjusted_inertia), B_H_dot)
        # omega_dot = np.dot(np.linalg.inv(B_I), B_H_dot)

        # Retrun B_H_dot (the right hand side of the equation)
        # [I] omega_dot = -omega x [I] x omega + ...

        return B_H_dot
    
    def _compute_wheel_speed_dot(self, 
                                # dcm_BG:np.array, 
                                vscmg:Vscmg,
                                vscmg_state:VscmgState,
                                B_omega_BN:np.array,) -> float:
                                # B_omega_BN_dot:np.array, 
                                # gimbal_rate_dot:float,) -> float:
                                # wheel_torque:float) -> float:
        """
        Compute the derivative of the wheel speed.
        
        Args:
            dcm_BG (ndarray): DCM of gimbal frame to body frame
            B_omega_BN_dot (ndarray): angular acceleration experessed in body frame
            wheel_torque (float): Torque applied to the wheel
            
        Returns:
            float: Time derivative of wheel speed
        """
        
        I_Ws = vscmg.I_Ws

        wheel_torque = vscmg.wheel_torque
        gimbal_rate = vscmg_state.gimbal_rate
        dcm_BG = vscmg._compute_gimbal_frame(vscmg_state=vscmg_state)

        ws, wt, wg = self._compute_angular_velocity_gimbal_frame_projection(B_omega_BN=B_omega_BN, dcm_BG=dcm_BG)
        
        # wheel_speed_dot = (wheel_torque / self.I_Ws) - np.dot(B_ghat_s, B_omega_BN_dot) - (gimbal_rate * wt)
        # Compute the right hand side of the wheel speed equations of motion (see ex 4.15 Schaub, Junkins)
        # print(f"_compute_wheel_speed\n")
        # print(f"    I_Ws: {I_Ws} gimbal_rate: {gimbal_rate}, \n")
        wheel_speed_dot = wheel_torque - I_Ws * gimbal_rate * wt 

        return wheel_speed_dot
    
    def _compute_gimbal_rate_dot(self, 
                                 vscmg:Vscmg, 
                                 vscmg_state:VscmgState,
                                #  B_omega_BN_dot, 
                                B_omega_BN, ) -> float:
                                # wheel_speed, 
                                # gimbal_torque):
        """
        Compute the derivative of the gimbal rate.
        
        Args:
            gimbal_torque (float): Torque applied to the gimbal
            
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
        dcm_BG = vscmg._compute_gimbal_frame(vscmg_state=vscmg_state)

        # # ghat basis vector of gimbal frame expressed in body frame coordinates
        # B_ghat_g = dcm_BG[:,2]

        # Get angular velocity projections
        ws, wt, wg = self._compute_angular_velocity_gimbal_frame_projection(B_omega_BN, dcm_BG)

        # gimbal_rate_dot = (1.0 / Jg * (gimbal_torque + (I_Ws * wheel_speed * wt) + ((Js - Jt) * ws * wt)) - 
        #                                 np.dot(B_ghat_g, B_omega_BN_dot))
        # Compute the right hand side of the gimbal rate equations of motion (see ex 4.15 Schaub, Junkins)
        gimbal_rate_dot = gimbal_torque + ((Js - Jt) * ws * wt) + (I_Ws * wheel_speed * wt)
        return gimbal_rate_dot
    
    def _compute_angular_velocity_gimbal_frame_projection(self, B_omega_BN:np.array, dcm_BG:np.array) -> tuple[float, float, float]:
        """
        Compute the gimbal frame angular velocity projection components ws, wt, wg
        
        Args:
            B_omega_BN (ndarray): The angular velocity of the body wrt the inertial frame expressed in body frame
            dcm_BG (ndarray): DCM from gimbal frame to body frame
        Returns:
            tuple: Three floats ws, wt, wg
        """

        # omega_BN_G = np.dot(dcm_BG.T,B_omega_BN)
        # ws = omega_BN_G[0]
        # wt = omega_BN_G[1]
        # wg = omega_BN_G[2]


        B_ghat_s = dcm_BG[:,0]
        B_ghat_t = dcm_BG[:,1]
        B_ghat_g = dcm_BG[:,2]

        ws = np.dot(B_ghat_s, B_omega_BN)
        wt = np.dot(B_ghat_t, B_omega_BN)
        wg = np.dot(B_ghat_g, B_omega_BN)

        return ws, wt, wg

    def _compute_gimbal_frame_matrices(self, state:SpacecraftState) -> np.array:
        """
        
        """

        GMat_s, GMat_t, GMat_g = [], [], []

        for (vscmg, vscmg_state) in zip(self.spacecraft.vscmgs, state.vscmg_states):

            dcm_BG = vscmg._compute_gimbal_frame(vscmg_state=vscmg_state)

            # Extract the gimbal frame vectors for this VSCMG (expressed in B frame)
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

    # def _compute_vscmg_psuedo_torques(self) -> tuple[np.array, np.array, np.array]:
    #     """
        
        
    #     """

    #     tau_s, tau_t, tau_g = [], [], []

    #     for vscmg in self.spacecraft.vscmgs:
    #         I_Ws = vscmg.I_Ws
    #         Omega_dot = vscmg.state.wheel_speed

    def _compute_vscmg_combined_inertia_matrices(self) -> tuple[np.array, np.array, np.array, np.array]:
        """
        
        """
        
        Js_mat, Jt_mat, Jg_mat, I_Ws_mat = [], [], [], []

        for vscmg in self.spacecraft.vscmgs:

            Js_mat.append(vscmg.G_J[0,0])
            Jt_mat.append(vscmg.G_J[1,1])
            Jg_mat.append(vscmg.G_J[2,2])
            I_Ws_mat.append(vscmg.I_Ws)

        Js_mat = np.diag(Js_mat)
        Jt_mat = np.diag(Jt_mat)
        Jg_mat = np.diag(Jg_mat)
        I_Ws_mat = np.diag(I_Ws_mat)

        return Js_mat, Jt_mat, Jg_mat, I_Ws_mat



    def _compute_M_matrix(self, B_I: np.array, state:SpacecraftState) -> np.array:
        """
        Function for computing the left-hand side system matrix for the VSCMG equations of motion

        Arguments:
            B_I (np.array): Full system moment of inertia matrix expressed in body frame

        Returns:
            (6 + 3N) x (6 + 3N) matrix
        """

        num_vscmgs = len(self.spacecraft.vscmgs)

        # Create the simple matrices that will be used to compose the M matrix
        i3 = np.eye(3)
        zero3 = np.zeros((3, 3))
        iN = np.eye(num_vscmgs)
        zero3N = np.zeros((3, num_vscmgs))
        zeroN3 = np.zeros((num_vscmgs, 3))
        zeroN = np.zeros((num_vscmgs, num_vscmgs))

        # Calculate the large gimbal frame matrices
        GMat_s, GMat_t, GMat_g = self._compute_gimbal_frame_matrices(state=state)

        # Calculate matrix of inertias of all vscmgs
        Js_mat, Jt_mat, Jg_mat, I_Ws_mat = self._compute_vscmg_combined_inertia_matrices()

        # Function to handle matrix multiplication based on dimensions
        def safe_matmul(A, B):
            if A.shape[1] == B.shape[0]:
                return np.matmul(A, B)
            elif A.shape == (1, 1):  # Scalar multiplication
                return A.item() * B
            else:
                raise ValueError(f"Shape mismatch for multiplication: {A.shape} and {B.shape}")

        # Construct the M matrix
        # Row 1: sigma_dot
        # Row 2: omega_dot
        # Row 3: gimbal_angle_dot
        # Row 4: gimbal_rate_dot
        # Row 5: wheel_speed_dot
        M = np.block([
            [i3, zero3, zero3N, zero3N, zero3N],
            [zero3, B_I, zero3N, safe_matmul(Jg_mat, GMat_g), safe_matmul(I_Ws_mat, GMat_s)],
            [zeroN3, zeroN3, iN, zeroN, zeroN],
            [zeroN3, safe_matmul(Jg_mat, GMat_g.T), zeroN, safe_matmul(Jg_mat, iN), zeroN],
            [zeroN3, safe_matmul(I_Ws_mat, GMat_s.T), zeroN, zeroN, safe_matmul(I_Ws_mat, iN)],
        ])

        # Check if M has the correct dimensions
        expected_shape = (6 + 3 * num_vscmgs, 6 + 3 * num_vscmgs)
        if M.shape != expected_shape:
            raise ValueError(f"Matrix M has shape {M.shape}, expected {expected_shape}")

        return M



    def simulate(self, 
                t_init:float,
                t_max:float, 
                t_step:float=0.1,) -> dict:
        """
        4th order RK4 integrator 
        Note this integrator does slightly more than just integrate equations of motion, it also
        calculates the necessary control torque to apply to the equations of motion

        Args:
            init_state (SpacecraftState): Initial state 
            t_init (float): Initial time corresponding to initial state
            t_max (float): End time of integrator
            t_step (float): Time step for integration
            diff_eq (callable): Function handle for the equations of motion to be integrated, must be of 
                form f(t, state, u, wheel_torque, gimbal_torque)
            ctrl_func (callable): Function handle for generating the control torque to be applied to the 
                differential equation, must be of form g(t, state, gains)
                # TODO: Put ctrl_func inside diff_eq?

        Returns:
            (dict) Dictionary mapping variables to lists of the values they had during integration
        """
        

        # Initialize state and time
        state = self.spacecraft.state
        t = t_init

        external_torque = np.zeros(3)

        init_energy = self.CalculateTotalEnergy(spacecraft=self.spacecraft)
        init_H = self.CalculateTotalInertialAngularMomentum(spacecraft=self.spacecraft)
        
        # TODO: turn this into class
        # Initialize containers for storing data
        solution_dict = {}
        solution_dict["MRP"] = [state.sigma_BN.as_array()]   # sigma_BN
        solution_dict["omega_B_N"] = [state.B_omega_BN] # B_omega_BN
        for idx in range(len(self.spacecraft.vscmgs)):
            vscmg = self.spacecraft.vscmgs[idx]
            vscmg_state = vscmg.state
            # TODO: replace this loop with a loop over the Vscmg objects attached to self.spacecraft
            solution_dict[f"wheel_speed_{idx}"] = [vscmg_state.wheel_speed]
            solution_dict[f"gimbal_angle_{idx}"] = [vscmg_state.gimbal_angle]
            solution_dict[f"gimbal_rate_{idx}"] = [vscmg_state.gimbal_rate]
        
        solution_dict["total_energy"] = [init_energy]
        solution_dict["N_H_total"] = [init_H]
        solution_dict["mode_value"] = [Mode.INVALID.value]
        solution_dict["time"] = [t]

        while t < t_max:

            # Make sure the input state is an array
            if isinstance(state, SpacecraftState):
                state = state.to_array()

            # Calculate intermediate values
            # print(f" > t: {t}")
            # print(f"STARTING k1...")
            k1 = t_step*self.TorqueFreeEquationsOfMotion(t, state, external_torque)
            # if t == 0.0:
            #     print(f"   k1 array: {k1}\n ")#as state: {SpacecraftState.from_array(k1)}")
            # print(f"STARTING k2...")
            k2 = t_step*self.TorqueFreeEquationsOfMotion(t + t_step/2, state + k1/2, external_torque)
            # if t == 0.0:
            #     print(f"   k2 array: {k2}\n ")#as state: {SpacecraftState.from_array(k2)}")
            # print(f"STARTING k3...")
            k3 = t_step*self.TorqueFreeEquationsOfMotion(t + t_step/2, state + k2/2, external_torque)
            # if t == 0.0:
            #     print(f"   k3 array: {k3}\n ")#as state: {SpacecraftState.from_array(k3)}")
            # print(f"STARTING k4...")
            k4 = t_step*self.TorqueFreeEquationsOfMotion(t + t_step, state + k3, external_torque)
            # if t == 0.0:
            #     print(f"   k4 array: {k4}\n ")#as state: {SpacecraftState.from_array(k4)}")

            # DEBUGGING: print the intermediate derivatives on the first step
            if t == 0.0:
                print(f"state at [t={t}]: \n{state}")
                print(f"k1 array: {k1}\n as state: {SpacecraftState.from_array(k1)}")
                print(f"k2 array: {k2}\n as state: {SpacecraftState.from_array(k2)}")
                print(f"k3 array: {k3}\n as state: {SpacecraftState.from_array(k3)}")
                print(f"k4 array: {k4}\n as state: {SpacecraftState.from_array(k4)}")

            # Update state
            state = state + 1.0/6.0*(k1 + 2*k2 + 2*k3 + k4)

            # Check MRP magnitude and covert to shadow set if necessary
            # TODO: update the state of self.spacecraft.state?
            state = SpacecraftState.from_array(state)

            if state.sigma_BN.norm() > 1.0:
                # TODO: this should go inside SpacecraftState
                state.sigma_BN = state.sigma_BN.convert_to_shadow_set()

            self.spacecraft.update_state(state=state)

            # Increment the time
            t = t + t_step

            total_energy = self.CalculateTotalEnergy(spacecraft=self.spacecraft)

            N_H_total = self.CalculateTotalInertialAngularMomentum(spacecraft=self.spacecraft)

            # Save states and controls
            solution_dict["MRP"].append(state.sigma_BN.as_array())
            solution_dict["omega_B_N"].append(state.B_omega_BN)
            solution_dict["total_energy"].append(total_energy)
            solution_dict["N_H_total"].append(N_H_total)
            solution_dict["mode_value"].append(Mode.INVALID.value)

            for idx in range(len(self.spacecraft.vscmgs)):
                vscmg = self.spacecraft.vscmgs[idx]
                vscmg_state = vscmg.state
                # TODO: replace this loop with a loop over the Vscmg objects attached to self.spacecraft
                solution_dict[f"wheel_speed_{idx}"].append(vscmg_state.wheel_speed)
                solution_dict[f"gimbal_angle_{idx}"].append(vscmg_state.gimbal_angle)
                solution_dict[f"gimbal_rate_{idx}"].append(vscmg_state.gimbal_rate)

            solution_dict["time"].append(t)


        # Convert lists to arrays so they're easier to work with later
        for key in solution_dict.keys():
            solution_dict[key] = np.array(solution_dict[key])

        return solution_dict