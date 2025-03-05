
from State import StateVscmg
import RigidBodyKinematics as RBK 
from PointingControl import Mode
from ModifiedRodriguesParameters import MRP

import numpy as np
import math
from scipy.integrate import solve_ivp

class VscmgDynamics:


    """
    Class to handle the equations of motion for a spacecraft with a single VSCMG
    """
    def __init__(self, 
                 B_Is:np.array, 
                 G_J:np.array, 
                 I_Ws:float, 
                 dcm_BG_init:np.array, 
                 gimbal_angle_init:float):
        """
        Initialize the dynamics model with the system parameters.
        
        Args:
            B_Is (ndarray): 3x3 inertia tensor of the spacecraft hub + P.A.T. terms of gimbal 
                and wheel (represented in body frame) [kgm2]
            G_J (ndarray): 3x3 inertia matrix of gimbal + wheel system (represented in gimbal 
            frame) [kgm2]
            I_Ws (float): Moment of inertia of the wheel about the spin axis [kgm^2]
            gimbal_angle_init (float):

        """
        self.B_I_s = B_Is
        self.G_J = G_J
        self.I_Ws = I_Ws
        self.dcm_BG_init = dcm_BG_init

        # Extract initial body frame components of the gimbal frame basis vectors
        self.B_ghat_s_init = dcm_BG_init[:,0]
        self.B_ghat_t_init = dcm_BG_init[:,1]
        self.B_ghat_g_init = dcm_BG_init[:,2]
        print(f"Initial gimbal frame:")
        print(f" dcm_BG: {self.dcm_BG_init}")

        print(f" B_ghat_s: {self.B_ghat_s_init}")
        print(f" B_ghat_t: {self.B_ghat_t_init}")
        print(f" B_ghat_g: {self.B_ghat_g_init}")

        self.gimbal_angle_init = gimbal_angle_init

    def CalculateTotalEnergy(self,
                             state:StateVscmg) -> float:
        """
        Calculate the total rotational kinetic energy of the system (see eq. 4.144 in Schaub, Junkins)

        Args:
            state (StateVscmg): Current state of the entire system
        """
        # Extract inertia variables
        Js = self.G_J[0,0]
        Jt = self.G_J[1,1]
        Jg = self.G_J[2,2]
        I_Ws = self.I_Ws
        I_Gs = Js - I_Ws

        gimbal_angle = state.gimbal_angle
        B_omega_BN = state.B_omega_BN
        wheel_speed = state.wheel_speed
        gimbal_rate = state.gimbal_rate

        dcm_BG = self._compute_gimbal_frame(gimbal_angle)
        ws, wt, wg = self._compute_angular_velocity_gimbal_frame_projection(B_omega_BN=B_omega_BN,
                                                                            dcm_BG=dcm_BG)

        # Energy of spacecraft
        T_sc = 0.5 * np.dot(B_omega_BN, np.matmul(self.B_I_s, B_omega_BN))

        # Energy of gimbal
        T_gimbal = 0.5 * ((I_Gs * ws**2) + (Jt * wt**2) + (Jg * (wg + gimbal_rate)**2 ))

        # Energy of wheel
        T_wheel = 0.5 * (I_Ws * (wheel_speed + ws)**2)

        T_total = T_sc + T_gimbal + T_wheel

        return T_total 

    def CalculateTotalInertialAngularMomentum(self, state:StateVscmg) -> np.array:
        """
        Calculate the total angular momentum of the system in inertial frame components 
        (derived from eq 4.113 in Schuab, Junkins)

        Args:
            state (StateVscmg): Current state of the entire system
        
        Reutrns:
            ndarray: Angular momentum vector of the total system expressed in the inertial frame [Nms]
        """

        sigma_BN = state.sigma_BN
        gimbal_angle = state.gimbal_angle
        B_omega_BN = state.B_omega_BN
        wheel_speed = state.wheel_speed
        gimbal_rate = state.gimbal_rate
        # Extract inertia variables
        Js = self.G_J[0,0]
        Jt = self.G_J[1,1]
        Jg = self.G_J[2,2]
        I_Ws = self.I_Ws
        I_Gs = Js - I_Ws

        dcm_BN = sigma_BN.MRP2C()
        dcm_NB = np.transpose(dcm_BN)
        dcm_BG = self._compute_gimbal_frame(gimbal_angle=gimbal_angle)
        ws, wt, wg = self._compute_angular_velocity_gimbal_frame_projection(B_omega_BN=B_omega_BN,
                                                                            dcm_BG=dcm_BG)

        # Angular momentum of the body/spacecraft
        B_H_B = np.matmul(self.B_I_s, B_omega_BN)
        N_H_B = np.matmul(dcm_NB, B_H_B)

        # Angular momentum of the gimbal + wheel in gimbal frame
        G_H_vscmg = np.array([Js * ws + I_Ws * wheel_speed ,
                              Jt * wt,
                              Jg * (wg + gimbal_rate)])

        B_H_vscmg = np.matmul(dcm_BG, G_H_vscmg)
        N_H_vscmg = np.matmul(dcm_NB, B_H_vscmg)

        # Total angular momentum
        N_H_total = N_H_B + N_H_vscmg

        return N_H_total

    def TorqueFreeEquationsOfMotion(self, 
                                    t:float, 
                                    state_array:np.array, 
                                    control_torque=np.array([0,0,0]), 
                                    wheel_torque=0.0, 
                                    gimbal_torque=0.0) -> np.array:
        """
        Compute the time derivatives for the spacecraft state in the absence of external torques, wheel motor torque, 
        and gimbal motor torque
        
        Args:
            t (float): Current time (for time-dependent systems)
            state (ndarray): Current state of the spacecraft
            control_torque (ndarray, optional): External control torque. Defaults to None.
            wheel_torque (float, optional): Torque applied to the VSCMG wheel. Defaults to None.
            gimbal_torque (float, optional): Torque applied to the VSCMG gimbal. Defaults to None.
            
        Returns:
            np.array: Time derivative of the state
        """

        state = StateVscmg.from_array(state_array)

        # Extract state variables
        sigma_BN = state.sigma_BN
        B_omega_BN = state.B_omega_BN
        wheel_speed = state.wheel_speed
        gimbal_angle = state.gimbal_angle
        gimbal_rate = state.gimbal_rate
        
        # Calculate the gimbal-to-body DCM
        dcm_BG = self._compute_gimbal_frame(gimbal_angle)
        
        # Convert the gimbal + wheel inertia [J] to body frame
        B_J = np.matmul(dcm_BG, np.matmul(self.G_J, np.transpose(dcm_BG)))

        # Compute the total spacecraft + VSCMG inertia
        B_I = self.B_I_s + B_J

        # Compute time derivatives of each state component
        sigma_BN_dot = self._compute_sigma_dot(sigma_BN, B_omega_BN)
        B_omega_BN_dot = self._compute_omega_dot(B_I=B_I,
                                                B_omega_BN=B_omega_BN,
                                                dcm_BG=dcm_BG,
                                                wheel_speed=wheel_speed, 
                                                gimbal_rate=gimbal_rate, 
                                                control_torque=control_torque, 
                                                wheel_torque=wheel_torque, 
                                                gimbal_torque=gimbal_torque)
        wheel_speed_dot = self._compute_wheel_speed_dot(dcm_BG=dcm_BG, 
                                                        B_omega_BN=B_omega_BN,
                                                        B_omega_BN_dot=B_omega_BN_dot, 
                                                        gimbal_rate=gimbal_rate,
                                                        wheel_torque=wheel_torque)
        gimbal_angle_dot = gimbal_rate
        gimbal_rate_dot = self._compute_gimbal_rate_dot(dcm_BG, B_omega_BN_dot, B_omega_BN, wheel_speed, gimbal_torque)

        # Combine all derivatives
        state_dot = np.concatenate([
            sigma_BN_dot,
            B_omega_BN_dot,
            [wheel_speed_dot],
            [gimbal_angle_dot],
            [gimbal_rate_dot]
        ])
        
        return state_dot

    def _compute_gimbal_frame(self, gimbal_angle:float) -> np.array:
        """
        Compute the gimbal to body DCM [BG] based on the current gimbal angle
        
        Args:
            gimbal_angle (float): Current gimbal angle [rad]
            
        Returns:
            ndarray: 3x3 DCM describing transformation from gimbal to body frame 
        """

        gamma0 = self.gimbal_angle_init
        B_ghat_s = ((math.cos(gimbal_angle - gamma0) * self.B_ghat_s_init) + 
                    (math.sin(gimbal_angle - gamma0) * self.B_ghat_t_init))
        B_ghat_t = ((-math.sin(gimbal_angle - gamma0) * self.B_ghat_s_init) + 
                    (math.cos(gimbal_angle - gamma0) * self.B_ghat_t_init))
        B_ghat_g = self.B_ghat_g_init

        # Construct the [GB] DCM
        # Note that [BG] = {B_g1, B_g2, B_g3} (column vectors) so the transpose [GB]
        # can be created with row vectors
        dcm_GB = np.array([B_ghat_s,
                           B_ghat_t,
                           B_ghat_g])
        dcm_BG = np.transpose(dcm_GB)

        return dcm_BG
    
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
                           B_omega_BN:np.array, 
                           dcm_BG:np.array, 
                           wheel_speed:float, 
                           gimbal_rate:float, 
                           control_torque:np.array, 
                           wheel_torque:float, 
                           gimbal_torque:float) -> np.array:
        """
        Compute the derivative of the angular velocity based on equation 4.137 from H. Schaub, J. Junkins
        Args:
            B_I (ndarray): Total inertia of the system expressed in body frame components
            B_omega_BN (ndarray): Ang vel of entire system wrt inertial frame expressed in body frame components
            dcm_BG (ndarray): 3x3 DCM from G frame to B frame
            wheel_speed (float): Speed of VSCMG wheel [rad/s]
            gimbal_rate (float): Rate of change of gimbal angle [rad/s]
            control_torque (ndarray): Control torque vector in body frame components [Nm]
            wheel_torque (float): Wheel torque [Nm]
            gimbal_torque (float): Gimbal torque [Nm]

        Returns:
            ndarray: Time derivative of angular velocity
        """

        # Extract inertia variables
        Js = self.G_J[0,0]
        Jt = self.G_J[1,1]
        Jg = self.G_J[2,2]
        I_Ws = self.I_Ws
        I_Gs = Js - I_Ws

        # Extract the gimbal frame vectors (expressed in B frame)
        B_ghat_s = dcm_BG[:,0]
        B_ghat_t = dcm_BG[:,1]
        B_ghat_g = dcm_BG[:,2]

        # Compute the projection of angular velocity on the gimbal frame
        ws, wt, wg = self._compute_angular_velocity_gimbal_frame_projection(B_omega_BN, dcm_BG)

        # Compute the VSCGM gyroscopic terms
        ghat_s_term = B_ghat_s * (wheel_torque + (I_Gs * gimbal_rate * wt) - 
                                  ((Jt - Jg) * wt * gimbal_rate))
        ghat_t_term = B_ghat_t * ((Js * ws + (I_Ws * wheel_speed)) * gimbal_rate - 
                                  ((Jt + Jg) * ws * gimbal_rate) +
                                  (I_Ws * wheel_speed * wg))
        ghat_g_term = B_ghat_g * (gimbal_torque + ((Js - Jt) * ws * wt))

        # External torques (in body frame)
        B_L = control_torque

        # Compute angular momentum derivative (expressed in B frame)
        # This is the RHS of the equation
        B_H_dot = np.cross(-B_omega_BN, np.matmul(B_I, B_omega_BN)) - ghat_s_term - ghat_t_term - ghat_g_term + B_L
        
        # Compute the augmented inertia term (on LHS of equation)
        adjusted_inertia = B_I - (I_Ws * np.outer(B_ghat_s, B_ghat_s)) - (Jg * np.outer(B_ghat_g, B_ghat_g))

        # Isolate omega_dot on the LHS of the equation
        omega_dot = np.dot(np.linalg.inv(adjusted_inertia), B_H_dot)

        return omega_dot
    
    def _compute_wheel_speed_dot(self, 
                                dcm_BG:np.array, 
                                B_omega_BN:np.array,
                                B_omega_BN_dot:np.array, 
                                gimbal_rate:float,
                                wheel_torque:float) -> float:
        """
        Compute the derivative of the wheel speed.
        
        Args:
            dcm_BG (ndarray): DCM of gimbal frame to body frame
            B_omega_BN_dot (ndarray): angular acceleration experessed in body frame
            wheel_torque (float): Torque applied to the wheel
            
        Returns:
            float: Time derivative of wheel speed
        """
        B_ghat_s = dcm_BG[:,0]
        ws, wt, wg = self._compute_angular_velocity_gimbal_frame_projection(B_omega_BN=B_omega_BN, dcm_BG=dcm_BG)
        
        wheel_speed_dot = (wheel_torque / self.I_Ws) - np.dot(B_ghat_s, B_omega_BN_dot) - (gimbal_rate * wt)
        
        return wheel_speed_dot
    
    def _compute_gimbal_rate_dot(self, dcm_BG, B_omega_BN_dot, B_omega_BN, wheel_speed, gimbal_torque):
        """
        Compute the derivative of the gimbal rate.
        
        Args:
            gimbal_torque (float): Torque applied to the gimbal
            
        Returns:
            float: Time derivative of gimbal rate
        """
        # Extract inertia properties
        I_Ws = self.I_Ws
        Js = self.G_J[0,0]
        Jt = self.G_J[1,1]
        Jg = self.G_J[2,2]

        # ghat basis vector of gimbal frame expressed in body frame coordinates
        B_ghat_g = dcm_BG[:,2]

        # Get angular velocity projections
        ws, wt, wg = self._compute_angular_velocity_gimbal_frame_projection(B_omega_BN, dcm_BG)

        gimbal_rate_dot = (1.0 / Jg * (gimbal_torque + (I_Ws * wheel_speed * wt) + ((Js - Jt) * ws * wt)) - 
                                        np.dot(B_ghat_g, B_omega_BN_dot))

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

    def simulate(self,
                init_state:StateVscmg, 
                t_init:float,
                t_max:float, 
                t_step:float=0.1,) -> dict:
        """
        4th order RK4 integrator 
        Note this integrator does slightly more than just integrate equations of motion, it also
        calculates the necessary control torque to apply to the equations of motion

        Args:
            init_state (StateVscmg): Initial state 
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
        state = init_state.to_array()
        t = t_init

        # TODO: replace with control function
        control_torque = np.zeros(3)
        wheel_torque = 0.0
        gimbal_torque = 0.0

        init_energy = self.CalculateTotalEnergy(state=init_state)
        init_H = self.CalculateTotalInertialAngularMomentum(state=init_state)
        
        # TODO: turn this into class
        # Initialize containers for storing data
        solution_dict = {}
        solution_dict["MRP"] = [init_state.sigma_BN.as_array()]   # sigma_BN
        solution_dict["omega_B_N"] = [init_state.B_omega_BN] # B_omega_BN
        solution_dict["wheel_speed"] = [init_state.wheel_speed]
        solution_dict["gimbal_angle"] = [init_state.gimbal_angle]
        solution_dict["gimbal_rate"] = [init_state.gimbal_rate]
        solution_dict["control"] = [control_torque]
        solution_dict["total_energy"] = [init_energy]
        solution_dict["N_H_total"] = [init_H]
        solution_dict["mode_value"] = [Mode.INVALID.value]
        solution_dict["time"] = [t]

        while t < t_max:

            # Make sure the input state is an array
            if isinstance(state, StateVscmg):
                state = state.to_array()

            # Calculate intermediate values
            k1 = t_step*self.TorqueFreeEquationsOfMotion(t, state, control_torque, wheel_torque, gimbal_torque)
            k2 = t_step*self.TorqueFreeEquationsOfMotion(t + t_step/2, state + k1/2, control_torque, wheel_torque, gimbal_torque)
            k3 = t_step*self.TorqueFreeEquationsOfMotion(t + t_step/2, state + k2/2, control_torque, wheel_torque, gimbal_torque)
            k4 = t_step*self.TorqueFreeEquationsOfMotion(t + t_step, state + k3, control_torque, wheel_torque, gimbal_torque)

            # DEBUGGING: print the intermediate derivatives on the first step
            if t == 0.0:
                print(f"initial state: \n{state}")
                print(f"k1: {StateVscmg.from_array(k1)}")
                print(f"k2: {StateVscmg.from_array(k2)}")
                print(f"k3: {StateVscmg.from_array(k3)}")
                print(f"k4: {StateVscmg.from_array(k4)}")

            # Update state
            state = state + 1.0/6.0*(k1 + 2*k2 + 2*k3 + k4)

            # Check MRP magnitude and covert to shadow set if necessary
            state = StateVscmg.from_array(state)
            
            if state.sigma_BN.norm() > 1.0:
                state.sigma_BN = state.sigma_BN.convert_to_shadow_set()

            # Increment the time
            t = t + t_step

            control_torque = np.zeros(3)
            wheel_torque = 0.0
            gimbal_torque = 0.0

            total_energy = self.CalculateTotalEnergy(state=state)

            N_H_total = self.CalculateTotalInertialAngularMomentum(state=state)

            # Save states and controls
            solution_dict["MRP"].append(state.sigma_BN.as_array())
            solution_dict["omega_B_N"].append(state.B_omega_BN)
            solution_dict["control"].append(control_torque)
            solution_dict["total_energy"].append(total_energy)
            solution_dict["N_H_total"].append(N_H_total)
            solution_dict["mode_value"].append(Mode.INVALID.value)
            solution_dict["wheel_speed"].append(state.wheel_speed)
            solution_dict["gimbal_angle"].append(state.gimbal_angle)
            solution_dict["gimbal_rate"].append(state.gimbal_rate)
            solution_dict["time"].append(t)


        # Convert lists to arrays so they're easier to work with later
        for key in solution_dict.keys():
            solution_dict[key] = np.array(solution_dict[key])

        return solution_dict