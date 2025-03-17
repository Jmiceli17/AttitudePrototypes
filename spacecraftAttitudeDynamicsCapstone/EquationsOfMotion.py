from State import SpacecraftState, VscmgState
import RigidBodyKinematics as RBK 
from PointingControl import Mode
from ModifiedRodriguesParameters import MRP
from Spacecraft import Spacecraft, Vscmg

import numpy as np

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

    def CalculateTotalPower(self,
                            spacecraft:Spacecraft,
                            external_torque:np.array) -> float:
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


    def CalculateTotalEnergy(self,
                             spacecraft:Spacecraft) -> float:
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
        """
        Compute the time derivatives for the spacecraft state in the absence of control torques, wheel motor torque, 
        and gimbal motor torque
        
        NOTE: None of the subfunctions in here should utilize self.spacecraft.state because that is not updated
        until after all intermediate RK4 calls to this function

        Args:
            t (float): Current time (for time-dependent systems)
            state_array (ndarray): State array of the spacecraft assumed form 
                [sigma_BN, omega_BN, gimbal_angles, gimbal_rates, wheel_speeds]
            external_torque (ndarray, optional): External torque. Defaults to 0

        Returns:
            np.array: Time derivative of the state
        """

        # Convert array to state object
        state = SpacecraftState.from_array(state_array)

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
                                                external_torque=external_torque, )

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

        Mmat = self._compute_M_matrix(B_I=B_I, state=state)

        state_diff = np.concatenate([
            sigma_BN_dot,
            B_omega_BN_dot,
            gimbal_angle_dot,
            gimbal_rate_dot,
            wheel_speed_dot
        ])

        # Calculate state_dot using 
        # [M] * state_dot = f(state)
        state_dot = np.linalg.solve(Mmat, state_diff)
        return state_dot


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
        dcm_BG = vscmg._compute_gimbal_frame(vscmg_state=vscmg_state)

        ws, wt, wg = self._compute_angular_velocity_gimbal_frame_projection(B_omega_BN=B_omega_BN, dcm_BG=dcm_BG)

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
        dcm_BG = vscmg._compute_gimbal_frame(vscmg_state=vscmg_state)

        # Get angular velocity projections
        ws, wt, wg = self._compute_angular_velocity_gimbal_frame_projection(B_omega_BN, dcm_BG)

        # Compute the right hand side of the gimbal rate equations of motion (see ex 4.15 Schaub, Junkins)
        f_gimbal_rate = gimbal_torque + ((Js - Jt) * ws * wt) + (I_Ws * wheel_speed * wt)
        return f_gimbal_rate
    
    def _compute_angular_velocity_gimbal_frame_projection(self, 
                                                          B_omega_BN:np.array, 
                                                          dcm_BG:np.array) -> tuple[float, float, float]:
        """
        Compute the gimbal frame angular velocity projection components ws, wt, wg
        
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


    def _compute_gimbal_frame_matrices(self, state:SpacecraftState) -> tuple[np.array, np.array, np.array]:
        """
        Compute the gimbal frame matrices based on eq 4.140 in Schaub, Junkins

        Args:
            state (SpacecraftState): Complete state of the spacecraft + vscmg system

        Returns:
            tuple: Three 3xN matrices [G_s], [G_t], [G_g]
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

        # Scale each G matrix by J
        # TODO: This assumes that all VSCMGs have the same inertia properties
        ex_vscmg = self.spacecraft.vscmgs[0]
        
        Js = ex_vscmg.G_J[0,0]
        Jt = ex_vscmg.G_J[1,1]
        Jg = ex_vscmg.G_J[2,2]
        I_Ws = ex_vscmg.I_Ws

        M = np.block([
            [i3, zero3, zero3N, zero3N, zero3N],
            [zero3, B_I, zero3N, (Jg * GMat_g), (I_Ws * GMat_s)],
            [zeroN3, zeroN3, iN, zeroN, zeroN],
            [zeroN3, (Jg * GMat_g.T), zeroN, (Jg * iN), zeroN],
            [zeroN3, (I_Ws * GMat_s.T), zeroN, zeroN, (I_Ws * iN)],
        ])

        # Check if M has the correct dimensions
        expected_shape = (6 + 3 * num_vscmgs, 6 + 3 * num_vscmgs)
        if M.shape != expected_shape:
            raise ValueError(f"Matrix M has shape {M.shape}, expected {expected_shape}")

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
                                              # state=state,  # TODO: change to spacecraft?
                                              external_torque=external_torque)
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
        solution_dict["total_power"] = [init_power]
        solution_dict["N_H_total"] = [init_H]
        solution_dict["mode_value"] = [Mode.INVALID.value]
        solution_dict["time"] = [t]

        while t < t_max:

            # Make sure the input state is an array
            if isinstance(state, SpacecraftState):
                state = state.to_array()

            # Calculate intermediate values
            k1 = t_step*self.TorqueFreeEquationsOfMotion(t, state, external_torque)
            k2 = t_step*self.TorqueFreeEquationsOfMotion(t + t_step/2, state + k1/2, external_torque)
            k3 = t_step*self.TorqueFreeEquationsOfMotion(t + t_step/2, state + k2/2, external_torque)
            k4 = t_step*self.TorqueFreeEquationsOfMotion(t + t_step, state + k3, external_torque)

            # DEBUGGING: print the intermediate derivatives on the first step
            if t == 0.0:
                print(f"state at [t={t}]: \n{state}")
                print(f"k1 array: {k1}\n as state: {SpacecraftState.from_array(k1)}")
                print(f"k2 array: {k2}\n as state: {SpacecraftState.from_array(k2)}")
                print(f"k3 array: {k3}\n as state: {SpacecraftState.from_array(k3)}")
                print(f"k4 array: {k4}\n as state: {SpacecraftState.from_array(k4)}")

            # Update state array for next step
            state = state + 1.0/6.0*(k1 + 2*k2 + 2*k3 + k4)

            # Check MRP magnitude and covert to shadow set if necessary
            state = SpacecraftState.from_array(state)

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

            for idx in range(len(self.spacecraft.state.vscmg_states)):
                vscmg_state = self.spacecraft.state.vscmg_states[idx]
                # TODO: replace this loop with a loop over the Vscmg objects attached to self.spacecraft
                solution_dict[f"wheel_speed_{idx}"].append(vscmg_state.wheel_speed)
                solution_dict[f"gimbal_angle_{idx}"].append(vscmg_state.gimbal_angle)
                solution_dict[f"gimbal_rate_{idx}"].append(vscmg_state.gimbal_rate)

            solution_dict["time"].append(t)


        # Convert lists to arrays so they're easier to work with later
        for key in solution_dict.keys():
            solution_dict[key] = np.array(solution_dict[key])

        return solution_dict