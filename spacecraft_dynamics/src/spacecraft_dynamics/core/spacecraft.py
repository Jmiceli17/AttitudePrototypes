import numpy as np

from ..state import SpacecraftState
from ..actuators import (
    WheelBase, 
    ReactionWheel, 
    Vscmg
)

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


class Spacecraft:
    def __init__(self, 
                 B_Is: np.array, 
                 init_state: SpacecraftState, 
                 actuators: list[WheelBase],
                 dummy_control: bool=False,
                 control_gains: ControlGains=None,
                 use_vscmg_null_motion: bool=False,
                 condition_number_cutoff: float=3.0,
                 null_motion_gradient_step: float=10.0,
                 null_motion_correction_gain: float = 0.1):
        """
        Spacecraft class

        Args:
            B_Is (ndarray): The inertia of the spacecraft hub (not including actuators) expressed in body frame [kgm^2]
            init_state (SpacecraftState): The initial state of the spacecraft
            actuators (list): List of all actuators attached to the spacececraft (currently assumes all actuators are of the same type)
            dummy_control (bool): Flag indicating if dummy actuator control should be used, when true actuator desired staets will be 
                calculated as a function of time rather than derived from requested control torque, currently only works with VSCMG actuators
            control_gains (ControlGains): The control gains to use when determining the required torque for attitude feedback control
            use_vscmg_null_motion (bool): When "true", the VSCMG null space will be used to actively re-configure the commanded
                VSCMG states to maximize the condition number of the VSCMG [D] matrix
            condition_number_cutoff (float): Deadband to apply to the condition number, must be > 1, if condition number is less 
                than this, the VSCMG orientation is considered "good enough" and no control effort will be applied
            null_motion_gradient_step (float): The step size, alpha, to use for gradient ascent of condition number when applying null
                motion control update
            null_motion_correction_gain (float): The null motion correction gain, ke
        """
        # Check that all actuators are of the same type
        if actuators:
            actuator_type = type(actuators[0])
            if not all(isinstance(actuator, actuator_type) for actuator in actuators):
                raise TypeError("All actuators must be of the same type")

        self.B_Is = B_Is
        self.state = init_state
        self.actuators = actuators
        self.num_actuators = len(actuators)
        self.control_gains = control_gains
        self.total_inertia = self.B_Is  # TODO: should this be initialized with actuator inertias?

        # TODO: move these to VSCMGs?
        self.previous_wheel_speed_dot_desired_vec = np.zeros((self.num_actuators,))
        self.previous_gimbal_rate_desired_vec = np.zeros((self.num_actuators,))
        self.wheel_speed_dot_desired_vec = np.zeros((self.num_actuators,))
        self.gimbal_rate_desired_vec = np.zeros((self.num_actuators,))

        # The 3xN block matrices used to simplify the VSCMG constraint condition
        self.D0 = np.zeros((3,self.num_actuators))
        self.D1 = np.zeros((3,self.num_actuators))
        self.D2 = np.zeros((3,self.num_actuators))
        self.D3 = np.zeros((3,self.num_actuators))
        self.D4 = np.zeros((3,self.num_actuators))
        self.D = np.zeros((3,self.num_actuators))

        # The 3x2N matrix used to simplfiy the VSCMG accel based steering law constraint condition
        self.Q = np.zeros((3, 2 * self.num_actuators))

        # VSCMG null motion flag
        self.use_vscmg_null_motion = use_vscmg_null_motion

        if self.use_vscmg_null_motion:
            # Store the latest SVD results for computing null-space desired states
            self.D_singular_values = None
            self.D_condition_number = None
            self.D_U = None  # Left singular vectors
            self.D_Vt = None  # Right singular vectors (transposed)
            if (condition_number_cutoff <= 1.0):
                raise ValueError("condition_number_cutoff must be greater than 1")
            self.D_condition_number_cutoff = condition_number_cutoff
            self.null_motion_gradient_step = null_motion_gradient_step
            self.null_motion_correction_gain = null_motion_correction_gain

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


    def update_control_torque(self, B_L_R:np.array, B_omega_BR:np.array=None, dt:float=0.01) -> None:
        """
        Maps a desired control torque vector to actuator torque commands
         
        When using reaction wheels, this mapping is performed using minimum norm pseudoinverse solution
        Solves the equation [Gs]us = -Lr where:
        - [Gs] is the matrix mapping wheel torques to body torques
        - us is the vector of wheel torque commands
        - Lr is the desired control torque vector

        When using VSCMGs the mapping is done by first determining the desired VSCMG states (optionally including
        VSCMG null motion) and then calculating the wheel and gimbal torques required to achieve those states

        Args:
            B_L_R (ndarray): 3x1 Desired control torque in body frame [Nm]
            B_omega_BR (ndarray): 3x1 vector descrbing ang vel of body wrt. desired reference frame, expressed 
                in body frame components [rad/s]
            dt (float): The size of the time step between dynamics updates [s] 
                NOTE: This may not be the same as the amount of time that elapses between calls to this function
        """
        # Get actuator type (we already ensure all actuators are the same type in __init__)
        if not self.actuators:
            return
            
        actuator_type = type(self.actuators[0])
        
        if actuator_type == ReactionWheel:
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

        elif actuator_type == Vscmg:

            # Update the desired states for the VSCMG
            self.set_desired_vscmg_states(B_L_R=B_L_R, B_omega_BR=B_omega_BR)

            for i, actuator_state_tuple in enumerate(zip(self.actuators, self.state.actuator_states)):
                vscmg, vscmg_state = actuator_state_tuple

                # Extract state info for the spacecraft
                B_omega_BN = self.state.B_omega_BN

                # Extract properties and state info for this VSCMG
                I_Ws = vscmg.I_Ws
                Js = vscmg.G_J[0,0]
                Jt = vscmg.G_J[1,1]
                Jg = vscmg.G_J[2,2]
                gimbal_rate = vscmg_state.gimbal_rate
                wheel_speed = vscmg_state.wheel_speed

                dcm_BG = vscmg._compute_gimbal_frame(vscmg_state)

                # Get angular velocity projections
                ws, wt, wg = vscmg._compute_angular_velocity_gimbal_frame_projection(B_omega_BN, dcm_BG)

                # Estimate gimbal accel with simple finite difference
                gimbal_rate_desired = self.gimbal_rate_desired_vec[i]
                previous_gimbal_rate_desired = self.previous_gimbal_rate_desired_vec[i]
                gimbal_rate_dot_desired = (gimbal_rate_desired - previous_gimbal_rate_desired) / dt

                # Gimbal rate tracking error
                delta_gamma_dot = gimbal_rate - gimbal_rate_desired

                vscmg.gimbal_torque = float(Jg * (gimbal_rate_dot_desired - (vscmg.K_gimbal * delta_gamma_dot)) \
                                        - ((Js - Jt) * ws * wt) - (I_Ws * wheel_speed * wt))

                wheel_speed_dot_desired = self.wheel_speed_dot_desired_vec[i]
                vscmg.wheel_torque = float(I_Ws * (wheel_speed_dot_desired + gimbal_rate * wt))

        else:
            raise TypeError(f"Unsupported actuator type: {actuator_type}")

    def set_desired_vscmg_states(self, B_L_R:np.array, B_omega_BR:np.array) -> None:
        """
        Update the VSCMG desired states using the latest commanded torque vector
        See eq 8.232 in Schaub, Junkins

        Args: 
            B_L_R (ndarray): Desired control torque expressed in body frame [Nm]
            B_omega_BR (ndarary): Angular velocity of the spacecraft body wrt the reference angular velocity (ang vel error)
                expressed in body frame components [rad/s]
        """
        # Update the block matrices used to simplify the stability constraint
        self.update_stability_constraint_matrices(B_omega_BR=B_omega_BR)
        Q = self.Q

        # Update the previous values
        import copy 
        self.previous_wheel_speed_dot_desired_vec = copy.deepcopy(self.wheel_speed_dot_desired_vec)
        self.previous_gimbal_rate_desired_vec = copy.deepcopy(self.gimbal_rate_desired_vec)

        W = self.get_vscmg_weight_matrix()

        # Solve weighted inverse equation 
        # NOTE: L_R is negative here because it is the torque acting on the VSCMG (the negative of which
        # will be applied to the spacecraft)
        inverse_mat = np.linalg.inv(np.matmul(Q, np.matmul(W, Q.T)))
        eta_dot = np.matmul(np.matmul(W, np.matmul(Q.T, (inverse_mat))), -B_L_R).reshape((2 * self.num_actuators,1))

        self.wheel_speed_dot_desired_vec = eta_dot[:self.num_actuators]
        self.gimbal_rate_desired_vec = eta_dot[self.num_actuators:]

        # If using VSCMG null space, we can calculate a new "desired" VSCMG state that maximizes the condition
        # number of the [D] matrix
        if self.use_vscmg_null_motion:
            
            # # Apply deadband to condition number so that we don't exert a ton of control effort when the 
            # # conditionality of the [D] matrix is good enough
            # if (self.D_condition_number <= self.D_condition_number_cutoff):
            #     gimbal_angle_null_motion_error = np.zeros((self.num_actuators,1))

            # else:
            # #     # gimbal_angle_null_motion_error = self.compute_delta_gamma()
            desired_gimbal_angles = np.deg2rad([-45, 45, -45, 45])  # TODO: Make hardsetting desired gimbal angles an option
            gimbal_angle_null_motion_error = []
            for vscmg_state, desired_gimbal_angle in zip(self.state.actuator_states,desired_gimbal_angles):
                gimbal_angle_null_motion_error.append(vscmg_state.gimbal_angle - desired_gimbal_angle)

            gimbal_angle_null_motion_error = np.array(gimbal_angle_null_motion_error)

            vscmg_state_error = np.vstack([
                np.zeros((self.num_actuators, 1)),
                gimbal_angle_null_motion_error.reshape(-1, 1)
            ])

            # Define the "[A]" matrix used to specify how much we care about acheiving the null motion states, in this 
            # case we only care about the gimbal angle states (the bottom diagonal)
            vscmg_null_motion_encoding = np.block([
                [np.zeros((self.num_actuators, self.num_actuators)), np.zeros((self.num_actuators, self.num_actuators))],
                [np.zeros((self.num_actuators, self.num_actuators)), np.eye(self.num_actuators)]
            ])
            W = self.get_vscmg_weight_matrix()
            
            # eta_null_motion = self.null_motion_correction_gain * np.matmul(np.matmul((np.matmul(np.matmul(W, 
            #                                 np.matmul(self.Q.T, 
            #                                     np.linalg.inv(np.matmul(self.Q, np.matmul(W, self.Q.T))))), 
            #                                         self.Q)) - np.eye(2 * self.num_actuators, 2 * self.num_actuators),
            #                                             vscmg_null_motion_encoding), 
            #                                             vscmg_state_error)
            
            # Step 1: Calculate Q*W*Q^T and its inverse
            QWQt = np.matmul(self.Q, np.matmul(W, self.Q.T))
            QWQt_inv = np.linalg.inv(QWQt)
            
            # Step 2: Calculate W*Q^T*(QWQ^T)^-1*Q
            temp1 = np.matmul(W, np.matmul(self.Q.T, QWQt_inv))
            temp2 = np.matmul(temp1, self.Q)
            
            # Step 3: Subtract identity matrix
            I_mat = np.eye(2 * self.num_actuators, 2 * self.num_actuators)
            temp3 = temp2 - I_mat
            
            # Step 4: Multiply by encoding matrix
            temp4 = np.matmul(temp3, vscmg_null_motion_encoding)
            
            # Step 5: Final multiplication with state error
            eta_null_motion = self.null_motion_correction_gain * np.matmul(temp4, vscmg_state_error)

            # Extract the wheel speed and gimbal portions from the desired state 
            # NOTE: this implementation means that order matters here
            wheel_speed_null_desired_null_motion = eta_null_motion[:self.num_actuators]
            gimbal_rate_desired_null_motion = eta_null_motion[self.num_actuators:]

            self.gimbal_rate_desired_vec += gimbal_rate_desired_null_motion
            self.wheel_speed_dot_desired_vec += wheel_speed_null_desired_null_motion



    def get_vscmg_weight_matrix(self) -> np.array:
        """
        Calculate the diagonal weight matrix for VSCMG acceleration based servo steering law
        NOTE: This function assumes that the stability constraint matrices have already been 
        updated using update_stability_constraint_matrices

        Returns:
            A 2N x 2N numpy array [W] = diag{ws0, ws1, ... wg0, wg1}
        """
        mu = 1e-9 # TODO: make configurable?
        weights = []
        D1_matrix = self.D1
        for vscmg, vscmg_state in zip(self.actuators, self.state.actuator_states):
            Js = vscmg.G_J[0,0]
            h = vscmg_state.initial_wheel_speed * Js

            delta = np.linalg.det(1 / (h**2) * np.matmul(D1_matrix, D1_matrix.T))

            wheel_speed_weight = 200.0 * np.exp(-mu * delta)
            weights.append(wheel_speed_weight)

        # Now add (constant) gimbal weights
        for vscmg in self.actuators:
            weights.append(1.0)

        weight_matrix = np.diag(weights)
        return weight_matrix

    def update_stability_constraint_matrices(self, B_omega_BR:np.array) -> None:
        """
        Update stability constraint matrices [D0], [D1], [D2], [D3], [D4], [D], [Q] and
        compute the SVD of [D1] for null motion control, these matrices are necessary to compute the desired
        states for the VSCMG servo

        Args:
            B_omega_BR (ndarray): Angular velocity of the spacecraft body wrt the reference angular velocity (ang vel error)
                expressed in body frame components [rad/s]
        """
        # Calulate the reference angular velocity from delta_omega = omega_BN - omega_RN
        B_omega_BN = self.state.B_omega_BN

        B_omega_RN = B_omega_BN - B_omega_BR

        for idx, vscmg_state_tuple in enumerate(zip(self.actuators, self.state.actuator_states)):
            vscmg, vscmg_state = vscmg_state_tuple

            # Extract properties and state info for this VSCMG
            I_Ws = vscmg.I_Ws
            Js = vscmg.G_J[0,0]
            Jt = vscmg.G_J[1,1]
            Jg = vscmg.G_J[2,2]
            wheel_speed = vscmg_state.wheel_speed
            dcm_BG = vscmg._compute_gimbal_frame(vscmg_state)
            B_ghat_s = dcm_BG[:,0]
            B_ghat_t = dcm_BG[:,1]
            B_ghat_g = dcm_BG[:,2]
            ws, wt, wg = vscmg._compute_angular_velocity_gimbal_frame_projection(B_omega_BN, dcm_BG)

            # Update matrices (see Schaub, Junkins eq 8.217)
            self.D0[:,idx] = B_ghat_s * I_Ws
            self.D1[:,idx] = B_ghat_t * (I_Ws * wheel_speed + (Js / 2.0) * ws) + B_ghat_s * (Js / 2.0) * wt
            self.D2[:,idx] = 0.5 * Jt * (B_ghat_s * wt + B_ghat_t * ws)
            self.D3[:,idx] = Jg * (B_ghat_s * wt - B_ghat_t * ws)
            self.D4[:,idx] = 0.5 * (Js - Jt) * (np.matmul(np.outer(B_ghat_s, B_ghat_t), B_omega_RN) + np.matmul(np.outer(B_ghat_t, B_ghat_s), B_omega_RN))

            self.D = self.D1 - self.D2 + self.D3 + self.D4

            self.Q = np.concatenate((self.D0, self.D),axis=1)

            # Calculate and store SVD and condition number of the D matrix
            U, S, Vt = np.linalg.svd(self.D1)
            self.D_U = U
            self.D_singular_values = S
            self.D_Vt = Vt
            if (S[-1] <= 1e-8):
                self.D_condition_number = np.inf
            else:
                self.D_condition_number = S[0] / S[-1]

    def calculate_dummy_vscmg_desired_states(self, t:float) -> tuple[np.array, np.array]:
        """
        Function for generating desired VSCMG actuutor states as a function of time,
        NOTE: This function is not currently used but can be used for testing

        Args:
            t (float): Simulation time [s]
        """
        wheel_speed_dot_desired_vec = np.deg2rad(np.array([np.sin((0.02 * t)),
                                                np.cos((0.02 * t)),
                                                -np.cos((0.03 * t)),
                                                -np.cos((0.03 * t))])) # [rad/s^2]

        gimbal_rate_desired_vec = np.deg2rad(np.array([np.sin((0.02 * t)),
                                                np.cos((0.02 * t)),
                                                -np.cos((0.03 * t)),
                                                -np.cos((0.03 * t))])) # [rad/s]

        return wheel_speed_dot_desired_vec, gimbal_rate_desired_vec


    def compute_delta_gamma(self) -> np.array:
        """
        Compute the gimbal angle tracking error that would minimize the condition number of VSCMG configruation

        Returns:
            Nx1 numpy array containing the gimbal angle tracking error for each VSCMG
        """
        
        alpha = self.null_motion_gradient_step
        kappa = self.D_condition_number

        # Chi_i vectors (one for each component of gamma)
        # If D is 3x3, then vec(D) has length 9, and u_j has length 3
        # This is just an example; replace with your actual Chi_i vectors
        num_gamma_components = self.num_actuators
        if (num_gamma_components != 4):
            raise ValueError(f"Spacecraft has {num_gamma_components} configured, Current null motion control assumes 4 VSCMGs")

        # Step 1: Extract current SVD of D to get u_j, sigma_j, and V
        U = self.D_U
        V = self.D_Vt.T  # Transpose Vt to get V

        # Extract sigma_1 and sigma_3 (first and third singular values)
        sigma_1 = self.D_singular_values[0]
        sigma_3 = self.D_singular_values[-1]

        # Extract u_1 and u_3 (first and third left singular vectors)
        u_1 = U[:, 0]
        u_3 = U[:, 2]

        # Step 2: Compute partial derivatives of sigma_1 and sigma_3 w.r.t. gamma_i
        # We'll store these as arrays for each component of gamma
        partial_sigma1_gamma = np.zeros(num_gamma_components)
        partial_sigma3_gamma = np.zeros(num_gamma_components)

        for i in range(num_gamma_components):

            # Compute the Chi[i] vector
            B_omega_BN = self.state.B_omega_BN
            vscmg = self.actuators[i]
            vscmg_state = self.state.actuator_states[i]
            I_Ws = vscmg.I_Ws
            Js = vscmg.G_J[0,0]

            wheel_speed = vscmg_state.wheel_speed

            dcm_BG = vscmg._compute_gimbal_frame(vscmg_state)
            B_ghat_s = dcm_BG[:,0]
            B_ghat_t = dcm_BG[:,1]

            ws, wt, wg = vscmg._compute_angular_velocity_gimbal_frame_projection(B_omega_BN, dcm_BG)

            Chi_i = -B_ghat_s * (I_Ws * wheel_speed + Js * ws) + B_ghat_t * Js * wt

            # Compute partial sigma_1 / partial gamma_i
            # partial sigma_j / partial gamma_i = (u_j^T Chi_i) V_ij
            partial_sigma1_gamma[i] = np.dot(u_1.T, Chi_i) * V[i, 0]  # j=1 (sigma_1)
            
            # Compute partial sigma_3 / partial gamma_i
            partial_sigma3_gamma[i] = np.dot(u_3.T, Chi_i) * V[i, 2]  # j=3 (sigma_3)

        # Step 3: Compute partial kappa / partial gamma_i
        partial_kappa_gamma = np.zeros(num_gamma_components)
        for i in range(num_gamma_components):
            term1 = (1 / sigma_3) * partial_sigma1_gamma[i]
            term2 = (sigma_1 / (sigma_3 ** 2)) * partial_sigma3_gamma[i]
            partial_kappa_gamma[i] = term1 - term2

        # Step 4: Compute Delta gamma
        # Delta gamma_i = -alpha (1 - kappa(t)) (partial kappa / partial gamma_i)
        delta_gamma = -alpha * (1 - kappa) * partial_kappa_gamma

        return delta_gamma  
