import numpy as np
import math

from State import SpacecraftState, VscmgState

class Vscmg:
    def __init__(self, 
                 G_J:np.array, 
                 I_Ws:np.array,
                 init_state:VscmgState, 
                 dcm_BG_init:np.array=None, 
                 gimbal_angle_init:float=None,
                 wheel_torque:float=0.0,
                 gimbal_torque:float=0.0):
        self.G_J = G_J
        self.I_Ws = I_Ws
        self.state = init_state

        # Initial gimbal to body DCM and gimbal angle:
        #   While [BG] and gamma vary with time, their initial values need to be stored to
        #   perform integration
        self.dcm_BG_init = dcm_BG_init
        self.gimbal_angle_init = gimbal_angle_init  # TODO: change so we get this from the initial state
        self.wheel_torque = wheel_torque
        self.gimbal_torque = gimbal_torque


    def _compute_gimbal_frame(self, vscmg_state:VscmgState) -> np.array:
                            #   gimbal_angle:float) -> np.array:
        """
        Compute the gimbal to body DCM [BG] for this VSCMG for a given VSCMG state
        NOTE: This is a utility function and does not provide the DCM for the current
        state of this VSCMG
        
        Args:
            vscmg_state (float): VSCMG state to use to calculate the corresponding DCM
            
        Returns:
            ndarray: 3x3 DCM describing transformation from gimbal to body frame 
        """
        # Get the current gimbal angle for this VSCMG
        gimbal_angle = vscmg_state.gimbal_angle

        # Get the initial gimbal angle and corresponding DCM for this VSCMG
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

        # Construct the [GB] DCM
        # Note that [BG] = {B_g1, B_g2, B_g3} (column vectors) so the transpose [GB]
        # can be created with row vectors
        dcm_GB = np.array([B_ghat_s,
                           B_ghat_t,
                           B_ghat_g])
        dcm_BG = np.transpose(dcm_GB)

        return dcm_BG


class Spacecraft:

    def __init__(self, B_Is:np.array, init_state:SpacecraftState, vscmgs:list[Vscmg]):
        self.B_Is = B_Is
        self.state = init_state
        self.vscmgs = vscmgs

    def update_state(self, state: SpacecraftState) -> None:
        """
        Updates the spacecraft's state and all attached VSCMG states.
        
        Arguments:
            state (SpacecraftState): The state to update the spacecraft with
        """

        if not isinstance(state, SpacecraftState):
            raise TypeError(f"Expected SpacecraftState, got {type(state)}")

        # Check if the number of VSCMG states matches the number of attached VSCMGs
        if len(state.vscmg_states) != len(self.vscmgs):
            raise ValueError(
                f"Mismatch in number of VSCMGs: {len(state.vscmg_states)} states provided, "
                f"but spacecraft has {len(self.vscmgs)} VSCMGs."
            )

        # Update spacecraft state
        self.state = state

        # Update each VSCMG's state
        for vscmg, new_vscmg_state in zip(self.vscmgs, state.vscmg_states):
            vscmg.state = new_vscmg_state


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
