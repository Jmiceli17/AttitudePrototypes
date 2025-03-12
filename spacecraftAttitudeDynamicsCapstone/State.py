import numpy as np

from ModifiedRodriguesParameters import MRP


class VscmgState:
    """
    Class representing the state of a single VSCMG (gimbal + wheel)

    Attributes:

    """
    def __init__(self, wheel_speed:float=0, gimbal_angle:float=0, gimbal_rate:float=0):
        """
        Args:
            wheel_speed (float, optional): Angular velocity of the VSCMG wheel [rad/s]
                                          Defaults to 0.0.
            gimbal_angle (float, optional): Orientation angle of the VSCMG gimbal [rad]
                                           Defaults to 0.0.
            gimbal_rate (float, optional): Rate of change of the gimbal angle [rad/s]
                                          Defaults to 0.0.
        """

        self.wheel_speed = wheel_speed
        self.gimbal_angle = gimbal_angle
        self.gimbal_rate = gimbal_rate

    def to_array(self):
        """
        Convert VSCMG state to array representation.
        
        Returns:
            ndarray: Array containing [wheel_speed, gimbal_angle, gimbal_rate]
        """
        return np.array([self.gimbal_angle, self.gimbal_rate, self.wheel_speed,])


    @classmethod
    def from_array(cls, array):
        """
        Create a VscmgState from an array.
        
        Args:
            array (ndarray): Array with [wheel_speed, gimbal_angle, gimbal_rate]
            
        Returns:
            VscmgState: New VSCMG state instance
        """
        return cls(
            gimbal_angle=array[0], 
            gimbal_rate=array[1],
            wheel_speed=array[2], 
        )
    
    def __str__(self):
        """String representation of the VSCMG state."""
        return (f"VscmgState:\n"
                f"  wheel_speed: {self.wheel_speed} rad/s\n"
                f"  gimbal_angle: {self.gimbal_angle} rad\n"
                f"  gimbal_rate: {self.gimbal_rate} rad/s\n")


class SpacecraftState:
    """
    Class representing the state of a spacecraft with a Variable-Speed Control Moment Gyroscope (VSCMG).
    
    Attributes:
        sigma_BN (list/ndarray): Modified Rodrigues Parameters (MRPs) representing spacecraft attitude
        B_omega_BN (list/ndarray): Angular velocity of the spacecraft wrt the inertial frame expressed in body frame
        wheel_speed (float): Angular velocity of the VSCMG wheel
        gimbal_angle (float): Orientation angle of the VSCMG gimbal
        gimbal_rate (float): Rate of change of the gimbal angle
    """
    
    def __init__(self, sigma_BN=None, B_omega_BN=None, vscmg_states:np.array=[]):
        """
        Initialize the spacecraft state with N VSCMGs.
        
        Args:
            sigma_BN (list/ndarray/MRP, optional): Modified Rodrigues Parameters (MRPs) for attitude (body wrt inertial)
                                           Defaults to [0, 0, 0].
            B_omega_BN (list/ndarray, optional): Angular velocity wrt the inertial frame expressed in body frame [rad/s]
                                           Defaults to [0, 0, 0].
            vscmg_states (list/ndarray): List of VSCMG states attached to this spacecraft
        """
        if isinstance(sigma_BN, MRP):
            self.sigma_BN = sigma_BN
        elif sigma_BN is None:
            self.sigma_BN = MRP(0,0,0)
        else:
            self.sigma_BN = MRP.from_array(sigma_BN)

        self.B_omega_BN = np.array([0,0,0]) if B_omega_BN is None else B_omega_BN
        self.vscmg_states = vscmg_states
    
    def __str__(self):
        """String representation of the spacecraft state."""
        return (f"SpacecraftState:\n"
                f"  sigma_BN: {self.sigma_BN.as_array()}\n"
                f"  B_omega_BN: {self.B_omega_BN}\n"
                f"  VSCMG States:\n"
                f"    wheel_speeds: {[v.wheel_speed for v in self.vscmg_states]}\n"
                f"    gimbal_angles: {[v.gimbal_angle for v in self.vscmg_states]}\n"
                f"    gimbal_rates: {[v.gimbal_rate for v in self.vscmg_states]}\n")

    def add_vscmg_state(self, vscmg:VscmgState):
        """
        Add a VSCMG state to the spacecraft state
        
        Args:
            vscmg (VscmgState): VSCMG state to add
        """
        self.vscmg_states.append(vscmg)

    def to_array(self):
        """
        Convert the state to a flat array representation.
        
        Returns:
            list: Flattened state representation
        """
        # Start with attitude and angular velocity
        state_array = np.concatenate((self.sigma_BN.as_array(), self.B_omega_BN))
        
        # Ensure VSCMG states are added correctly
        if self.vscmg_states:
            vscmg_arrays = np.concatenate([vscmg.to_array() for vscmg in self.vscmg_states])
            state_array = np.concatenate((state_array, vscmg_arrays))
            
        return state_array
    
    @classmethod
    def from_array(cls, array):
        """
        Create a SpacecraftState instance from a flat array.
        
        Args:
            array (ndarray): Flattened state representation

        Returns:
            SpacecraftState: Instance created from the array
        """
        # Extract attitude and angular velocity
        sigma_BN = array[0:3]
        B_omega_BN = array[3:6]
        
        # Create spacecraft state without VSCMGs initially (empty list ensures that no history is carried
        # over from a previous object)
        # sc_state = cls(sigma_BN=sigma_BN, B_omega_BN=B_omega_BN, vscmg_states=[])
        
        # Calculate number of VSCMGs based on array length
        n_vscmgs = (len(array) - 6) // 3  # Each VSCMG has 3 state variables

        vscmg_states = []

        # Add each VSCMG
        for i in range(n_vscmgs):
            idx = 6 + i*3  # Starting index for this VSCMG in the array
            vscmg_array = array[idx:idx+3]
            # print(f"vscmg_array: {vscmg_array}")
            # Create VSCMG state
            vscmg = VscmgState.from_array(
                vscmg_array
            )

            vscmg_states.append(vscmg)
            # # Add to spacecraft
            # sc_state.add_vscmg_state(vscmg)

        sc_state = cls(sigma_BN=sigma_BN, B_omega_BN=B_omega_BN, vscmg_states=vscmg_states)
        return sc_state