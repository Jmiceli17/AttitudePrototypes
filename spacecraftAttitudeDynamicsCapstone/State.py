import numpy as np

from ModifiedRodriguesParameters import MRP


class StateVscmg:
    """
    Class representing the state of a spacecraft with a Variable-Speed Control Moment Gyroscope (VSCMG).
    
    Attributes:
        sigma_BN (list/ndarray): Modified Rodrigues Parameters (MRPs) representing spacecraft attitude
        B_omega_BN (list/ndarray): Angular velocity of the spacecraft wrt the inertial frame expressed in body frame
        wheel_speed (float): Angular velocity of the VSCMG wheel
        gimbal_angle (float): Orientation angle of the VSCMG gimbal
        gimbal_rate (float): Rate of change of the gimbal angle
    """
    
    def __init__(self, sigma_BN=None, B_omega_BN=None, wheel_speed=0.0, gimbal_angle=0.0, gimbal_rate=0.0):
        """
        Initialize the spacecraft state with a VSCMG.
        
        Args:
            sigma_BN (list/ndarray/MRP, optional): Modified Rodrigues Parameters (MRPs) for attitude (body wrt inertial)
                                           Defaults to [0, 0, 0].
            B_omega_BN (list/ndarray, optional): Angular velocity wrt the inertial frame expressed in body frame [rad/s]
                                           Defaults to [0, 0, 0].
            wheel_speed (float, optional): Angular velocity of the VSCMG wheel [rad/s]
                                          Defaults to 0.0.
            gimbal_angle (float, optional): Orientation angle of the VSCMG gimbal [rad]
                                           Defaults to 0.0.
            gimbal_rate (float, optional): Rate of change of the gimbal angle [rad/s]
                                          Defaults to 0.0.
        """
        if isinstance(sigma_BN, MRP):
            self.sigma_BN = sigma_BN
        elif sigma_BN is None:
            self.sigma_BN = MRP(0,0,0)
        else:
            self.sigma_BN = MRP.from_array(sigma_BN)

        self.B_omega_BN = np.array([0,0,0]) if B_omega_BN is None else B_omega_BN
        self.wheel_speed = wheel_speed
        self.gimbal_angle = gimbal_angle
        self.gimbal_rate = gimbal_rate
    
    def __str__(self):
        """String representation of the spacecraft state."""
        return (f"StateVscmg:\n"
                f"  sigma_BN: {self.sigma_BN.as_array()}\n"
                f"  B_omega_BN: {self.B_omega_BN}\n"
                f"  wheel_speed: {self.wheel_speed}\n"
                f"  gimbal_angle: {self.gimbal_angle}\n"
                f"  gimbal_rate: {self.gimbal_rate}")
    
    def to_array(self):
        """
        Convert the state to a flat array representation.
        
        Returns:
            list: Flattened state representation
        """
        return np.concatenate((self.sigma_BN.as_array(),
                              self.B_omega_BN, 
                              [self.wheel_speed], 
                              [self.gimbal_angle], 
                              [self.gimbal_rate]))
    
    @classmethod
    def from_array(cls, array):
        """
        Create a StateVscmg instance from a flat array.
        
        Args:
            array (list/ndarray): Flattened state representation
            
        Returns:
            StateVscmg: Instance created from the array
        """
        return cls(
            sigma_BN=array[0:3],
            B_omega_BN=array[3:6],
            wheel_speed=array[6],
            gimbal_angle=array[7],
            gimbal_rate=array[8]
        )