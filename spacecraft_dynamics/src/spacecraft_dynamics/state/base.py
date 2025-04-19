import numpy as np
from abc import ABC, abstractmethod

from ..utils import MRP

class ActuatorState(ABC):
    """Base class for actuator states (VSCMG or Reaction Wheel)"""
    @abstractmethod
    def to_array(self):
        pass

    @classmethod
    @abstractmethod
    def from_array(cls, array):
        pass

class VscmgState(ActuatorState):
    """Class representing the state of a single VSCMG (gimbal + wheel)"""
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
        self.initial_wheel_speed = wheel_speed
        self.wheel_speed = wheel_speed
        self.gimbal_angle = gimbal_angle
        self.gimbal_rate = gimbal_rate

    def to_array(self):
        """
        Convert VSCMG state to array representation.
        
        Returns:
            ndarray: Array containing [wheel_speed, gimbal_angle, gimbal_rate]
        """
        return np.array([self.gimbal_angle, self.gimbal_rate, self.wheel_speed])

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

class ReactionWheelState(ActuatorState):
    """Class representing the state of a single reaction wheel"""
    def __init__(self, wheel_speed:float=0):
        self.wheel_speed = wheel_speed

    def to_array(self):
        return np.array([self.wheel_speed])

    @classmethod
    def from_array(cls, array):
        return cls(wheel_speed=array[0])
    
    def __str__(self):
        return (f"ReactionWheelState:\n"
                f"  wheel_speed: {self.wheel_speed} rad/s\n")

class SpacecraftState:
    """
    Class representing the state of a spacecraft with a Variable-Speed Control Moment Gyroscope (VSCMG).
    
    Attributes:
        sigma_BN (list/ndarray): Modified Rodrigues Parameters (MRPs) representing spacecraft attitude
        B_omega_BN (list/ndarray): Angular velocity of the spacecraft wrt the inertial frame expressed in body frame
        actuator_states (list): List of actuator states (VSCMG or Reaction Wheel)
    """
    
    def __init__(self, sigma_BN=None, B_omega_BN=None, actuator_states:list[ActuatorState]=[]):
        """
        Initialize the spacecraft state with N actuators.
        
        Args:
            sigma_BN (list/ndarray/MRP, optional): Modified Rodrigues Parameters (MRPs) for attitude (body wrt inertial)
                                           Defaults to [0, 0, 0].
            B_omega_BN (list/ndarray, optional): Angular velocity wrt the inertial frame expressed in body frame [rad/s]
                                           Defaults to [0, 0, 0].
            actuator_states (list): List of actuator states (VSCMG or Reaction Wheel)
        """
        if isinstance(sigma_BN, MRP):
            self.sigma_BN = sigma_BN
        elif sigma_BN is None:
            self.sigma_BN = MRP(0,0,0)
        else:
            self.sigma_BN = MRP.from_array(sigma_BN)

        self.B_omega_BN = np.array([0,0,0]) if B_omega_BN is None else B_omega_BN
        self.actuator_states = actuator_states
    
    def __str__(self):
        """String representation of the spacecraft state."""
        base_str = (f"SpacecraftState:\n"
                   f"  sigma_BN: {self.sigma_BN.as_array()}\n"
                   f"  B_omega_BN: {self.B_omega_BN}\n"
                   f"  Actuator States:\n")
        
        for i, state in enumerate(self.actuator_states):
            base_str += f"    Actuator {i}:\n      {str(state)}"
        return base_str

    def add_actuator_state(self, state:ActuatorState):
        """
        Add an actuator state to the spacecraft state
        
        Args:
            state (ActuatorState): Actuator state to add
        """
        self.actuator_states.append(state)

    def to_array(self, format="new"):
        """Convert state to array format"""
        state_array = np.concatenate((self.sigma_BN.as_array(), self.B_omega_BN))

        if self.actuator_states:
            if format == "new":
                if isinstance(self.actuator_states[0], VscmgState):
                    gimbal_angles = np.array([state.gimbal_angle for state in self.actuator_states])
                    gimbal_rates = np.array([state.gimbal_rate for state in self.actuator_states])
                    wheel_speeds = np.array([state.wheel_speed for state in self.actuator_states])
                    actuator_arrays = np.concatenate((gimbal_angles, gimbal_rates, wheel_speeds))
                else:  # ReactionWheelState
                    wheel_speeds = np.array([state.wheel_speed for state in self.actuator_states])
                    actuator_arrays = wheel_speeds
            else:  # "old" format
                actuator_arrays = np.concatenate([state.to_array() for state in self.actuator_states])

            state_array = np.concatenate((state_array, actuator_arrays))

        return state_array

    @classmethod
    def from_array(cls, array, spacecraft=None, format="new"):
        """
        Create state from array using spacecraft configuration
        
        Args:
            array: The state array
            spacecraft: The spacecraft object that defines the configuration
            format: Array format ("new" or "old")
        """
        if spacecraft is None:
            raise ValueError("Spacecraft must be provided to reconstruct state")
            
        sigma_BN = array[0:3]
        B_omega_BN = array[3:6]
        remaining_data = array[6:]
        
        actuator_states = []
        if len(remaining_data) > 0:
            actuator_type = type(spacecraft.actuators[0].state)
            n_actuators = len(spacecraft.actuators)
            
            if actuator_type == VscmgState:
                if format == "new":
                    gimbal_angles = remaining_data[:n_actuators]
                    gimbal_rates = remaining_data[n_actuators:2*n_actuators]
                    wheel_speeds = remaining_data[2*n_actuators:3*n_actuators]
                    actuator_states = [
                        VscmgState(
                            gimbal_angle=gimbal_angles[i],
                            gimbal_rate=gimbal_rates[i],
                            wheel_speed=wheel_speeds[i]
                        )
                        for i in range(n_actuators)
                    ]
                else:  # "old" format
                    actuator_states = [
                        VscmgState.from_array(remaining_data[i*3:(i+1)*3])
                        for i in range(n_actuators)
                    ]
            else:  # ReactionWheelState
                actuator_states = [
                    ReactionWheelState(wheel_speed=remaining_data[i])
                    for i in range(n_actuators)
                ]
                
        return cls(sigma_BN=sigma_BN, B_omega_BN=B_omega_BN, actuator_states=actuator_states)
