
import numpy as np

from . import rigid_body_kinematics as RBK

class MRP:
    def __init__(self, s1, s2, s3):
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3

    def __str__(self):
        """String representation of the MRP"""
        return (f"[{self.s1}, {self.s2}, {self.s3}]")


    def norm(self):
        """
        Calculate the magnitude of this MRP
        """
        return np.sqrt(self.s1**2 + self.s2**2 + self.s3**2)
    def as_array(self):
        """
        Convert to numpy array to make it easier to work with matrices
        """
        return np.array([self.s1, self.s2, self.s3])
    
    @classmethod
    def from_array(cls,a):
        """
        Create from a numpy array
        """
        s1 = a[0]
        s2 = a[1]
        s3 = a[2]
        return cls(s1,s2,s3)
    def convert_to_shadow_set(self):
        """
        Create a new MRP from the shadow set of this MRP
        """
        s1 = -self.s1/(self.norm()**2)
        s2 = -self.s2/(self.norm()**2)
        s3 = -self.s3/(self.norm()**2)
        return MRP(s1,s2,s3)
    def MRP2C(self):
        """
        Calculate the DCM corresponding to this MRP
        """
        C = RBK.MRP2C(self.as_array())
        return C
    def C2MRP(self, C):
        """
        Calculate the MRP corresponding to a DCM
        """
        sigma_array = RBK.C2MRP(C)
        sigma = self.from_array(sigma_array)
        return sigma