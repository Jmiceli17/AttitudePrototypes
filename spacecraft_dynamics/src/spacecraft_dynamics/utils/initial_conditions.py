"""
Initial conditions of LMO and GMO spacecraft in final project
"""

import numpy as np

from . import MRP

# LMO spacecraft
LMO_OMEGA_0_RAD = np.deg2rad(20.0)
LMO_INC_0_RAD = np.deg2rad(30.0)
LMO_THETA_0_RAD = np.deg2rad(60.0)
LMO_ORBIT_RATE = 0.000884797 # [rad/s]
LMO_ALT_M = 400.0*1000 
LMO_SIGMA_BN_0 = MRP(0.3, -0.4, 0.5)
LMO_B_OMEGA_BN_0 = np.deg2rad(np.array([1.0, 1.75, -2.20]))
LMO_INERTIA=np.array([[10.0, 0.0, 0.0],
                      [0.0, 5.0, 0.0],
                      [0.0, 0.0, 7.5]]) # [kg*m^2]


# GMO spacecraft
GMO_OMEGA_0_RAD = np.deg2rad(0.0)
GMO_INC_0_RAD = np.deg2rad(0.0)
GMO_THETA_0_RAD = np.deg2rad(250.0)
GMO_ORBIT_RATE = 0.0000709003 # [rad/s]
GMO_ALT_M = 17028.01*1000
