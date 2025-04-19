"""Utility functions and constants for spacecraft dynamics calculations."""

from .constants import MU_MARS, R_MARS
from .modified_rodrigues_parameters import MRP
from .rigid_body_kinematics import (
    MRP2C,
    C2MRP,
    v3Tilde,
    euler3132C
)
from .initial_conditions import (
    LMO_OMEGA_0_RAD,
    LMO_INC_0_RAD,
    LMO_THETA_0_RAD,
    LMO_ORBIT_RATE,
    LMO_ALT_M,
    LMO_SIGMA_BN_0,
    LMO_B_OMEGA_BN_0,
    LMO_INERTIA,
    GMO_OMEGA_0_RAD,
    GMO_INC_0_RAD,
    GMO_THETA_0_RAD,
    GMO_ORBIT_RATE,
    GMO_ALT_M
)

__all__ = [
    "MU_MARS",
    "R_MARS",
    "MRP",
    "MRP2C",
    "C2MRP",
    "v3Tilde",
    "euler3132C",
    "LMO_OMEGA_0_RAD",
    "LMO_INC_0_RAD",
    "LMO_THETA_0_RAD",
    "LMO_ORBIT_RATE",
    "LMO_ALT_M",
    "LMO_SIGMA_BN_0",
    "LMO_B_OMEGA_BN_0",
    "LMO_INERTIA",
    "GMO_OMEGA_0_RAD",
    "GMO_INC_0_RAD",
    "GMO_THETA_0_RAD",
    "GMO_ORBIT_RATE",
    "GMO_ALT_M"
]