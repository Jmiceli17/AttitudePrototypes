"""Guidance algorithms and reference trajectory generation."""

from .pointing_guidance import (
    Mode,
    SunPointingControl,
    NadirPointingControl,
    GmoPointingControl,
    MissionPointingControl,
)
from .reference_attitudes import (
    GetSunPointingReferenceAttitude,
    GetSunPointingReferenceAngularVelocity,
    GetNadirPointingReferenceAttitude,
    GetNadirPointingReferenceAngularVelocity,
    GetGmoPointingReferenceAttitude,
    GetGmoPointingReferenceAngularVelocity,
    GetHomework6ReferenceAttitude,
    GetHomework6ReferenceAngularVelocity,
    GetHomework6ReferenceAngularAcceleration
)
from .calculate_dcm_hn import CalculateDcmHN

__all__ = [
    "Mode",
    "SunPointingControl",
    "NadirPointingControl",
    "GmoPointingControl",
    "MissionPointingControl",
    "GetSunPointingReferenceAttitude",
    "GetSunPointingReferenceAngularVelocity",
    "GetNadirPointingReferenceAttitude",
    "GetNadirPointingReferenceAngularVelocity",
    "GetGmoPointingReferenceAttitude",
    "GetGmoPointingReferenceAngularVelocity",
    "GetHomework6ReferenceAttitude",
    "GetHomework6ReferenceAngularVelocity",
    "GetHomework6ReferenceAngularAcceleration",
    "CalculateDcmHN"
]