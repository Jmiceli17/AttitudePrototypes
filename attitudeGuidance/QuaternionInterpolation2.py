# Prototyping quaternion interpolation
# This file leverages the pyquaternion module
# http://kieranwynn.github.io/pyquaternion/#welcome 
# and attempts to utilize the Hermite spline approach to 
# interpolate a path between quaternions
# 
# The Hermite spline being referenced is from here:
# http://graphics.cs.cmu.edu/nsp/course/15-464/Fall05/papers/kimKimShin.pdf


import numpy as np
from pyquaternion import Quaternion

# define angular velocity as a function of time
def Omega(t):

    w0 = np.sin(0.1*t)
    w1 = 0.01
    w2 = np.cos(0.1*t)

    theta = np.deg2rad(20)
    w = theta*np.array([w0, w1, w2])    # rad/s

    return w


# Define the kinematic differential equation of quaternions
# Qdot = F(w,q)
# Note: pyquaternion has a built in integration function
# def QDot(omega, quat:Quaternion):

#     W = np.array([[0, -omega[0], -omega[1], -omega[2]],
#                 [omega[0], 0, -omega[2], -omega[1]],
#                 [omega[1], -omega[2], 0, omega[0]],
#                 [omega[2], omega[1], -omega[0], 0]])
#     qDot = 1/2*np.dot(W,quat)
#     print("qDot: {}".format(qDot))
#     return qDot

# Do a basic numerical integration scheme 
# X = X + Xdot*dt
q = Quaternion(1,0,0,0)
dt = 0.1
tf = 10
t = 0
quaternionArray = []    # This would store the attitude path 
omegaArray = []
while t < tf:
    w = Omega(t)        # Time-varying angular velocity, could also make this constant
    # qDot = QDot(w, q)
    # q = q + qDot*dt # This only works for very small steps(?)
    # q = q.normalised
    q.integrate(w, dt)   # Note: see implementation here: https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L948 

    omegaArray.append(w)
    quaternionArray.append(q)

    t = t + dt

print("Integrated Quaternions: \n{}".format(quaternionArray))
print("Integrated Angular Velocity: \n{}".format(omegaArray))


# Construct Hermite Spline
# Given sequence of quaternions [q] and rates [w]
# For qi and qi+1 in [q] and wi and wi+1 in [w]
#   Calculate Beta
#   
# Gievn two quaternions, and two rates, and a time
# caluclate a quaternion in between these two that obeys the 
# rates
def QuaternionHermiteSpline(q, w, t):
    Beta1 = 1 - (1-t**3)
    Beta2 = 3*(t**2)-(2*(t**3))
    Beta3 = t**3

    qa = q[0]
    qb = q[1]

    w1 = w[0]/3
    w1_asQuat = Quaternion(vector=w1)
    w1_exp = Quaternion.exp(w1_asQuat)

    w3 = w[1]/3
    w3_asQuat = Quaternion(vector=w3)
    w3_exp = Quaternion.exp(w3_asQuat)

    w2 = Quaternion.log(w1_exp.inverse*qa.inverse*qb*w3_exp)

    w1B1_asQuat = Quaternion(vector=(w1*Beta1))
    w2B2_asQuat = w2*Beta2
    w3B3_asQuat = Quaternion(vector=(w3*Beta3))

    q_interpolated = qa*Quaternion.exp(w1B1_asQuat)*Quaternion.exp(w2B2_asQuat)*Quaternion.exp(w3B3_asQuat)

    return q_interpolated

q_ineterp = QuaternionHermiteSpline(quaternionArray, omegaArray, dt)

print("Initial Quaternion should be: \n{}".format(quaternionArray[0]))
print("Interpolated Quaternion: \n{}".format(q_ineterp))

print("Q[{}] should be: \n{}".format(1, quaternionArray[1]))
print("Interpolated Quaternion: \n{}".format(q_ineterp))
