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
quaternionArray = []
omegaArray = []
while t < tf:
    w = Omega(t)
    # qDot = QDot(w, q)
    # q = q + qDot*dt # This only works for very small steps(?)
    # q = q.normalised
    q.integrate(w, dt)

    omegaArray.append(w)
    quaternionArray.append(q)

    t = t + dt

print("Integrated Quaternions: \n{}".format(quaternionArray))
print("Integrated Angular Velocity: \n{}".format(omegaArray))


# Construct Hermite Spline



