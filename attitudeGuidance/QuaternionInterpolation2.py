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
import random

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

# Define data structures to contain the waypoints of the desired trajectory
quaternionArray = []    # This would store the attitude path 
omegaArray = []
timeArray = []
q = Quaternion(1,0,0,0)
dt = 0.1
tf = 10
t = 0
# Make up a trajectory just by integrating the attitude
while t < tf:
    w = Omega(t)        # Time-varying angular velocity, could also make this constant
    # qDot = QDot(w, q)
    # q = q + qDot*dt # This only works for very small steps(?)
    # q = q.normalised
    q.integrate(w, dt)   # Note: see implementation here: https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L948 

    omegaArray.append(w)
    quaternionArray.append(q)
    timeArray.append(t)

    t = t + dt

# print("Integrated Quaternions: \n{}".format(quaternionArray))
# print("Integrated Angular Velocity: \n{}".format(omegaArray))
# print("Times of waypoints: \n{}".format(timeArray))

def QuaternionHermiteSpline(qa, qb, wa, wb, t):
    """
    Hermite Spline interpolation of quaternions
    
    Given two quaternions, and two rates, and a time
    caluclate a quaternion in between these two that obeys the 
    rates
    """
    Beta1 = 1 - (1-t**3)
    Beta2 = 3*(t**2)-(2*(t**3))
    Beta3 = t**3

    # TODO: need to confirm that the 'w's in this algorithm actually correspond to angular velocity
    w1 = wa/3
    w1_asQuat = Quaternion(vector=w1)
    w1_exp = Quaternion.exp(w1_asQuat)  # See http://kieranwynn.github.io/pyquaternion/#exp-and-log-maps 

    w3 = wb/3
    w3_asQuat = Quaternion(vector=w3)
    w3_exp = Quaternion.exp(w3_asQuat)

    w2 = Quaternion.log(w1_exp.inverse*qa.inverse*qb*w3_exp)

    w1B1_asQuat = Quaternion(vector=(w1*Beta1))
    w2B2_asQuat = w2*Beta2
    w3B3_asQuat = Quaternion(vector=(w3*Beta3))

    q_interpolated = qa*Quaternion.exp(w1B1_asQuat)*Quaternion.exp(w2B2_asQuat)*Quaternion.exp(w3B3_asQuat)

    return q_interpolated


# Generate a random list of (incrementing) times to query the Hermite spline and generate an attitude command
evalTimeArray = []
t = 0
# Generate time samples from 0 to the last time of the quaternion path
# TODO: how do we guarantee we actually get to the goal?
while t < timeArray[-1]:
    evalTimeArray.append(t)

    # Generate random time step
    dt = random.uniform(0,1)

    # Increment to the next time
    t = t + dt
# print("Evaluation times: \n{}".format(evalTimeArray))


interpolatedQuaternionArray = []    # Stores the results of Hermite interpolation
i = 0                               # The index of the current waypoint
# Loop through the evaluation times and waypoints and evaluate the Hermite spline
for time in evalTimeArray:
    if time >= timeArray[i] and time < timeArray[i+1]:
        # Evaulate the Hermite spline
        qa = quaternionArray[i]
        qb = quaternionArray[i+1]
        wa = omegaArray[i]
        wb = omegaArray[i+1]
    
        q_interpolated = QuaternionHermiteSpline(qa, qb, wa, wb, time)

        interpolatedQuaternionArray.append(q_interpolated)

        # Increment the current index
        i = i + 1

    else:
        # The current time is outside the current window so increment the index but don't evaluate
        i = i + 1
    

print("Initial Quaternion should be: \n{}".format(quaternionArray[0]))
print("Interpolated Quaternion: \n{}".format(interpolatedQuaternionArray[0]))

print("Final Quaternion should be: \n{}".format(quaternionArray[-1]))
print("Interpolated Quaternion: \n{}".format(interpolatedQuaternionArray[-1]))
