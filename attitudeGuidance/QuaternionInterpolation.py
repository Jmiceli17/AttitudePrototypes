# Prototyping quaternion interpolation
# This file leverages the numpy quaternion module 
# https://github.com/moble/quaternion which is based on
# Quaternionic (the pure python version)

# Other modules that may be worth trying:
# http://kieranwynn.github.io/pyquaternion/#welcome 


import numpy as np
import quaternion

# Generate a 10x4 random array
randArray = np.random.rand(10, 4)

# Convert each row to a quaternion
quaternionArray = quaternion.as_quat_array(randArray)

print("In Quaternions: \n{}".format(quaternionArray))

# Define times corresponding to each quaternion in the array
inTimes = np.array([i*10 for i in range(0,10)])
print("inTimes: {}".format(inTimes))

# Define the times at which we want to evaluate the polynomial
outTimes = np.array([i*5 for i in range(0,20)])
print("outTimes: {}".format(outTimes))

interpolatedQuaternionArray = quaternion.squad(quaternionArray, inTimes, outTimes, unflip_input_rotors=True)
print("Out Quaternions: \n{}".format(interpolatedQuaternionArray))
