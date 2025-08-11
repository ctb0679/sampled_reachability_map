#!/usr/bin/env python3

import random
import numpy as np
from symb_mat_read import *

#Define the robot joint limits
limits = [[-np.radians(165), np.radians(165)], [-np.radians(165), np.radians(165)], [-np.radians(165), np.radians(165)], 
          [-np.radians(165), np.radians(165)], [-np.radians(165), np.radians(165)], [-np.radians(175), np.radians(175)]]

joint_offsets = [0, -np.pi/2, 0, -np.pi/2, np.pi/2, 0]

#Function to create a random starting point for IK if the provided one doesn't work
def generate_random_vector():
    random_vector = []
    
    for limit in limits:
        lower_limit, upper_limit = limit
        random_angle = random.uniform(lower_limit, upper_limit)
        random_vector.append(random_angle)
    return random_vector


#Function to check if the result is within limits or can be brought in limits by adding/subtracting multiples of 2*pi
def check_joint_vector_within_limits(joint_vector, limits):

    for i in range(len(joint_vector)):
        
        if joint_vector[i] < limits[i][0]:
            while joint_vector[i] < limits[i][0]:
                joint_vector[i] += 2*np.pi  
        elif joint_vector[i] > limits[i][1]:
            while joint_vector[i] > limits[i][1]:
                joint_vector[i] -= 2*np.pi

        if not (limits[i][0] <= joint_vector[i] <= limits[i][1]):
            return False

    return True


# Provide q_currrent as an array of current joint angles and goal pose as a transformation matrix
def incremental_ik(init_joint_state, goal, steps = 1000, tol=0.001):

    q_current = init_joint_state

    #Iterate till we achieve satisfactory results
    while(True):
        current_pose = A_lamb(*(q_current))

        #get the goal pose in (12,1) vector
        goal_pose = goal[0:3, 0:4] #crop last row
        goal_pose = goal_pose.transpose().reshape(12,1) #reshape the matrix

        error = goal_pose - current_pose
        i=0

        while np.max(np.abs(error)) > tol and i<steps:

            J_current = J_lamb(*(q_current))

            delta_q = np.linalg.pinv(J_current) @ error
            q_current = (q_current + delta_q.flatten())

            current_pose = A_lamb(*(q_current.flatten()))
            error = goal_pose - current_pose
            i += 1
        
        q_current = q_current.flatten().tolist()

        for i in range(len(q_current)):
            q_current[i] -= joint_offsets[i]

        limit_check = check_joint_vector_within_limits(q_current, limits)

        if (limit_check == True):
            break
        else:
            q_current = generate_random_vector()
            continue

    return q_current