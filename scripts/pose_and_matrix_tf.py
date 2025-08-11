#!/usr/bin/env python3

import numpy as np
from math import acos, atan2, sqrt

def calculate_transformation_matrix(translation_vector, rotation_angles):

    # Extract rotation angles for each axis
    theta_x, theta_y, theta_z = rotation_angles

    # Create rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])

    R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])

    R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])

    # Create the transformation matrix
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.dot(R_z, np.dot(R_y, R_x))
    T[:3, 3] = translation_vector

    return T


def extract_translation_and_rotation(transformation_matrix):

    pose_vector = [0, 0, 0, 0, 0, 0]
    # Extract the translational vector from the transformation matrix
    pose_vector[0:3] = transformation_matrix[:3, 3]

    # Extract the rotational entities (roll, pitch, yaw) from the rotation matrix
    rotation_matrix = transformation_matrix[:3, :3]
    pose_vector[3:6] = extract_rotation(rotation_matrix)

    return pose_vector


def extract_rotation(rotation_matrix):
    # Extract the rotational entities (roll, pitch, yaw) from the rotation matrix
    roll = atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    pitch = atan2(-rotation_matrix[2, 0], sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
    yaw = atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    return [roll, pitch, yaw]
