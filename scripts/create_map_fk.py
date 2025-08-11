#!/usr/bin/env python3
import math
import numpy as np
import h5py
import rospy
from scipy.spatial import KDTree
from symb_mat_read import T_lamb
import tf.transformations as tft
import time
from datetime import datetime
import os

# ----- Robot parameters -----
joint_offsets = [0, -np.pi/2, 0, -np.pi/2, np.pi/2, 0]

JOINT_LIMITS_DEG = [
        [-165.0, 165.0],
        [-165.0, 165.0],
        [-165.0, 165.0],
        [-165.0, 165.0],
        [-165.0, 165.0],
        [-175.0, 175.0]
    ]
JOINT_LIMITS = [(math.radians(lim[0]), math.radians(lim[1])) for lim in JOINT_LIMITS_DEG]

RESOLUTION = 0.02  # 5cm voxels
HALF_RANGE = 0.3   # 60cm cube workspace
POSES_PER_VOXEL = 32
NUM_FK_SAMPLES = 1000000  # Reduced from 50K for faster testing

name_end_effector = "joint6_flange" # "arm_left_tool_link"
name_base_link = "joint1"
reach_map_file_name = 'reach_map_'+str(name_end_effector)+'_'+str(RESOLUTION)+'_'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
reach_map_file_path = '/home/junaidali/inspection_ws/src/sampled_reachability_map/maps/'

# ----- Kinematics functions -----
def forward_kinematics(joint_angles):
    for i in range(len(joint_angles)):
        joint_angles[i] += joint_offsets[i]
    
    trans_mat_final = np.around(T_lamb(*(joint_angles)), decimals=5)
    pos = np.array([trans_mat_final[0,3], trans_mat_final[1,3], trans_mat_final[2,3]])
    quat = tft.quaternion_from_matrix(trans_mat_final)
    return pos, quat

# ----- Reachability helpers -----
def sample_joint_positions(num_samples, joint_limits):
    joint_limits = np.array(joint_limits, dtype=float)
    low = joint_limits[:, 0]
    high = joint_limits[:, 1]
    rand = np.random.rand(num_samples, len(joint_limits))
    return rand * (high - low) + low

def generate_sphere_orientations(num_orientations):
    orientations = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    base_axis = np.array([0.0, 0.0, 1.0])
    for i in range(num_orientations):
        if num_orientations > 1:
            z = 1.0 - 2.0 * i / float(num_orientations - 1)
        else:
            z = 1.0
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * i
        x = math.cos(theta) * radius
        y = math.sin(theta) * radius
        direction = np.array([x, y, z])
        direction /= np.linalg.norm(direction)
        dot = np.dot(base_axis, direction)
        if dot > 0.999999:
            q = np.array([0.0, 0.0, 0.0, 1.0])
        elif dot < -0.999999:
            q = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            axis = np.cross(base_axis, direction)
            axis /= np.linalg.norm(axis)
            angle = math.acos(dot)
            half_angle = angle / 2.0
            sin_half = math.sin(half_angle)
            q = np.array([
                axis[0] * sin_half,
                axis[1] * sin_half,
                axis[2] * sin_half,
                math.cos(half_angle)
            ])
        q /= np.linalg.norm(q)
        orientations.append(q)
    return orientations

def is_pose_reachable(query_pos, query_ori, fk_tree, fk_orientations, pos_tol=0.02, ori_tol_deg=10.0):
    """Pure FK-based reachability check"""
    cos_half_ori_tol = math.cos(math.radians(ori_tol_deg) / 2.0)
    close_indices = fk_tree.query_ball_point(query_pos, r=pos_tol)
    
    for idx in close_indices:
        q_sample = fk_orientations[idx]
        dot = abs(np.dot(q_sample, query_ori))
        if dot > cos_half_ori_tol:
            return True
    return False

# ----- Main map generation -----
def create_reachability_map():
    rospy.init_node("reachability_map_generator", anonymous=True)
    start_time = time.time()
    rospy.loginfo("Reachability map generation started.")
    print(f"[Will save file named: \'{reach_map_file_name}\' at path: \'{reach_map_file_path}\']")
    
    # 1. Sample joints and compute FK
    joint_samples = sample_joint_positions(NUM_FK_SAMPLES, JOINT_LIMITS)
    fk_positions = []
    fk_orientations = []
    for angles in joint_samples:
        pos, quat = forward_kinematics(angles)
        fk_positions.append(pos)
        fk_orientations.append(quat)
    fk_positions = np.array(fk_positions)
    fk_orientations = np.array(fk_orientations)
    
    # 2. Build spatial index
    fk_tree = KDTree(fk_positions)
    
    # 3. Create voxel grid
    coords = np.arange(-HALF_RANGE, HALF_RANGE + 1e-9, RESOLUTION)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    voxel_centers = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    
    # 4. Generate test orientations
    sphere_orientations = generate_sphere_orientations(POSES_PER_VOXEL)
    
    # 5. Check reachability
    reachability_index = np.zeros(len(voxel_centers), dtype=float)
    max_reach = np.linalg.norm(fk_positions, axis=1).max() + 0.01
    
    for i, center in enumerate(voxel_centers):
        if i % 100 == 0:
            rospy.loginfo(f"Processed {i}/{len(voxel_centers)} voxels")
        
        if np.linalg.norm(center) > max_reach:
            continue
            
        for j, quat in enumerate(sphere_orientations):
            if is_pose_reachable(center, quat, fk_tree, fk_orientations):
                reachability_index[i] += 1.0/POSES_PER_VOXEL
    
    # 6. Save results
    reachable_mask = reachability_index > 0
    
    with h5py.File(reach_map_file_path+"3D_"+reach_map_file_name+".h5", 'w') as f:
        sphere_group = f.create_group('/Spheres')
        sphere_data = np.hstack((voxel_centers[reachable_mask], 
                               reachability_index[reachable_mask].reshape(-1,1)))
        sphere_group.create_dataset('sphere_dataset', data=sphere_data)
        
        pose_group = f.create_group('/Poses')
        
        # Replace pose_group creation with:
        reachable_poses = []
        for i, center in enumerate(voxel_centers):
            if np.linalg.norm(center) > max_reach:
                continue
            for quat in sphere_orientations:
                if is_pose_reachable(center, quat, fk_tree, fk_orientations):
                    reachable_poses.append([center[0], center[1], center[2], 
                                        quat[0], quat[1], quat[2], quat[3]])

        pose_group.create_dataset('pose_dataset', data=np.array(reachable_poses))

if __name__ == "__main__":
    try:
        create_reachability_map()
    except rospy.ROSInterruptException:
        pass