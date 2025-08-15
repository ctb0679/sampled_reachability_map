#!/usr/bin/env python3
import math
import numpy as np
import h5py
import rospy
from scipy.spatial import KDTree
from symb_mat_read import T_lamb
import tf.transformations as tft
from incremental_ik import incremental_ik
import time

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

RESOLUTION = 0.02  # 0.02 m
HALF_RANGE = 0.3
POSES_PER_VOXEL = 32
NUM_FK_SAMPLES = 100000000

# ----- Helper functions -----
def check_joint_vector_within_limits(joint_angles, joint_limits):
    """Check if all joint angles are within their limits"""
    for i, angle in enumerate(joint_angles):
        low, high = joint_limits[i]
        if angle < low or angle > high:
            return False
    return True

def joint_distance_score(joint_a, joint_b):
    """Calculate distance between two joint configurations"""
    return np.linalg.norm(np.array(joint_a) - np.array(joint_b))

def proximity_to_limits_score(joint_angles, joint_limits):
    """Calculate penalty score for being close to joint limits"""
    score = 0.0
    for i, angle in enumerate(joint_angles):
        low, high = joint_limits[i]
        center = (low + high) / 2.0
        half_range = (high - low) / 2.0
        if half_range > 1e-6:  # Avoid division by zero
            score += abs(angle - center) / half_range
    return score

# ----- Kinematics functions -----
def forward_kinematics(joint_angles):
    for i in range(len(joint_angles)):
        joint_angles[i] += joint_offsets[i]
    
    trans_mat_final = np.around(T_lamb(*(joint_angles)), decimals=5)
    pos = np.array([trans_mat_final[0,3], trans_mat_final[1,3], trans_mat_final[2,3]])
    quat = tft.quaternion_from_matrix(trans_mat_final)
    return pos, quat

def inverse_kinematics(position, orientation, prev_joint_state=None, max_attempts=10, debug=False):
    """
    Enhanced IK solver with multiple attempts and solution validation
    - Tries up to max_attempts different initial states
    - Validates solutions against joint limits
    - Selects best solution based on distance and limit proximity
    - Prints first 10 solutions for debugging
    """
    # Build transformation matrix
    tf_mat = tft.quaternion_matrix(orientation)
    tf_mat[0:3, 3] = position
    
    # Initialize solution tracking
    candidates = []
    solutions_tried = 0
    solutions_found = 0
    
    # Generate initial states to try
    states_to_try = []
    if prev_joint_state is not None:
        states_to_try.append(prev_joint_state)
    for _ in range(max_attempts - len(states_to_try)):
        states_to_try.append([np.random.uniform(low, high) for (low, high) in JOINT_LIMITS])
    
    # Try all initial states
    for init_state in states_to_try:
        try:
            sol = incremental_ik(init_state, tf_mat)
            solutions_tried += 1
            
            # Validate solution
            if sol is not None and check_joint_vector_within_limits(sol, JOINT_LIMITS):
                solutions_found += 1
                
                # Score solution quality
                dist_score = joint_distance_score(init_state, sol)
                limit_score = proximity_to_limits_score(sol, JOINT_LIMITS)
                total_score = dist_score + limit_score
                candidates.append((total_score, sol))
                
                # Print first 10 solutions for debugging
                if debug and solutions_found <= 10:
                    rospy.loginfo(f"IK Solution #{solutions_found}:")
                    # rospy.loginfo(f"  Position: {position}")
                    # rospy.loginfo(f"  Orientation: {orientation}")
                    # rospy.loginfo(f"  Joints: {np.degrees(sol)}")
                    # rospy.loginfo(f"  Scores: dist={dist_score:.4f}, limit={limit_score:.4f}, total={total_score:.4f}")
        except Exception as e:
            pass
    
    # Select best candidate
    if candidates:
        candidates.sort(key=lambda x: x[0])  # Sort by lowest score
        best_sol = candidates[0][1]
        '''
        # Log summary
        if debug:
            rospy.loginfo(f"IK Success: {solutions_found}/{solutions_tried} solutions found")
            rospy.loginfo(f"Best solution score: {candidates[0][0]:.4f}")
        return best_sol
        '''
    # Log failure
    if debug:
        rospy.logwarn(f"IK Failed: 0/{solutions_tried} solutions found for pose:")
        rospy.logwarn(f"  Position: {position}")
        rospy.logwarn(f"  Orientation: {orientation}")
    return None

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

def is_pose_reachable(query_pos, query_ori, fk_tree, fk_orientations, joint_samples, 
                      pos_tol=0.02, ori_tol_deg=10.0, debug=False):
    """Check if pose is reachable with enhanced IK"""
    # First try fast FK sample check
    cos_half_ori_tol = math.cos(math.radians(ori_tol_deg)) / 2.0
    close_indices = fk_tree.query_ball_point(query_pos, r=pos_tol)
    if close_indices:
        for idx in close_indices:
            q_sample = fk_orientations[idx]
            dot = float(q_sample[0]*query_ori[0] + q_sample[1]*query_ori[1] + 
                   q_sample[2]*query_ori[2] + q_sample[3]*query_ori[3])
            if dot < 0.0: 
                dot = -dot
            if dot > cos_half_ori_tol:
                return True
    
    # If FK fails, try IK with good initial guess
    _, closest_idx = fk_tree.query(query_pos)
    solution = inverse_kinematics(
        query_pos, 
        query_ori, 
        prev_joint_state=joint_samples[closest_idx],
        debug=debug
    )
    return solution is not None

# ----- Main map generation -----
def create_reachability_map():
    rospy.init_node("reachability_map_generator", anonymous=True)
    start_time = time.time()
    rospy.loginfo("Reachability map generation started.")
    
    # 1. Sample joints
    rospy.loginfo(f"Sampling {NUM_FK_SAMPLES} joint configurations...")
    joint_samples = sample_joint_positions(NUM_FK_SAMPLES, JOINT_LIMITS)
    
    # 2. Compute FK
    rospy.loginfo("Computing forward kinematics...")
    fk_positions = []
    fk_orientations = []
    for angles in joint_samples:
        pos, quat = forward_kinematics(angles)
        fk_positions.append(pos)
        fk_orientations.append(quat)
    fk_positions = np.array(fk_positions)
    fk_orientations = np.array(fk_orientations)
    
    # Build KDTree for fast position search
    fk_tree = KDTree(fk_positions)
    rospy.loginfo("FK KDTree built.")
    
    # 3. Build voxel grid
    coords = np.arange(-HALF_RANGE, HALF_RANGE + 1e-9, RESOLUTION)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    voxel_centers = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    num_voxels = voxel_centers.shape[0]
    rospy.loginfo(f"Voxel grid: {len(coords)}^3 = {num_voxels} voxels")
    
    # 4. Generate orientations
    sphere_orientations = generate_sphere_orientations(POSES_PER_VOXEL)
    
    # Storage
    reachability_index = np.zeros(num_voxels, dtype=float)
    reachable_flags = np.zeros((num_voxels, POSES_PER_VOXEL), dtype=bool)
    
    # 5. Main processing loop
    max_reach = np.linalg.norm(fk_positions, axis=1).max() + 0.01
    rospy.loginfo("Starting reachability checks...")
    
    # Debug counter - only debug first voxel
    debug_counter = 0
    
    for i, center in enumerate(voxel_centers):
        # Progress logging
        if i % 100 == 0:
            elapsed = time.time() - start_time
            rospy.loginfo(f"Voxel {i}/{num_voxels} - Elapsed: {elapsed:.1f}s")
        
        # Skip unreachable areas
        if np.linalg.norm(center) > max_reach:
            continue
            
        # Only debug first voxel to avoid too many logs
        debug_this_voxel = (debug_counter < 1)
        debug_counter += 1
        
        for j, quat in enumerate(sphere_orientations):
            if is_pose_reachable(
                center, 
                quat, 
                fk_tree, 
                fk_orientations, 
                joint_samples,
                debug=debug_this_voxel
            ):
                reachable_flags[i, j] = True
                
        count_reachable = np.count_nonzero(reachable_flags[i])
        reachability_index[i] = (count_reachable / POSES_PER_VOXEL) * 100.0
    
    rospy.loginfo("Reachability checking complete. Preparing results...")
    
    # Filter and save results (same as before)
    # Filter out voxels that have zero reachability (no poses reachable) to store only reachable voxel centers
    reachable_voxel_mask = reachability_index > 0.0
    reachable_voxels = voxel_centers[reachable_voxel_mask]
    reachable_indices = reachability_index[reachable_voxel_mask]
    
    # Collect all reachable poses (voxel position + orientation) for output
    reachable_poses = []
    for idx, center in enumerate(voxel_centers):
        if reachability_index[idx] <= 0.0:
            continue  # skip unreachable voxels
        # for each orientation at this voxel that is reachable, store the full pose (position + quaternion)
        for j in range(POSES_PER_VOXEL):
            if reachable_flags[idx, j]:
                q = sphere_orientations[j]
                # store as [x, y, z, qx, qy, qz, qw]
                reachable_poses.append([center[0], center[1], center[2], q[0], q[1], q[2], q[3]])
    reachable_poses = np.array(reachable_poses, dtype=float)
    
    # 7. Save the data in HDF5 format (with structure compatible with ROS reachability map tools)
    output_file = "mycobot_reachability_map.h5"
    rospy.loginfo(f"Saving reachability map to {output_file} ...")
    with h5py.File(output_file, 'w') as f:
        # Create group for spheres (voxels)
        sphere_group = f.create_group('/Spheres')
        # Create dataset for sphere centers and reachability index
        # Combine center coordinates and reachability index into one array [x, y, z, reach_idx]
        sphere_data = np.hstack((reachable_voxels, reachable_indices.reshape(-1, 1))).astype(np.float32)
        sphere_dataset = sphere_group.create_dataset('sphere_dataset', data=sphere_data)
        sphere_dataset.attrs.create('Resolution', data=RESOLUTION)
        # Create group for poses
        pose_group = f.create_group('/Poses')
        pose_dataset = pose_group.create_dataset('pose_dataset', data=reachable_poses.astype(np.float32))
        # (No specific attributes for poses; each pose is [x,y,z,qx,qy,qz,qw])
    rospy.loginfo("HDF5 file saved. Reachability map generation completed.")
    # 8. The script uses standard Python and ROS libraries (no Qt) and outputs data for RViz visualization.
    # End of create_reachability_map

# Run as script
if __name__ == "__main__":
    try:
        create_reachability_map()
    except rospy.ROSInterruptException:
        rospy.logerr("Interrupted before completion")
    except Exception as e:
        rospy.logfatal(f"Fatal error: {str(e)}")