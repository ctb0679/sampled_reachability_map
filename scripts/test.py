#!/usr/bin/env python3
import math
import numpy as np
import h5py
import rospy
from scipy.spatial import KDTree
from symb_mat_read import T_lamb
import tf.transformations as tft
from incremental_ik import incremental_ik, A_lamb
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

RESOLUTION = 0.05  # 5cm voxels
HALF_RANGE = 0.3
POSES_PER_VOXEL = 8
NUM_FK_SAMPLES = 5000

# Estimated reach limits (adjust based on robot specs)
MAX_THEORETICAL_REACH = 0.4  # 40cm for MyCobot
MIN_Z = -0.2  # Lowest reachable point

# ----- Helper functions -----
def check_joint_vector_within_limits(joint_vector, limits):
    """Improved joint limit checking with 2π wrapping"""
    adjusted = list(joint_vector)  # Make a copy
    for i in range(len(adjusted)):
        # Normalize to [-π, π] range
        adjusted[i] = (adjusted[i] + np.pi) % (2 * np.pi) - np.pi
        
        # Apply 2π wrapping to bring within limits
        low, high = limits[i]
        while adjusted[i] < low:
            adjusted[i] += 2 * np.pi
        while adjusted[i] > high:
            adjusted[i] -= 2 * np.pi
            
        # Final check
        if not (low <= adjusted[i] <= high):
            return False
    return True

# ----- Kinematics functions -----
def forward_kinematics(joint_angles):
    """Compute FK with proper joint offset handling"""
    # Apply joint offsets for FK calculation
    adjusted_angles = [a + offset for a, offset in zip(joint_angles, joint_offsets)]
    
    # Compute transformation matrix
    trans_mat_final = np.around(T_lamb(*adjusted_angles), decimals=5)
    
    # Extract position and orientation
    pos = np.array([trans_mat_final[0,3], trans_mat_final[1,3], trans_mat_final[2,3]])
    quat = tft.quaternion_from_matrix(trans_mat_final)
    return pos, quat

def inverse_kinematics(position, orientation, prev_joint_state=None, max_attempts=3, debug=False):
    """
    Enhanced IK solver with verification and limit handling
    - Uses incremental_ik core solver
    - Adds FK verification
    - Proper joint limit handling
    """
    # Convert orientation to transformation matrix
    tf_mat = tft.quaternion_matrix(orientation)
    tf_mat[0:3, 3] = position
    
    # Prepare initial states
    init_states = []
    if prev_joint_state is not None:
        init_states.append(prev_joint_state)
    for _ in range(max_attempts - len(init_states)):
        # Generate random within limits
        init_states.append([np.random.uniform(low, high) for (low, high) in JOINT_LIMITS])
    
    best_solution = None
    best_error = float('inf')
    
    for init_state in init_states:
        try:
            # Solve IK (returns joints WITHOUT offsets)
            raw_solution = incremental_ik(init_state, tf_mat)
            
            if raw_solution is None:
                continue
                
            # Check joint limits with proper wrapping
            if not check_joint_vector_within_limits(raw_solution, JOINT_LIMITS):
                continue
                
            # Verify solution with FK
            # Note: FK expects joints WITH offsets
            fk_joints = [r + o for r, o in zip(raw_solution, joint_offsets)]
            achieved_pos, achieved_quat = forward_kinematics(fk_joints)
            
            # Calculate errors
            pos_error = np.linalg.norm(achieved_pos - position)
            ori_error = 1 - abs(np.dot(orientation, achieved_quat))  # 1 - |dot product|
            
            total_error = pos_error + ori_error
            
            if total_error < best_error:
                best_solution = raw_solution
                best_error = total_error
                
            if debug:
                rospy.loginfo(f"IK attempt: Pos error={pos_error:.4f}m, Ori error={ori_error:.4f}")
                
        except Exception as e:
            if debug:
                rospy.logwarn(f"IK exception: {str(e)}")
    
    if best_solution is not None and debug:
        rospy.loginfo(f"Best IK solution found with error {best_error:.4f}")
        
    return best_solution

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
                      pos_tol=0.05, ori_tol_deg=25.0, debug=False):
    """Optimized reachability check with pre-filter"""
    # 1. Quick out-of-reach check
    if (np.linalg.norm(query_pos) > MAX_THEORETICAL_REACH or 
        query_pos[2] < MIN_Z):
        return False
    
    # 2. Broad FK filter
    cos_half_ori_tol = math.cos(math.radians(ori_tol_deg) / 2.0)
    close_indices = fk_tree.query_ball_point(query_pos, r=pos_tol)
    
    for idx in close_indices:
        q_sample = fk_orientations[idx]
        dot = abs(np.dot(q_sample, query_ori))
        if dot > cos_half_ori_tol:
            return True
    
    # 3. Fallback to IK
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
    for i, angles in enumerate(joint_samples):
        pos, quat = forward_kinematics(angles)
        fk_positions.append(pos)
        fk_orientations.append(quat)
        
        # Progress logging
        if i % 1000 == 0:
            rospy.loginfo(f"FK computed for {i}/{NUM_FK_SAMPLES} samples")
    
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
    
    # Progress tracking
    total_poses = num_voxels * POSES_PER_VOXEL
    processed_poses = 0
    start_loop_time = time.time()
    
    for i, center in enumerate(voxel_centers):
        # Skip unreachable areas
        if np.linalg.norm(center) > max_reach:
            processed_poses += POSES_PER_VOXEL
            continue
            
        # Only debug first voxel
        debug_this_voxel = (i == 0)
        
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
                
            processed_poses += 1
            
            # Log progress every 60 seconds
            current_time = time.time()
            if current_time - start_loop_time > 60:
                elapsed_total = current_time - start_time
                elapsed_loop = current_time - start_loop_time
                progress = processed_poses / total_poses * 100
                rospy.loginfo(
                    f"Progress: {progress:.1f}% | "
                    f"Poses: {processed_poses}/{total_poses} | "
                    f"Elapsed: {elapsed_total/60:.1f} min | "
                    f"Rate: {processed_poses/elapsed_loop:.1f} poses/s"
                )
                start_loop_time = current_time
                
        # Update reachability index after each voxel
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
    
    total_time = time.time() - start_time
    rospy.loginfo(f"Map generation completed in {total_time/60:.1f} minutes")

# Run as script
if __name__ == "__main__":
    try:
        create_reachability_map()
    except rospy.ROSInterruptException:
        rospy.logerr("Interrupted before completion")
    except Exception as e:
        rospy.logfatal(f"Fatal error: {str(e)}")