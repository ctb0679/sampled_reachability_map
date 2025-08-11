#!/usr/bin/env python3
import math
import numpy as np
import h5py
import rospy
from symb_mat_read import T_lamb
import tf.transformations as tft
from incremental_ik import incremental_ik
# (Optional: import of ROS message types if needed, e.g., for IK inputs)
# from geometry_msgs.msg import Pose

# ----- Define robot-specific kinematics (assumed to be provided by user) -----

# Forward kinematics function (to be provided by user or implemented using URDF/DH).
# This should take a sequence of 6 joint angles (radians) and return the end-effector
# pose (position and orientation). Here we define a placeholder.

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

POSES_PER_VOXEL = 32  # number of orientation samples per voxel (between 30 and 50 as specified)
NUM_FK_SAMPLES = 50000  # can be adjusted for more coverage (trade-off with computation time)


def forward_kinematics(joint_angles):
    # *** User should replace this with actual FK computation using the URDF or known DH parameters. ***
    # For demonstration, return a dummy pose (this would be replaced by real calculations).
    # (e.g., using KDL or custom DH model to get transform of end-effector).
    # Here we just return origin with identity orientation as a placeholder.

    for i in range(len(joint_angles)):
        joint_angles[i] += joint_offsets[i]
    
    trans_mat_final = np.around(T_lamb(*(joint_angles)), decimals=5)
    # Extract position
    pos = np.array([trans_mat_final[0,3], trans_mat_final[1,3], trans_mat_final[2,3]])
    # Compute quaternion from matrix
    quat = tft.quaternion_from_matrix(trans_mat_final)
    return pos, quat

# Inverse kinematics function (to be provided by user).
# This should take a desired pose (position and orientation) and return a valid joint solution if one exists.
def inverse_kinematics(position, orientation):
    # *** User should replace this with actual IK solver (analytical or numerical). ***
    # For demonstration, return None to indicate no solution (to be replaced with real IK).
    # Build 4x4 transform matrix from quaternion+translation
    tf_mat = tft.quaternion_matrix(orientation)
    tf_mat[0:3,3] = position
    # Provide a random initial guess within limits
    init = [np.random.uniform(low, high) for (low, high) in JOINT_LIMITS]
    try:
        sol = incremental_ik(init, tf_mat)
        return sol
    except Exception:
        return None

# ----- Helper functions for reachability map computation -----

# 1. Sample joint positions within the provided joint limits.
def sample_joint_positions(num_samples, joint_limits):
    """
    Sample random joint configurations within given joint limits.
    joint_limits: list of (min_angle, max_angle) in radians for each joint.
    Returns: NumPy array of shape (num_samples, 6) with sampled joint angles.
    """
    joint_limits = np.array(joint_limits, dtype=float)  # shape (6, 2)
    low = joint_limits[:, 0]
    high = joint_limits[:, 1]
    # Uniform random sampling in joint space
    rand = np.random.rand(num_samples, len(joint_limits))
    samples = rand * (high - low) + low
    return samples

# 4. Generate a set of poses (orientations) uniformly distributed on a sphere.
def generate_sphere_orientations(num_orientations):
    """
    Generate `num_orientations` uniformly distributed orientation quaternions 
    such that the end-effector's tool axis (assumed to be the z-axis) points 
    in those directions.
    Returns: list of quaternions [qx, qy, qz, qw] representing orientations.
    """
    orientations = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))  # golden angle for even distribution
    base_axis = np.array([0.0, 0.0, 1.0])  # assume end-effector's tool axis is Z-axis
    for i in range(num_orientations):
        # Uniformly distribute points on unit sphere using spherical coordinates (Fibonacci lattice)
        if num_orientations > 1:
            z = 1.0 - 2.0 * i / float(num_orientations - 1)  # z coordinate
        else:
            z = 1.0
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * i
        x = math.cos(theta) * radius
        y = math.sin(theta) * radius
        direction = np.array([x, y, z])
        direction = direction / np.linalg.norm(direction)  # normalize (should already be unit)
        # Compute quaternion that rotates base_axis ([0,0,1]) to this direction vector.
        dot = np.dot(base_axis, direction)
        if dot > 0.999999:
            # direction is almost the same as base_axis
            q = np.array([0.0, 0.0, 0.0, 1.0])  # no rotation needed
        elif dot < -0.999999:
            # direction is opposite to base_axis
            # Rotate 180° around an arbitrary axis perpendicular to base_axis (e.g., X-axis)
            q = np.array([1.0, 0.0, 0.0, 0.0])  # 180 deg rotation around X-axis
        else:
            # General case: axis-angle rotation from base_axis to direction
            axis = np.cross(base_axis, direction)
            axis = axis / np.linalg.norm(axis)
            angle = math.acos(dot)
            half_angle = angle / 2.0
            sin_half = math.sin(half_angle)
            q = np.array([
                axis[0] * sin_half,
                axis[1] * sin_half,
                axis[2] * sin_half,
                math.cos(half_angle)
            ])
        # Normalize quaternion to be safe
        q = q / np.linalg.norm(q)
        orientations.append(q)
    return orientations

# 5. Check if a given pose (position + orientation) is reachable by comparing to FK samples and optional IK.
def is_pose_reachable(query_pos, query_ori, fk_positions, fk_orientations, pos_tol=0.02, ori_tol_deg=10.0):
    """
    Determine if a pose is reachable by the robot by checking against precomputed FK samples 
    and using IK as fallback.
    - query_pos: target position [x, y, z]
    - query_ori: target orientation quaternion [qx, qy, qz, qw]
    - fk_positions: array of shape (N, 3) of positions from sampled FK
    - fk_orientations: array of shape (N, 4) of quaternions from sampled FK
    - pos_tol: position tolerance (meters) for matching FK samples
    - ori_tol_deg: orientation tolerance (degrees) for matching FK samples
    Returns: True if reachable, False otherwise.
    """
    # Position tolerance and orientation tolerance (in radians for half-angle)
    pos_tol_sq = pos_tol ** 2
    cos_half_ori_tol = math.cos(math.radians(ori_tol_deg) / 2.0)
    # First, quickly filter FK samples by position proximity
    # (find samples within pos_tol in Euclidean distance)
    # We do a squared distance check for efficiency
    diff = fk_positions - query_pos
    dist_sq = np.sum(diff**2, axis=1)
    close_idx = np.where(dist_sq <= pos_tol_sq)[0]
    if close_idx.size > 0:
        # Among close positions, check orientation similarity
        for idx in close_idx:
            # Compute quaternion dot product (assuming both quaternions are normalized)
            q_sample = fk_orientations[idx]
            # Ensure we account for quaternion double-cover (q and -q represent same orientation)
            dot = float(q_sample[0]*query_ori[0] + q_sample[1]*query_ori[1] + 
                        q_sample[2]*query_ori[2] + q_sample[3]*query_ori[3])
            if dot < 0.0:
                dot = -dot  # use absolute, as q and -q are equivalent
            if dot > cos_half_ori_tol:
                # Position and orientation are close to a sampled FK pose
                return True
    # If no FK sample was close enough, use the IK solver as a fallback
    solution = inverse_kinematics(query_pos, query_ori)
    if solution is not None:
        return True
    return False

# ----- Main function to build the reachability map -----

def create_reachability_map():
    # Initialize ROS node (for logging, etc.)
    rospy.init_node("reachability_map_generator", anonymous=True)
    rospy.loginfo("Reachability map generation started.")
    
    # Number of random joint samples for forward kinematics coverage    
    # 1. Sample random joint positions within joint limits
    rospy.loginfo(f"Sampling {NUM_FK_SAMPLES} random joint configurations...")
    joint_samples = sample_joint_positions(NUM_FK_SAMPLES, JOINT_LIMITS)
    
    # 2. Use forward kinematics to compute end-effector poses for each joint sample
    rospy.loginfo("Computing forward kinematics for sampled joint configurations...")
    fk_positions = []
    fk_orientations = []
    for angles in joint_samples:
        pos, quat = forward_kinematics(angles)
        fk_positions.append(pos)
        fk_orientations.append(quat)
    fk_positions = np.array(fk_positions)
    fk_orientations = np.array(fk_orientations)
    rospy.loginfo("Forward kinematics sampling complete.")
    
    # 3. Build a 3D voxel grid around the robot base (0,0,0) with specified resolution and size
       # half the side length (0.6m cube centered at base)
    # Create grid coordinates for voxel centers within [-0.3, 0.3] in x, y, z
    coords = np.arange(-HALF_RANGE, HALF_RANGE + 1e-9, RESOLUTION)  # include end point
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    voxel_centers = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T  # shape (num_voxels, 3)
    num_voxels = voxel_centers.shape[0]
    rospy.loginfo(f"Voxel grid created: {len(coords)}^3 = {num_voxels} voxels.")
    
    # 4. For each voxel, pre-generate a set of orientations on a sphere
    rospy.loginfo(f"Generating {POSES_PER_VOXEL} orientations on a sphere for reachability testing...")
    sphere_orientations = generate_sphere_orientations(POSES_PER_VOXEL)
    
    # Prepare storage for results
    reachability_index = np.zeros(num_voxels, dtype=float)        # reachability index (%) for each voxel
    reachable_flags = np.zeros((num_voxels, POSES_PER_VOXEL), dtype=bool)  # flags for each orientation pose
    
    # 5. Check reachability for each voxel center and each sampled orientation
    rospy.loginfo("Checking reachability of each voxel and orientation...")
    # Determine a cutoff distance beyond which we skip checking (robot's approximate reach)
    max_reach = np.linalg.norm(fk_positions, axis=1).max()
    # (Optional) Add a small margin to max_reach if needed
    max_reach += 0.01  # 1 cm margin
    for i, center in enumerate(voxel_centers):
        # Skip this voxel if it is clearly outside the robot's reach (distance > max_reach)
        if np.linalg.norm(center) > max_reach:
            reachability_index[i] = 0.0
            continue
        # For each orientation on the sphere at this voxel, check reachability
        for j, quat in enumerate(sphere_orientations):
            # Construct the end-effector pose at this voxel center with orientation quat
            pos = center
            ori = quat
            # Check if this pose is reachable via any FK sample or IK solution
            if is_pose_reachable(pos, ori, fk_positions, fk_orientations):
                reachable_flags[i, j] = True
        # 6. Compute reachability index for this voxel (percentage of orientations reachable)
        count_reachable = np.count_nonzero(reachable_flags[i])
        reachability_index[i] = (count_reachable / float(POSES_PER_VOXEL)) * 100.0
    
    rospy.loginfo("Reachability checking complete. Preparing results for output...")
    
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

# If run as a script, execute the reachability map generation
if __name__ == "__main__":
    try:
        create_reachability_map()
    except rospy.ROSInterruptException:
        pass
