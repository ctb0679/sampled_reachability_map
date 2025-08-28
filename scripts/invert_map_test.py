#!/usr/bin/env python3
import math
import numpy as np
import h5py
import rospy
import tf.transformations as tft
from datetime import datetime
import os
import json
# Import the IK solver and necessary forward kinematics function (assumes same package structure)
from incremental_ik import incremental_ik, A_lamb
from symb_mat_read import T_lamb

# ----- Configuration parameters -----
BASE_RESOLUTION = 0.1      # Base position grid resolution in meters (e.g., 0.1 m)
YAW_STEP_DEG = 45.0        # Base orientation sampling step in degrees (e.g., 45°)
SAMPLE_POSE_COUNT = 100    # Number of end-effector poses to sample from reachability map
OUTPUT_CSV = False         # Whether to export results to CSV files
OUTPUT_JSON = False        # Whether to export results to a JSON file

# ----- Robot and IK parameters -----
joint_offsets = [0, -np.pi/2, 0, -np.pi/2, np.pi/2, 0]  # Joint angle offsets for forward kinematics
JOINT_LIMITS_DEG = [                       # Joint limits in degrees (from create_map_fk_ik.py)
    [-165.0, 165.0],
    [-165.0, 165.0],
    [-165.0, 165.0],
    [-165.0, 165.0],
    [-165.0, 165.0],
    [-175.0, 175.0]
]
JOINT_LIMITS = [(math.radians(lim[0]), math.radians(lim[1])) for lim in JOINT_LIMITS_DEG]  # Convert to radians
MAX_THEORETICAL_REACH = 0.4   # Maximum reach radius (in meters) of the manipulator (from create_map_fk_ik.py)

# Initialize IK success/failure counters for statistics
IK_SUCCESS_COUNT = 0
IK_FAILURE_COUNT = 0
IK_TIMEOUT_COUNT = 0

# Tolerances (from create_map_fk_ik.py)
POSITION_TOLERANCE = 0.005      # 5 mm
ORIENTATION_TOLERANCE = 0.0087  # ~0.5 degrees in radians

# ----- Helper functions -----
def check_joint_vector_within_limits(joint_vector, limits):
    """Check if a joint vector lies within specified joint limits (with wrapping)."""
    adjusted = list(joint_vector)  # copy to adjust
    for i in range(len(adjusted)):
        # Normalize angle to [-π, π] range
        adjusted[i] = (adjusted[i] + math.pi) % (2 * math.pi) - math.pi
        # Wrap around 2π to bring within limits
        low, high = limits[i]
        while adjusted[i] < low:
            adjusted[i] += 2 * math.pi
        while adjusted[i] > high:
            adjusted[i] -= 2 * math.pi
        # Final check
        if not (low <= adjusted[i] <= high):
            return False
    return True

def verify_ik_solution(joint_solution, target_tf):
    """
    Verify an IK solution by forward computing the end-effector pose and 
    comparing with target (position and orientation), and checking joint limits.
    Returns (pos_error, ori_error, is_valid_solution).
    """
    # Apply joint offsets for forward kinematics
    fk_joints = [angle + offset for angle, offset in zip(joint_solution, joint_offsets)]
    # Compute forward kinematics transformation matrix for these joints
    T = T_lamb(*fk_joints)   # 4x4 homogeneous transform
    achieved_pos = T[:3, 3]
    achieved_rot = T[:3, :3]
    # Target pose components
    target_pos = target_tf[:3, 3]
    target_rot = target_tf[:3, :3]
    # Compute position error and orientation error
    pos_error = np.linalg.norm(achieved_pos - target_pos)
    rot_diff = achieved_rot @ target_rot.T
    angle = math.acos(np.clip((np.trace(rot_diff) - 1) / 2.0, -1.0, 1.0))
    ori_error = abs(angle)
    # Check joint limits
    within_limits = check_joint_vector_within_limits(joint_solution, JOINT_LIMITS)
    valid_position = pos_error < POSITION_TOLERANCE
    valid_orientation = ori_error < ORIENTATION_TOLERANCE
    is_valid = valid_position and valid_orientation and within_limits
    return pos_error, ori_error, is_valid

def inverse_kinematics(position, orientation, prev_joint_state=None, max_attempts=3, debug=False):
    """
    Attempt to solve IK for the given end-effector pose (position + orientation quaternion).
    Uses the incremental_ik solver with multiple initial guesses and verifies each solution.
    Respects joint limits and returns the first valid joint solution found (or None if none found).
    """
    global IK_SUCCESS_COUNT, IK_FAILURE_COUNT, IK_TIMEOUT_COUNT
    # Construct target transformation matrix from position and quaternion
    tf_mat = tft.quaternion_matrix(orientation)  # 4x4 rotation matrix
    tf_mat[0:3, 3] = position                    # set translation
    # Prepare initial joint state guesses
    init_states = []
    if prev_joint_state is not None:
        init_states.append(prev_joint_state)     # try previous solution first
    # Fill the rest of initial states with random values within joint limits
    for _ in range(max_attempts - len(init_states)):
        random_state = [np.random.uniform(low, high) for (low, high) in JOINT_LIMITS]
        init_states.append(random_state)
    best_solution = None
    best_error_sum = float('inf')
    # Try each initial state
    for init_state in init_states:
        try:
            raw_solution = incremental_ik(init_state, tf_mat)  # IK solver returns joint angles (no offsets)
            if raw_solution is None:
                IK_TIMEOUT_COUNT += 1
                continue  # solver did not find a solution in this attempt
            # Verify the IK result
            pos_err, ori_err, valid = verify_ik_solution(raw_solution, tf_mat)
            if not valid:
                if debug:
                    rospy.logwarn(f"IK solution invalid: Pos error={pos_err*1000:.2f} mm, Ori error={math.degrees(ori_err):.2f}°")
                IK_FAILURE_COUNT += 1
                continue  # solution out of tolerance or joint limits
            # Compute a combined error metric
            total_error = pos_err + ori_err
            if total_error < best_error_sum:
                best_solution = raw_solution
                best_error_sum = total_error
            IK_SUCCESS_COUNT += 1
        except Exception as e:
            if debug:
                rospy.logwarn(f"IK solver exception: {e}")
            IK_FAILURE_COUNT += 1
    if best_solution is not None:
        if debug:
            rospy.loginfo(f"Valid IK solution found with total error {best_error_sum:.6f}")
    return best_solution

def create_inverse_reachability_map(reach_map_path):
    """Generate an inverse reachability map from a given reachability map file."""
    rospy.init_node("inverse_reachability_map_generator", anonymous=True)
    rospy.loginfo("Inverse reachability map generation started.")
    start_time = rospy.Time.now()

    # 1. Load reachable end-effector poses from the reachability map
    rospy.loginfo(f"Loading reachability map from '{reach_map_path}'...")
    if not os.path.isfile(reach_map_path):
        rospy.logerr(f"Reachability map file not found: {reach_map_path}")
        return
    with h5py.File(reach_map_path, 'r') as f:
        poses_dataset = f['/Poses/pose_dataset']  # dataset of reachable poses [x, y, z, qx, qy, qz, qw]
        total_poses = poses_dataset.shape[0]
        rospy.loginfo(f"Reachability map contains {total_poses} reachable end-effector poses.")
        # Determine how many poses to sample
        num_samples = SAMPLE_POSE_COUNT if SAMPLE_POSE_COUNT < total_poses else total_poses
        # Randomly sample a subset of pose indices (without replacement if possible)
        if num_samples < total_poses:
            # Read all poses first, then sample from the array
            all_poses = poses_dataset[:]
            sample_indices = np.random.choice(total_poses, num_samples, replace=False)
            sampled_poses = all_poses[sample_indices]
        else:
            # Read all poses directly
            sampled_poses = poses_dataset[:]
    rospy.loginfo(f"Sampled {len(sampled_poses)} end-effector poses from the reachability map.")

    # Prepare data structures for results
    target_data_list = []         # will hold [x, y, z, qx, qy, qz, qw, inv_reachability%] for each target pose
    base_data_list = []           # will hold [x, y, z, qx, qy, qz, qw] for each valid base placement
    target_to_base_index = []     # will hold [start_index, count] mapping for each target's base placements (in base_data_list)
    results_json = [] if OUTPUT_JSON else None  # optional JSON structure

    # Pre-compute yaw sampling list in radians
    yaw_samples = np.arange(0.0, 360.0, YAW_STEP_DEG)
    yaw_samples = np.append(yaw_samples, 360.0) if yaw_samples[-1] < 360.0 else yaw_samples
    yaw_samples = np.deg2rad(yaw_samples)  # convert to radians

    # 2. Process each sampled end-effector pose
    for t_index, pose in enumerate(sampled_poses):
        # Extract target position and orientation (quaternion) from the pose
        target_pos = np.array(pose[0:3], dtype=float)
        target_quat = np.array(pose[3:7], dtype=float)
        # Ensure quaternion is normalized (safety check)
        target_quat = target_quat / np.linalg.norm(target_quat)

        # Determine horizontal reach limit based on target height (z) for base on ground (z=0)
        vertical_diff = abs(target_pos[2] - 0.0)
        if vertical_diff > MAX_THEORETICAL_REACH:
            # Target is above reachable height for any base on ground (no valid base placements)
            rospy.logwarn(f"Target pose {t_index} at height {target_pos[2]:.2f}m is out of reach for base on ground.")
            # Mark zero reachability and continue
            target_data_list.append([target_pos[0], target_pos[1], target_pos[2],
                                      target_quat[0], target_quat[1], target_quat[2], target_quat[3], 0.0])
            target_to_base_index.append([len(base_data_list), 0])
            if OUTPUT_JSON:
                results_json.append({
                    "target": {
                        "position": [float(target_pos[0]), float(target_pos[1]), float(target_pos[2])],
                        "orientation": [float(q) for q in target_quat],
                        "invReachability": 0.0
                    },
                    "bases": []
                })
            continue

        horizontal_limit = math.sqrt(max(0.0, MAX_THEORETICAL_REACH**2 - vertical_diff**2))
        # Determine grid range in each direction for base positions around the target
        max_offset = horizontal_limit
        step = BASE_RESOLUTION
        n_max = int(math.floor(max_offset / step))
        # Sort candidate base grid points by distance from target (for efficient IK seeding)
        grid_points = []
        for i in range(-n_max, n_max + 1):
            for j in range(-n_max, n_max + 1):
                dx = i * step
                dy = j * step
                # Check within circular radius
                if dx*dx + dy*dy > horizontal_limit**2:
                    continue
                base_x = target_pos[0] + dx
                base_y = target_pos[1] + dy
                # Only consider base at or above ground (base_z = 0 here, so it's always >=0)
                base_z = 0.0
                # Compute squared distance for sorting
                dist_sq = dx*dx + dy*dy
                grid_points.append((dist_sq, base_x, base_y, base_z))
        grid_points.sort(key=lambda entry: entry[0])  # sort by distance squared (ascending)

        # 3. For each target pose, find valid base placements via IK
        valid_bases_for_target = []
        tested_count = 0
        success_count = 0

        # Always include base at origin (0,0,0) with yaw=0 as a candidate if within reach radius.
        # (This corresponds to the original base placement that yielded this reachable pose.)
        base_origin_in_range = (target_pos[0]**2 + target_pos[1]**2) <= horizontal_limit**2
        initial_joint_solution = None
        if base_origin_in_range:
            # Base pose at (0,0,0) with yaw = 0 (identity orientation)
            base_pos = np.array([0.0, 0.0, 0.0], dtype=float)
            base_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)  # quaternion for yaw=0
            # Compute target pose relative to this base frame
            # (Since base at origin with no rotation, relative pose = target pose itself)
            rel_pos = target_pos
            rel_quat = target_quat
            tested_count += 1
            # Solve IK for this pose
            solution = inverse_kinematics(rel_pos, rel_quat, prev_joint_state=None, debug=False)
            if solution is not None:
                success_count += 1
                initial_joint_solution = solution  # store the found joint config
                valid_bases_for_target.append((base_pos.copy(), base_quat.copy()))
        else:
            rospy.loginfo(f"Base at origin is out of range for target {t_index}, skipping direct origin test.")

        # Iterate through each candidate base position (sorted from closest to target outward)
        last_solution = initial_joint_solution
        for _, base_x, base_y, base_z in grid_points:
            # Skip the base position if it coincides with the origin (already handled above)
            if abs(base_x) < 1e-9 and abs(base_y) < 1e-9 and abs(base_z) < 1e-9:
                continue
            # Try all sampled base orientations (yaw angles) at this base position
            for yaw in yaw_samples:
                # Construct base orientation quaternion (assuming base roll/pitch = 0)
                base_quat = tft.quaternion_from_euler(0, 0, yaw)  # (x,y,z,w) format
                base_quat = np.array(base_quat, dtype=float)
                # Compute target pose relative to this base frame
                # Position in base frame: rotate (target_pos - base_pos) by base's inverse rotation
                diff = target_pos - np.array([base_x, base_y, base_z], dtype=float)
                base_rot_inv = tft.quaternion_matrix(tft.quaternion_inverse(base_quat))[:3, :3]
                rel_pos = base_rot_inv.dot(diff)
                # Orientation in base frame: base_inv_quat * target_quat
                rel_quat = tft.quaternion_multiply(tft.quaternion_inverse(base_quat), target_quat)
                rel_quat = rel_quat / np.linalg.norm(rel_quat)  # normalize
                tested_count += 1
                # Solve IK for this relative pose, using last found solution as a hint
                solution = inverse_kinematics(rel_pos, rel_quat, prev_joint_state=last_solution, debug=False)
                if solution is not None:
                    success_count += 1
                    last_solution = solution  # update last successful joint state
                    valid_bases_for_target.append((np.array([base_x, base_y, base_z], dtype=float),
                                                   base_quat.copy()))
            # (Optional performance break: if this target is already fully reachable (e.g., success_count==len(yaw_samples)*(#positions))
            # one could break early. But we'll exhaust all possibilities for thoroughness.)

        # 4. Compute inverse reachability metric for this target pose
        inv_reach_percent = (success_count / tested_count * 100.0) if tested_count > 0 else 0.0
        rospy.loginfo(f"Target {t_index+1}/{len(sampled_poses)}: {success_count} valid base placements out of {tested_count} tested "
                      f"({inv_reach_percent:.1f}% reachability)")

        # Record target data and mapping indices
        target_data_list.append([target_pos[0], target_pos[1], target_pos[2],
                                  target_quat[0], target_quat[1], target_quat[2], target_quat[3],
                                  inv_reach_percent])
        start_index = len(base_data_list)
        count = len(valid_bases_for_target)
        target_to_base_index.append([start_index, count])
        # Append all valid bases for this target to the global base list
        for (b_pos, b_quat) in valid_bases_for_target:
            base_data_list.append([b_pos[0], b_pos[1], b_pos[2],
                                    b_quat[0], b_quat[1], b_quat[2], b_quat[3]])
        # Prepare JSON output entry if needed
        if OUTPUT_JSON:
            bases_json = []
            for (b_pos, b_quat) in valid_bases_for_target:
                bases_json.append({
                    "position": [float(b_pos[0]), float(b_pos[1]), float(b_pos[2])],
                    "orientation": [float(b_quat[0]), float(b_quat[1]), float(b_quat[2]), float(b_quat[3])]
                })
            results_json.append({
                "target": {
                    "position": [float(target_pos[0]), float(target_pos[1]), float(target_pos[2])],
                    "orientation": [float(q) for q in target_quat],
                    "invReachability": float(inv_reach_percent)
                },
                "bases": bases_json
            })
    # End of target loop

    # 5. Save inverse reachability data in HDF5 format
    timestamp_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_res_str = str(BASE_RESOLUTION)
    yaw_step_str = str(int(YAW_STEP_DEG))
    inv_map_file_name = f"inverse_reach_map_{base_res_str}m_{yaw_step_str}deg_{timestamp_str}.h5"
    output_dir = os.path.dirname(os.path.abspath(reach_map_path))
    output_path = os.path.join(output_dir, inv_map_file_name)
    rospy.loginfo(f"Saving inverse reachability map to '{output_path}'...")
    with h5py.File(output_path, 'w') as f_out:
        # Create group for target poses
        target_group = f_out.create_group('/Targets')
        target_dataset = target_group.create_dataset('target_dataset',
                                                    data=np.array(target_data_list, dtype=np.float32))
        target_dataset.attrs.create('Fields', data=np.string_(['x','y','z','qx','qy','qz','qw','invReachPercent']))
        # Mapping from target index to base indices
        index_dataset = target_group.create_dataset('target_index', data=np.array(target_to_base_index, dtype=np.int32))
        index_dataset.attrs.create('Fields', data=np.string_(['base_start_index','base_count']))
        # Create group for base placements
        base_group = f_out.create_group('/Bases')
        base_dataset = base_group.create_dataset('base_dataset', data=np.array(base_data_list, dtype=np.float32))
        base_dataset.attrs.create('Fields', data=np.string_(['base_x','base_y','base_z','base_qx','base_qy','base_qz','base_qw']))
    rospy.loginfo("HDF5 inverse reachability map file saved.")

    # 6. (Optional) Export CSV or JSON if requested
    if OUTPUT_CSV:
        # Write targets CSV
        csv_target_path = os.path.join(output_dir, inv_map_file_name.replace('.h5', '_targets.csv'))
        csv_base_path = os.path.join(output_dir, inv_map_file_name.replace('.h5', '_bases.csv'))
        rospy.loginfo(f"Exporting CSV data to '{csv_target_path}' and '{csv_base_path}'...")
        with open(csv_target_path, 'w') as f_t:
            f_t.write("target_id,x,y,z,qx,qy,qz,qw,invReachability(%)\n")
            for idx, target in enumerate(target_data_list):
                f_t.write(f"{idx},"
                          + ",".join(f"{val:.6f}" if isinstance(val, float) else str(val) for val in target)
                          + "\n")
        with open(csv_base_path, 'w') as f_b:
            f_b.write("target_id,base_x,base_y,base_z,base_qx,base_qy,base_qz,base_qw\n")
            # Use target_to_base_index to map base entries to target
            for t_idx, (start_idx, count) in enumerate(target_to_base_index):
                for b_idx in range(start_idx, start_idx + count):
                    base = base_data_list[b_idx]
                    f_b.write(f"{t_idx},"
                              + ",".join(f"{val:.6f}" if isinstance(val, float) else str(val) for val in base)
                              + "\n")
        rospy.loginfo("CSV export complete.")
    if OUTPUT_JSON:
        json_path = os.path.join(output_dir, inv_map_file_name.replace('.h5', '.json'))
        rospy.loginfo(f"Exporting JSON data to '{json_path}'...")
        with open(json_path, 'w') as f_json:
            json.dump(results_json, f_json, indent=2)
        rospy.loginfo("JSON export complete.")

    # Log summary information
    total_targets = len(target_data_list)
    total_bases = len(base_data_list)
    avg_bases_per_target = (total_bases / total_targets) if total_targets > 0 else 0.0
    rospy.loginfo(f"Inverse reachability mapping completed for {total_targets} target poses.")
    rospy.loginfo(f"Total valid base placements found: {total_bases} (avg {avg_bases_per_target:.1f} per target).")
    # Optionally log IK solver statistics
    rospy.loginfo(f"IK Successes: {IK_SUCCESS_COUNT}, Failures: {IK_FAILURE_COUNT}, Timeouts: {IK_TIMEOUT_COUNT}")
    end_time = rospy.Time.now()
    elapsed = (end_time - start_time).to_sec()
    rospy.loginfo(f"Total computation time: {elapsed/60.0:.1f} minutes.")

# ----- Script entry point -----
if __name__ == "__main__":
    try:
        # Example usage: specify the reachability map file path
        reach_map_file = '/home/idac/Junaidali/catkin_ws/src/sampled_reachability_map/maps/3D_reach_map_0.03_2025-08-14-15-31-51.h5'
        # reachability_map_file = "3D_reach_map_0.03_2025-08-14-15-31-51.h5"
        create_inverse_reachability_map(reach_map_file)
    except rospy.ROSInterruptException:
        rospy.logerr("Inverse reachability map generation interrupted before completion.")
    except Exception as e:
        rospy.logfatal(f"Unexpected error: {e}")
