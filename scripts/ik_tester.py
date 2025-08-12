import numpy as np
from incremental_ik import incremental_ik, A_lamb

# Define the same joint_offsets as in incremental_ik.py
joint_offsets = [0, -np.pi/2, 0, -np.pi/2, np.pi/2, 0]

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles (roll, pitch, yaw) to a 3x3 rotation matrix using ZYX convention."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])

# Known target pose parameters
FIXED_ORIENTATION = (-1.57, 0, 1.57)  # (roll, pitch, yaw) in radians
TARGET_POSITION = (-0.1488, -0.06333, 0.21874)

# Create target transformation matrix
rotation = euler_to_rotation_matrix(*FIXED_ORIENTATION)
target_pose = np.eye(4)
target_pose[:3, :3] = rotation
target_pose[:3, 3] = TARGET_POSITION

# Known solution (WITH OFFSETS APPLIED)
expected_solution = np.array([0.0, 0.523599, 2.0944, -2.61799, 3.14, 0.0])

# Convert to representation used internally by IK solver (WITHOUT OFFSETS)
expected_solution_raw = expected_solution + np.array(joint_offsets)

# Test parameters
INITIAL_PERTURBATION = 0.1  # radians

# Generate perturbed initial state (in solver's internal representation)
perturbed_initial = expected_solution_raw + np.random.uniform(
    -INITIAL_PERTURBATION, 
    INITIAL_PERTURBATION, 
    size=6
)

# Run IK solver
solution = incremental_ik(
    init_joint_state=perturbed_initial,
    goal=target_pose,
)

# Convert to numpy array for comparison
solution = np.array(solution)

# Get forward kinematics result (12x1 vector)
# Must use OFFSET-ADJUSTED angles for FK
solution_for_fk = solution + np.array(joint_offsets)
fk_result = A_lamb(*solution_for_fk).flatten()

# Extract position from FK result (last 3 elements)
position_achieved = fk_result[9:12]

# Calculate errors
joint_error = np.max(np.abs(expected_solution - solution))
position_error = np.linalg.norm(position_achieved - TARGET_POSITION)

# Format results
print("\n" + "="*50)
print("IK SOLVER TEST RESULTS")
print("="*50)
print(f"Target Position: {TARGET_POSITION}")
print(f"Target Orientation (RPY): {FIXED_ORIENTATION} rad")
print("\nExpected Joint Angles (rad):")
print(expected_solution.round(5))
print("\nComputed Joint Angles (rad):")
print(solution.round(5))
print("\nValidation Metrics:")
print(f"Max Joint Error: {joint_error:.6f} rad")
print(f"Position Error: {position_error:.6f} m")
print("="*50)