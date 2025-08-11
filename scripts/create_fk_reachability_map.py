import os
from datetime import datetime
import pickle
import h5py

import math
import numpy as np
import pytorch_kinematics as pk
import torch

import time
import pdb

### Code to create a reachability map using pytorch forward kinematics (GPU-based tensor calculations)


# Use CUDA if available
if torch.cuda.is_available():
    d = "cuda"
    torch.cuda.empty_cache()
    print("[GPU MEMORY size in GiB]: " + str((torch.cuda.get_device_properties(0).total_memory-torch.cuda.memory_reserved(0))/1024**3))
else:
    d = "cpu"
dtype = torch.float32 # Choose float32 or 64 etc.


## Settings for the reachability map:
robot_urdf = "mycobot_urdf.urdf"
name_end_effector = "joint6_flange" # "arm_left_tool_link"
name_base_link = "joint1"
use_torso = False
n_dof = 6 # Implied from the URDF and chosen links. 'use_torso=False' will reduce this by one in practice
# Number of DOFs and joint limits
joint_pos_min = torch.tensor([-2.87979, -2.87979, -2.87979, -2.87979, -2.87979, -3.05433], dtype=dtype, device=d)
joint_pos_max = torch.tensor([2.87979, 2.87979, 2.87979, 2.87979, 2.87979, 3.05433], dtype=dtype, device=d)	
joint_pos_centers = joint_pos_min + (joint_pos_max - joint_pos_min)/2
joint_pos_range_sq = (joint_pos_max - joint_pos_min).pow(2)/4
## Build kinematic chain from URDF
print("[Building kinematic chain from URDF...]:\n...\n...")
chain = pk.build_serial_chain_from_urdf(open(robot_urdf).read(), name_end_effector)
chain = chain.to(dtype=dtype, device=d)
assert (len(chain.get_joint_parameter_names()) == n_dof), "Incorrect number of DOFs set"
print("...\n...")
# Number of Forward Kinematic solutions to sample
N_fk = 10000000 # 25600000000 # Sampling 20^8 joint configurations. NOTE: Tweak this paramter based on GPU Memory available
# Map resolution and limits
angular_res = np.pi/8 # or 22.5 degrees per bin)
r_lim = [-np.pi, np.pi] # NOTE: Using 'intrinsic' euler rotations in XYZ
p_lim = [-np.pi/2, np.pi/2]
yaw_lim = [-np.pi, np.pi]
roll_bins = math.ceil((2*np.pi)/angular_res) # 16
pitch_bins = math.ceil((np.pi)/angular_res)  # 8. Only half the bins needed (half elevation and full azimuth sufficient to cover sphere)
yaw_bins = math.ceil((2*np.pi)/angular_res)  # 16
cartesian_res = 0.05 # metres
x_lim = [-0.3, 0.3]  # Meters (adjust after initial tests)
y_lim = [-0.3, 0.3]
z_lim = [0.0, 0.3]   # Base height to max reach
x_bins = math.ceil((x_lim[1] - x_lim[0])/cartesian_res)
y_bins = math.ceil((y_lim[1] - y_lim[0])/cartesian_res)
z_bins = math.ceil((z_lim[1] - z_lim[0])/cartesian_res)
post_process = False # Whether to filter out voxels with post processing


## Create 6D reachability map initialized with zeros
# NOTE: Save map in CPU. Save GPU memory for kinematics, jacobian calculations
num_voxels = x_bins*y_bins*z_bins*roll_bins*pitch_bins*yaw_bins
num_values = 6+2 # Values in every voxel. Reachability map will store the 6D pose + two values: 'Visitation Frequency' and 'Manipulability'
reach_map = torch.zeros((num_voxels,num_values), dtype=dtype, device="cpu") # Store map in CPU to save memory
print("[Number of 6D Voxels possible (Based on resolution settings)]: " + str(num_voxels))
# Full path and file name to save
reach_map_file_path = os.getcwd()+'/../maps/'
reach_map_file_name = 'reach_map_'+str(name_end_effector)+'_torso_'+str(use_torso)+'_'+str(cartesian_res)+'_'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
# TODO: Allow loading an existing reach_map from a file
# Offsets for indexing the map
x_ind_offset = y_bins*z_bins*roll_bins*pitch_bins*yaw_bins
y_ind_offset = z_bins*roll_bins*pitch_bins*yaw_bins
z_ind_offset = roll_bins*pitch_bins*yaw_bins
roll_ind_offset = pitch_bins*yaw_bins
pitch_ind_offset = yaw_bins
yaw_ind_offset = 1
offsets = torch.tensor([x_ind_offset, y_ind_offset, z_ind_offset, roll_ind_offset, pitch_ind_offset, yaw_ind_offset], dtype=torch.long, device=d)


## Set number of sampled joint configurations as per GPU memory limitations
num_loops = 100 # 10000 # NOTE: Tweak this paramter based on GPU Memory available
N_fk_loop = int(N_fk/num_loops)
save_freq = int(num_loops/100)
sampling_distr = torch.distributions.uniform.Uniform(joint_pos_min, joint_pos_max) # For sampling joints within limits

## Compute
print("[Starting FK and Jacobian calculations...]")
print("[Number of loops is: "+str(num_loops)+"]")
print("[Map save frequency is "+str(save_freq)+" loops]")
print(f"[Will save file named: \'{reach_map_file_name}\' at path: \'{reach_map_file_path}\']")
t0 = time.perf_counter()
for i in range(num_loops):
    loop_t0 = time.perf_counter()

    # Sample joints
    th_batch = sampling_distr.sample([N_fk_loop])
    if not use_torso:
        # Set torso joint pos to zero
        th_batch[:,0] = torch.tensor(0.0,dtype=dtype,device=d)
    ee_transf_batch = chain.forward_kinematics(th_batch).get_matrix()
    torch.cuda.empty_cache() # Keep clearing cache to get rid of redundant variables
    poses_6d = torch.hstack((ee_transf_batch[:,:3,3], pk.transforms.matrix_to_euler_angles(ee_transf_batch[:,:3,:3], 'XYZ'))) # NOTE: Using 'intrinsic' euler rotations in XYZ
    # Get indices by subtracting lower lim and dividing by resolution
    indices_6d = poses_6d - torch.tensor([x_lim[0],y_lim[0],z_lim[0],r_lim[0],p_lim[0],yaw_lim[0]], dtype=dtype, device=d)
    indices_6d /= torch.tensor([cartesian_res,cartesian_res,cartesian_res,angular_res,angular_res,angular_res], dtype=dtype, device=d)
    indices_6d = torch.floor(indices_6d) # Floor to get the appropriate discrete indices
    # Sanity checks and handling of edge cases of discretization (angles can especially cause issues if values contain both ends [-pi, pi] which we don't want
    indices_6d[indices_6d[:,3]>=roll_bins, 3] = roll_bins-1
    indices_6d[indices_6d[:,4]>=pitch_bins, 4] = pitch_bins-1
    indices_6d[indices_6d[:,5]>=yaw_bins, 5] = yaw_bins-1
    if(torch.sum((indices_6d[:,3]>=roll_bins)|(indices_6d[:,4]>=pitch_bins)|(indices_6d[:,5]>=yaw_bins)|(indices_6d[:,0]>=x_bins)|(indices_6d[:,1]>=y_bins)|(indices_6d[:,2]>=z_bins)) > 0):
        print("[WARNING: There are some overflow errors in discretization (at higher end)]")
    if(torch.sum(indices_6d<0) > 0):
        print("[WARNING: There are some overflow errors in discretization (at lower end)]")
    # Discretize the poses
    poses_6d = indices_6d*torch.tensor([cartesian_res,cartesian_res,cartesian_res,angular_res,angular_res,angular_res], dtype=dtype, device=d)
    poses_6d += torch.tensor([(cartesian_res/2)+x_lim[0],(cartesian_res/2)+y_lim[0],(cartesian_res/2)+z_lim[0],(angular_res/2)+r_lim[0],(angular_res/2)+p_lim[0],(angular_res/2)+yaw_lim[0]], dtype=dtype, device=d)
    # Convert to indices in a 2D array
    indices_6d = indices_6d.to(dtype=torch.long)
    indices_6d = indices_6d[:,5]*offsets[5] + indices_6d[:,4]*offsets[4]+ indices_6d[:,3]*offsets[3]+ indices_6d[:,2]*offsets[2]+ indices_6d[:,1]*offsets[1]+ indices_6d[:,0]*offsets[0]
    indices_6d = indices_6d.cpu() # Copy to CPU
    poses_6d = poses_6d.cpu() # Copy to CPU
    del ee_transf_batch
    torch.cuda.empty_cache()

    # TODO: Filter out joint configurations that lead to self-collision and ground collision! (Try pinocchio?)

    J = chain.jacobian(th_batch)
    # Optional: Exclude torso joint jacobian
    # if not use_torso:
    #     J = J[:,:,1:] # Excluding first (torso) joint
    torch.cuda.empty_cache()
    # # Optional: Compute augmented Jacobian with joint limit penalization (based on "Manipulability Analysis" - Vahrenkamp et al 2012")
    # grad = (joint_pos_range_sq*(2*th_batch -joint_pos_max -joint_pos_min).abs()) / ((th_batch-joint_pos_max).pow(2) * (th_batch-joint_pos_max).pow(2))
    # Limit_mult = 1/((1 + grad).sqrt().sqrt()) # Note: penalty may be too harsh. Therefore we are taking an extra square root...
    # J = torch.transpose(Limit_mult.unsqueeze(2)*torch.transpose(J,1,2), 1, 2) # Augmented Jacobian
    # del grad, Limit_mult
    # torch.cuda.empty_cache()
    # Compute the Yoshikawa manipulability measure and copy to CPU
    M = torch.det(J @ torch.transpose(J,1,2)).cpu()
    del J
    torch.cuda.empty_cache()

    # Add computed pose and manipulability to Reachability Map
    reach_map[indices_6d,:6] = poses_6d # Save pose at appropriate index

    # Max Manipulability
    reach_map[indices_6d,-2] += 1 # Increment Visitation Frequency
    reach_map[indices_6d,-1] = np.maximum(reach_map[indices_6d,-1],M) # Add Manipulability IF larger than before (optimistic measure)
    # # Alternatively, we can loop here to get accurate values since many voxels are repeated....
    # for j in range(len(indices_6d)):
    #     index = indices_6d[j]
    #     reach_map[index,-2] += 1 # Increment Visitation Frequency
    #     reach_map[index,-1] = np.maximum(reach_map[index,-1],M[j]) # Add Manipulability IF larger than before (optimistic measure)
    # # Average Manipulability
    # reach_map[indices_6d,-1] *= reach_map[indices_6d,-2] # Accumulate Manipulability. M*Visitation Frequency
    # reach_map[indices_6d,-1] += M # Add the new M
    # reach_map[indices_6d,-2] += 1 # Increment Visitation Frequency
    # reach_map[indices_6d,-1] /= reach_map[indices_6d,-2] # divide to get the average
    # torch.cuda.empty_cache()

    if i%save_freq == 0:
        # Print loop computation time
        t_comp = time.perf_counter() - loop_t0
        print("Loop: " + str(i) + ". Comp Time = {0:.9e}s".format(t_comp))
        
        # Save reachability map to file (as numpy pkl)
        nonzero_rows = torch.abs(reach_map).sum(dim=1) > 0
        reach_map_nonzero = reach_map[nonzero_rows].numpy()

        with open(reach_map_file_path+reach_map_file_name+'.pkl', 'wb') as f:            
            pickle.dump(reach_map_nonzero,f) # Save only non-zero entries
            # pickle.dump(reach_map,f) # Optional: Save full map to add entries to it later
        
        # Accumulate 6D voxel scores into 3D sphere scores (for visualization)
        indx = 0
        first = True
        while(indx < reach_map_nonzero.shape[0]):
            sphere_3d = reach_map_nonzero[indx][:3]
            # Count num_repetitions of current 3D sphere (in the next z_ind_offset subarray)
            num_repetitions = (reach_map_nonzero[indx:indx+z_ind_offset][:,:3] == sphere_3d).all(axis=1).sum().astype(dtype=np.int16)
            # Store sphere and average manipulability as the score. (Also, scale by a factor)
            Manip_scaling = 500
            Manip_avg = reach_map_nonzero[indx:indx+num_repetitions, 7].mean()*Manip_scaling
            if first:
                first = False
                sphere_array = np.append(reach_map_nonzero[indx][:3], Manip_avg)
                # sphere_array = np.append(reach_map_nonzero[indx][:3], num_repetitions) # Optional: Use num_repetitions as score instead
                pose_array = np.append(reach_map_nonzero[0,:6], np.array([0., 0., 0., 1.])).astype(np.single) # dummy value
            else:
                sphere_array = np.vstack((sphere_array, np.append(reach_map_nonzero[indx][:3], Manip_avg)))
                # sphere_array = np.vstack((sphere_array, np.append(reach_map_nonzero[indx][:3], num_repetitions)))  # Optional: Use num_repetitions as score instead
                pose_array = np.vstack((pose_array, np.append(reach_map_nonzero[indx,:6], np.array([0., 0., 0., 1.])).astype(np.single))) # dummy value
            indx += num_repetitions
        # Save 3D map as hdf5 (Mimic reuleux data structure)
        with h5py.File(reach_map_file_path+"3D_"+reach_map_file_name+".h5", 'w') as f:
            sphereGroup = f.create_group('/Spheres')
            sphereDat = sphereGroup.create_dataset('sphere_dataset', data=sphere_array)
            sphereDat.attrs.create('Resolution', data=cartesian_res)
            # (Optional) Save all the 6D poses in each 3D sphere. Currently only dummy pose values (10 dimensional)
            poseGroup = f.create_group('/Poses')
            poseDat = poseGroup.create_dataset('poses_dataset', dtype=float, data=pose_array)


## Post processing: (Optional)
if post_process:
    ## Filter out voxels where end-effector overlaps with robot or ground
    # NOTE: This will not completely filter out all self-collisions from the map since there may still be joint configurations
    # that lead to self-collisions. However, this will still give us a reasonable but optimistic reachability score
    # Remove points that are on or below ground
    reach_map_filtered = reach_map_nonzero[reach_map_nonzero[:,2] > 0.01]  # 1cm above ground
    
    # Add simple self-collision buffer (example values - adjust for your robot)
    base_radius = 0.05  # Approximate base radius
    base_height = 0.05   # Approximate base height
    
    # Filter points near robot base
    base_filter = np.logical_or(
        reach_map_filtered[:,2] > base_height,
        np.linalg.norm(reach_map_filtered[:,:2], axis=1) > base_radius
    )
    reach_map_filtered = reach_map_filtered[base_filter]

# END
t_comp = time.perf_counter() - t0
print("[TOTAL Comp Time] = {0:.2e}s".format(t_comp))

# Debug: Time perf counter
# pdb.set_trace()
# tmat = time.perf_counter()

# t_comp = time.perf_counter() - tmat
# print("Comp Time = {0:.9e}s".format(t_comp))
# pdb.set_trace()
