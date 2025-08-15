#!/usr/bin/env python3
import rospy
import h5py
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, Vector3
from std_msgs.msg import ColorRGBA, Header

def create_reachability_marker(spheres, resolution):
    marker = Marker()
    marker.header.frame_id = "joint1"  # Mycobot base frame
    marker.header.stamp = rospy.Time.now()
    marker.ns = "reachability"
    marker.id = 0
    marker.type = Marker.CUBE_LIST
    marker.action = Marker.ADD
    marker.scale = Vector3(resolution, resolution, resolution)
    marker.lifetime = rospy.Duration(0)  # Persistent marker
    
    # Downsample for better performance
    if len(spheres) > 10000:
        spheres = spheres[np.random.choice(len(spheres), 10000, replace=False)]
    
    for sphere in spheres:
        pos, reachability = sphere[:3], sphere[3]
        marker.points.append(Point(*pos))
        
        # Color gradient: red (0%) -> yellow (50%) -> green (100%)
        r = min(2.0 * (1 - reachability/100), 1.0)
        g = min(2.0 * (reachability/100), 1.0)
        color = ColorRGBA(r, g, 0, 0.6)  # Semi-transparent
        marker.colors.append(color)
    
    return marker

def create_pose_markers(poses):
    marker_array = MarkerArray()
    # Show only every 10th pose for performance
    for i, pose in enumerate(poses[::10]):
        marker = Marker()
        marker.header.frame_id = "joint1"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "poses"
        marker.id = i
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position = Point(*pose[:3])
        marker.pose.orientation.x = pose[3]
        marker.pose.orientation.y = pose[4]
        marker.pose.orientation.z = pose[5]
        marker.pose.orientation.w = pose[6]
        marker.scale = Vector3(0.01, 0.002, 0.002)
        marker.color = ColorRGBA(0, 0.5, 1, 0.8)  # Light blue
        marker_array.markers.append(marker)
    return marker_array

def main():
    rospy.init_node("reachability_visualizer")
    
    # Get parameters
    map_path = rospy.get_param("~map_path", "")
    show_poses = rospy.get_param("~show_poses", False)
    
    # Load map data
    try:
        with h5py.File(map_path, 'r') as f:
            spheres = f['/Spheres/sphere_dataset'][:]
            poses = f['/Poses/pose_dataset'][:] if show_poses else None
            resolution = f['/Spheres/sphere_dataset'].attrs['Resolution']
    except Exception as e:
        rospy.logerr(f"Failed to load map: {str(e)}")
        return
    
    # Setup publishers
    voxel_pub = rospy.Publisher("/reachability/voxels", Marker, queue_size=1, latch=True)
    poses_pub = rospy.Publisher("/reachability/poses", MarkerArray, queue_size=1, latch=True)
    
    # Create markers
    voxel_marker = create_reachability_marker(spheres, resolution)
    voxel_pub.publish(voxel_marker)
    
    if show_poses and poses is not None:
        pose_markers = create_pose_markers(poses)
        poses_pub.publish(pose_markers)
    
    rospy.loginfo("Reachability visualization published")
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass