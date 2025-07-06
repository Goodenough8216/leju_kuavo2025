#!/usr/bin/env python3

import rospy
import tf2_ros
import geometry_msgs.msg
from tf.transformations import quaternion_matrix, translation_matrix, concatenate_matrices, inverse_matrix, translation_from_matrix, quaternion_from_matrix
import yaml
import os
import numpy as np

def from_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        in_dict = yaml.load(f, Loader=yaml.FullLoader)
    return from_dict(in_dict)

def from_dict(in_dict):
    """
    Sets values parsed from a given dictionary.

    :param in_dict: input dictionary.
    :type in_dict: dict[string, string|dict[string,float]]

    :rtype: None
    """
    param = in_dict['parameters']
    tr = in_dict['transformation']
    return param, tr

def matrix_from_transform(transform):
    """Convert a transform message to a 4x4 matrix."""
    trans = transform.transform.translation
    rot = transform.transform.rotation
    trans_matrix = translation_matrix([trans.x, trans.y, trans.z])
    rot_matrix = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
    return concatenate_matrices(trans_matrix, rot_matrix)

def transform_from_matrix(matrix, parent_frame, child_frame, stamp=None):
    """Convert a 4x4 matrix to a transform message."""
    if stamp is None:
        stamp = rospy.Time.now()
    
    transform = geometry_msgs.msg.TransformStamped()
    transform.header.stamp = stamp
    transform.header.frame_id = parent_frame
    transform.child_frame_id = child_frame
    
    translation = translation_from_matrix(matrix)
    quaternion = quaternion_from_matrix(matrix)
    
    transform.transform.translation.x = translation[0]
    transform.transform.translation.y = translation[1]
    transform.transform.translation.z = translation[2]
    
    transform.transform.rotation.x = quaternion[0]
    transform.transform.rotation.y = quaternion[1]
    transform.transform.rotation.z = quaternion[2]
    transform.transform.rotation.w = quaternion[3]
    
    return transform

if __name__ == '__main__':
    rospy.init_node('publish_calibration_tf')
    
    cali_file = rospy.get_param('~calibration_file', '~/.ros/easy_handeye/humanoid_eye_on_base.yaml')
    cali_file = os.path.expanduser(cali_file)

    try:
        param, tr = from_yaml(cali_file)
        
        print(param)
        print(tr)
        
        # create a tf listener to get the transform between camera_color_optical_frame and camera_link
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        
        # wait for the transform between camera_color_optical_frame and camera_link to be available
        try:
            rospy.loginfo("Waiting for transform between camera_color_optical_frame and camera_link...")
            optical_to_link = tfBuffer.lookup_transform('camera_color_optical_frame', 'camera_link', rospy.Time(0), rospy.Duration(5.0))
            rospy.loginfo("Got transform from camera_color_optical_frame to camera_link")
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Failed to lookup transform: {e}")
            exit(1)
        
        # convert the calibration transform (robot_base_frame to camera_color_optical_frame) to a matrix
        base_to_optical = np.identity(4)
        
        # fill the transform matrix in the calibration
        base_to_optical[:3, 3] = [tr["x"], tr["y"], tr["z"]]
        quat_matrix = quaternion_matrix([tr["qx"], tr["qy"], tr["qz"], tr["qw"]])
        base_to_optical[:3, :3] = quat_matrix[:3, :3]
        
        # convert the optical_to_link transform to a matrix
        optical_to_link_matrix = matrix_from_transform(optical_to_link)
        
        # calculate the transform between robot_base_frame and camera_link
        base_to_link = np.dot(base_to_optical, optical_to_link_matrix)
        
        # create and fill the new TransformStamped message
        static_transform = transform_from_matrix(
            base_to_link, 
            param['robot_base_frame'], 
            'camera_link'
        )
        
        # create a static transform broadcaster
        broadcaster = tf2_ros.StaticTransformBroadcaster()
        
        # publish the static transform
        broadcaster.sendTransform(static_transform)
        rospy.loginfo(f"Publishing static transform from {param['robot_base_frame']} to camera_link")
        
        rospy.spin()
        
    except Exception as e:
        rospy.logerr(f"failed to publish calibration tf: {e}")