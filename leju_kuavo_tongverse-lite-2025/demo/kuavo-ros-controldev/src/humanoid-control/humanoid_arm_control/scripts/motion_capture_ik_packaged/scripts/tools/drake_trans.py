import numpy as np
import quaternion
import tf
from pydrake.all import (
    RotationMatrix,
    Quaternion,
    RollPitchYaw,
)

from scipy.spatial.transform import Rotation as R

def quaternion_to_RPY(quat):
    rpy = RollPitchYaw(RotationMatrix(Quaternion(quat[3], quat[0], quat[1], quat[2])))
    return rpy.vector()

def RPY_to_quaternion(rpy):
    q = RollPitchYaw(rpy[0], rpy[1], rpy[2]).ToQuaternion()
    return [q.x(), q.y(), q.z(), q.w()]

def quaternion_to_matrix(quat):
    q1 = np.quaternion(quat[3], quat[0], quat[1], quat[2])
    q1 = q1.normalized()
    R = RotationMatrix(Quaternion(q1.w, q1.x, q1.y, q1.z))
    return R.matrix()

def matrix_to_quaternion(mat):
    mat = np.asarray(mat, dtype=np.float64)
    quat = Quaternion(mat)
    quat_vec = [quat.x(), quat.y(), quat.z(), quat.w()]
    return quat_vec

def matrix_to_rpy(mat):
    return RollPitchYaw(mat).vector()

def rpy_to_matrix(rpy):
    R = RollPitchYaw(rpy).ToRotationMatrix()
    return R.matrix()

def pos_rpy_to_transform(pos, rpy):
    """Convert position and rpy to a 4x4 transformation matrix."""
    transform = np.eye(4)
    transform[:3, :3] = rpy_to_matrix(rpy)
    transform[:3, 3] = pos
    return transform

def transform_to_pos_rpy(transform):
    """Convert a 4x4 transformation matrix to position and rpy."""
    pos = transform[:3, 3]
    rpy = matrix_to_rpy(transform[:3, :3])
    return pos, rpy

def transform_inverse(transform):
    """Compute the inverse of a 4x4 transformation matrix."""
    R_inv = transform[:3, :3].T
    p_inv = -R_inv @ transform[:3, 3]
    transform_inv = np.eye(4)
    transform_inv[:3, :3] = R_inv
    transform_inv[:3, 3] = p_inv
    return transform_inv

def LHS_to_RHS(p_LHS):
    """Transform the position of T_LHS to the position of T_RHS."""
    """ Only for z-axis!!! """
    S = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
    T_L2R = np.eye(4)
    T_L2R[:3, :3] = S
    T_result = T_L2R @ p_LHS
    # T_result[:3, :3] = S @ p_LHS[:3, :3] @ S
    T_result[:3, :3] = T_result[:3, :3] @ S # R_r = S * R_l @ S
    return T_result

def head_to_robot_transform(T_HB_B):
    """Compute the transform from head to robot."""
    # T_HW_W = np.eye(4)
    # R_BW_W = rpy_to_matrix([-np.pi/2.0, np.pi/2.0, 0.0])
    # R_HW_W = np.transpose(R_BW_W) @ T_HB_B[:3, :3] @ R_BW_W
    # T_HW_W[:3, :3] = R_HW_W
    # p_HW = np.transpose(R_BW_W) @ T_HB_B[:3, 3]
    # T_HW_W[:3, 3] = p_HW
    # return T_HW_W
    T_BW_W = np.eye(4)
    T_BW_W[:3, :3] = rpy_to_matrix([-np.pi/2.0, np.pi/2.0, 0.0])
    T_HW_W = transform_inverse(T_BW_W) @ T_HB_B @ T_BW_W
    return T_HW_W

def pos_rpy_LHS_to_RHS(pos_LHS, rpy_LHS, is_reverse=False):
    """Transform the position and rpy of T_LHS to the position and rpy of T_RHS."""
    """ Only for z-axis!!! """
    rpy_RHS = [rpy_LHS[0], rpy_LHS[1], -rpy_LHS[2]]
    pos_RHS = [pos_LHS[0], pos_LHS[1], -pos_LHS[2]]
    rpy_RHS[1] += np.pi if is_reverse else 0.0
    return np.array(pos_RHS), np.array(rpy_RHS)

def vr_quat2robot_quat(vr_quat, side, bias_agl=20*np.pi/180):
    q1 = np.quaternion(vr_quat[3], vr_quat[0], vr_quat[1], vr_quat[2])
    q1 = q1.normalized()
    # print(f"q1: {q1}")
    # print(f"q1: {np.asarray(q1)}")
    mat1 = quaternion_to_matrix([q1.x, q1.y, q1.z, q1.w])
    R_bias = rpy_to_matrix([-bias_agl, 0, 0])
    R12 = rpy_to_matrix([-np.pi/2, 0.0, -np.pi/1])
    # R12 = rpy_to_matrix([0,0,0])
    if side == "Right": #right hand
        # R12 = rpy_to_matrix([np.pi/2, np.pi/2, -np.pi])
        R_bias = rpy_to_matrix([bias_agl, 0, 0])
        R12 = rpy_to_matrix([-np.pi/2, 0.0, 0.0])
    mat2 = mat1 @ R12 @ R_bias
    # mat2 = R21.T @ mat1 @ R21
    # mat2 = mat1 @ R21
    quat = matrix_to_quaternion(mat2)
    return quat

def rpy_to_quaternion(roll, pitch, yaw):
    """
    Converts roll-pitch-yaw angles to a quaternion.

    Args:
    - roll: Roll angle in radians (float).
    - pitch: Pitch angle in radians (float).
    - yaw: Yaw angle in radians (float).

    Returns:
    - Quaternion as a list [x, y, z, w].
    """
    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    quaternion = [quaternion[0], quaternion[1], quaternion[2], quaternion[3]]
    return quaternion

def original_LHS_pos_rpy_to_RHS_in_robot_frame(pose_LHS, head_pose_LHS, is_reverse=False):
    p1_LHS_world, p1_rpy_LHS_world = pose_LHS
    head_LHS_world, head_rpy_LHS_world = head_pose_LHS
    p1_RHS_world, p1_rpy_RHS_world = pos_rpy_LHS_to_RHS(p1_LHS_world, p1_rpy_LHS_world, is_reverse)
    # print(f"p1_RHS_world: {p1_RHS_world}, p1_rpy_RHS_world: {rad2deg * p1_rpy_RHS_world}")
    p1_RHS_world_transform = pos_rpy_to_transform(p1_RHS_world, p1_rpy_RHS_world)
    T_HW_R = head_to_robot_transform(p1_RHS_world_transform)
    pos, rpy = transform_to_pos_rpy(T_HW_R)
    # print(f"T_HW_R pos: {pos}, rpy: {rad2deg * rpy}")

    head_RHS_world, head_rpy_RHS_world = pos_rpy_LHS_to_RHS(head_LHS_world, head_rpy_LHS_world)
    # print(f"head_RHS_world: {head_RHS_world}, head_rpy_RHS_world: {rad2deg * head_rpy_RHS_world}")
    head_RHS_world_transform = pos_rpy_to_transform(head_RHS_world, head_rpy_RHS_world)
    T_BW_R = head_to_robot_transform(head_RHS_world_transform)
    pos, rpy = transform_to_pos_rpy(T_BW_R)
    # print(f"T_BW_R pos: {pos}, rpy: {rad2deg * rpy}")

    # T_HB_R = T_HW_R @ transform_inverse(T_BW_R)
    T_HB_R = np.eye(4)
    T_HB_R[:3, :3] = T_HW_R[:3, :3] @ np.transpose(T_BW_R[:3, :3])
    T_HB_R[:3, 3] = T_HW_R[:3, 3] - T_BW_R[:3, 3]
    pos, rpy = transform_to_pos_rpy(T_HB_R)
    return pos, rpy

def compute_rpy_error(rpy1, rpy2):
    rot_matrix1 = rpy_to_matrix(rpy1)
    rot_matrix2 = rpy_to_matrix(rpy2)
    error_rot_matrix = np.dot(np.transpose(rot_matrix1), rot_matrix2)
    error_rpy = RollPitchYaw(error_rot_matrix).vector()
    return error_rpy


def matrix_to_axis_angle(Rot):
    rot = R.from_matrix(Rot)
    # 转换为轴角表示
    axis_angle = rot.as_rotvec()
    axis = axis_angle[:3]
    angle = np.linalg.norm(axis_angle)
    return axis, angle

def axis_angle_to_matrix(rotvec):
    # 使用旋转向量创建 Rotation 对象
    rot = R.from_rotvec(rotvec)
    # 转换为旋转矩阵
    R_matrix = rot.as_matrix()
    return R_matrix


if __name__ == '__main__':    
    # quat1 = [0.65, 0.75, -0.02, -0.015]
    # quat2 = [0.47, -0.46, -0.53, 0.517]
    quat1 = [-0.08, -0.04, -0.77, -0.61]
    quat2 = [0.3959, 0.5, -0.57, -0.5]
    mat1 = quaternion_to_matrix(quat1)
    mat2 = quaternion_to_matrix(quat2)
    R = mat1.T @ mat2
    error_rpy = RollPitchYaw(R).vector()
    print(f"error_rpy: {error_rpy}")
    rpy2 = RollPitchYaw(R.T).vector()
    print(f"rpy2: {rpy2}")
