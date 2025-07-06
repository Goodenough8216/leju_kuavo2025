import fcl
import numpy as np


class CollisionDetector:
    def __init__(self, base_pos, base_xyz, check_point_radius):
        """
        躯干用长方体近似，检查点用球体近似
        """
        self.__base_pos = base_pos
        self.__base_xyz = base_xyz

        self.__base_box = fcl.Box(0.8, 1.0, 1.0)


def collision_detector(base, check_point):
    """
    parameters:
    base: 3D point of the base of the robot: box
    check_point: 检查该点是否与机器人发生碰撞: sphere
    """
    