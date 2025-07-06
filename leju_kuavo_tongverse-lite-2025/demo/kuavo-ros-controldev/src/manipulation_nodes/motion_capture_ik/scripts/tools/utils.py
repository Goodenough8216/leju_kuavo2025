import rospkg
from enum import Enum
import numpy as np

def get_package_path(package_name):
    try:
        rospack = rospkg.RosPack()
        package_path = rospack.get_path(package_name)
        return package_path
    except rospkg.ResourceNotFound:
        return None

def rotation_matrix_diff_in_angle_axis(R1, R2):
    R_rel = np.dot(np.transpose(R1), R2)

    # 计算旋转角度
    theta = np.arccos((np.trace(R_rel) - 1) / 2)
    if np.isnan(theta):
        theta = 0.0

    # 计算旋转轴
    x = (R_rel[2, 1] - R_rel[1, 2]) / (2 * np.sin(theta)+1e-8)
    y = (R_rel[0, 2] - R_rel[2, 0]) / (2 * np.sin(theta)+1e-8)
    z = (R_rel[1, 0] - R_rel[0, 1]) / (2 * np.sin(theta)+1e-8)

    axis = np.array([x, y, z])

    return theta, axis

def limit_value(value, min_value, max_value):
    """Limit the value between min_value and max_value."""
    if value < min_value:
        return min_value
    elif value > max_value:
        return max_value
    else:
        return value

class ArmIdx(Enum):
    LEFT = 0
    RIGHT = 1
    BOTH = 2

    def name(self):
        if self == ArmIdx.LEFT:
            return "LEFT"
        elif self == ArmIdx.RIGHT:
            return "RIGHT"
        elif self == ArmIdx.BOTH:
            return "BOTH"


class IkTypeIdx(Enum):
    TorsoIK = 0
    DiffIK = 1

    def name(self):
        if self == IkTypeIdx.TorsoIK:
            return "TorsoIK"
        elif self == IkTypeIdx.DiffIK:
            return "DiffIK"

if __name__ == '__main__':
    package_name = 'motion_capture_ik'
    path = get_package_path(package_name)

    if path:
        print(f"Package '{package_name}' is located at: {path}")
    else:
        print(f"Package '{package_name}' not found")
    
    # Test ArmIdx
    Name = ArmIdx.LEFT.name()
    print("NAME: ", Name)
