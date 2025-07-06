import rospy
import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 将项目根目录添加到 Python 解释器的搜索路径中
sys.path.append(project_root)

from h12pro_node.node import H12PROControllerNode


def parse_args():
    parser = argparse.ArgumentParser(description="h12pro_node")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Enable real mode",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    rospy.init_node("h12pro_controller_node")
    h12pro_controller_node = H12PROControllerNode(real=args.real)
    rospy.spin()
