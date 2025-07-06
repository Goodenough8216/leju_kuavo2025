import signal
import argparse
import multiprocessing
from multiprocessing import Process


from hand_controller_node import run_node as hand_controller_run_node
from arm_controller_node import run_node as arm_controller_run_node

reverse = 0


def signal_handler(sig):
    current_process = multiprocessing.current_process()
    print(
        f"Received signal {sig} in process {current_process.name}, exiting..."
    )
    multiprocessing.active_children()
    for process in multiprocessing.active_children():
        process.terminate()
    exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Hand Controller Node")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode"
    )
    parser.add_argument(
        "--side",
        type=str,
        default="all",
        help="Side of the hand controller",
    )
    parser.add_argument(
        "--pose",
        type=str,
        default="all",
        help="Pose of the hand controller",
    )
    parser.add_argument(
        "--hand_tracking",
        type=bool,
        default=False,
        help="Enable hand tracking",
    )
    args = parser.parse_args()
    args.reverse = reverse
    return args


def ask_reverse_input():
    while True:
        global reverse
        try:
            reverse_noitom_hi5_hand_ori = int(
                input(
                    "请问校正完手套设备后手掌模型是否朝向外（与我们的手的方向一致）一致请输入：1， 相反请输入：2 : "
                )
            )
            if reverse_noitom_hi5_hand_ori == 1:
                reverse = 0
                break
            elif reverse_noitom_hi5_hand_ori == 2:
                reverse = 180
                break
            else:
                print("Invalid input, please input 1 or 2.")
                continue
        except ValueError:
            print("Invalid input, please input 1 or 2.")
            continue

    return reverse


def start_nodes(args):
    node_configs = []
    hand_controller_node_config = {
        "target": hand_controller_run_node,
        "kwargs": {
            "config_file_path": "config/left_right_z_axis.json",
            "debug": args.debug,
        },
        "name": "hand_controller_node",
    }

    node_configs.append(hand_controller_node_config)

    arm_controller_node_config = {
        "target": arm_controller_run_node,
        "kwargs": {
            "reverse": args.reverse,
            "debug": args.debug,
            "side": args.side,
            "pose": args.pose,
            "hand_tracking": args.hand_tracking,
            "config_file_path": "config/robot_arm_config.json",
        },
        "name": "arm_controller_node",
    }

    node_configs.append(arm_controller_node_config)

    node_process = []
    for node_config in node_configs:
        node = Process(
            target=node_config["target"],
            kwargs=node_config["kwargs"],
            name=node_config["name"],
        )
        node.start()
        node_process.append(node)

    for node in node_process:
        node.join()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    reverse = ask_reverse_input()
    args = parse_args()
    start_nodes(args)
