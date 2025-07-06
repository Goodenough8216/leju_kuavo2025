import rospy
import json

# 搞清楚这是用来做什么的
INIT_ARM_POS = [20, 0, 0, -30, 0, 0, 0, 20, 0, 0, -30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

""" 
    get_q_from_action_file:
    从kuavo app转出的动作数据中提取关节信息
    @ param action_list: 动作帧数据列表
    @ return
    
    用于给IK提供初始位姿
"""
def get_q_from_action_file(action_list) -> list:
    arm_torque_list = []
    for i in action_list:
        end_point_i = action_list[i][-1][0][1]
        arm_torque_list.append(end_point_i)
    return arm_torque_list


"""
    copy_data_from_end_point:
    根据最后一维的数据，构造一个时间延后gap_time，但数值不变的贝塞尔曲线点
"""
def copy_data_from_end_point(action_list_i: list, gap_time: int) -> list:
    last_curve_point = action_list_i[-1]
    # 时间增加
    for j in range(len(last_curve_point)):
        last_curve_point[j][0] = last_curve_point[j][0] + gap_time
    return last_curve_point


"""
    get_curve_from_points:
    根据生成的数据结构构造点，单个关节数据的构造
"""
def get_points_from_action_data(action_data_i):
    result = []
    point = []
    # print(len(action_data_i))
    for i in range(len(action_data_i)-1):
        point.append(action_data_i[i][0])   # 端点
        point.append(action_data_i[i][2])   # 右控制点
        point.append(action_data_i[i+1][1]) # 左控制点
        point.append(action_data_i[i+1][0])
        # print(point)
        result.append(point)
        point = []
    return result


def load_json_file(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except IOError as e:
        rospy.logerr(f"Error reading file {file_path}: {e}")
        return None

# 注意这里，加载时会自己加上一帧初始位置
def frames_to_custom_action_data(frames):
    action_data = {}
    for frame in frames:
        servos, keyframe, attribute = frame["servos"], frame["keyframe"], frame["attribute"]
        for index, value in enumerate(servos):
            key = index + 1
            if key not in action_data:
                action_data[key] = []
                if keyframe != 0 and len(action_data[key]) == 0:
                    if key <= len(INIT_ARM_POS):
                        action_data[key].append([
                            [0, INIT_ARM_POS[key-1]],
                            [0, 0],
                            [0, INIT_ARM_POS[key-1]],
                ])
            if value is not None:
                CP = attribute[str(key)]["CP"]
                left_CP, right_CP = CP
                action_data[key].append([
                    [round(keyframe/50, 1), value],
                    [round((keyframe+left_CP[0])/50, 1), value+left_CP[1]],
                    [round((keyframe+right_CP[0])/50, 1), value+right_CP[1]],
                ])
    return action_data