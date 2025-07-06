from SimpleSDK import WHJ30Tools
import time,sys

DEV_ID = 0x02
positionA = 20
positionB = -20
step = 9


if __name__ == "__main__":
    whj30 = WHJ30Tools()
    whj30.open_canbus()
    print(whj30.skip_iap_update(DEV_ID))
    time.sleep(2)
    whj30.disable_joint(DEV_ID)
    time.sleep(3)
    
    # whj30.set_joint_min_position(DEV_ID, -200)
    # whj30.set_joint_max_position(DEV_ID, 200)
    # whj30.save_joint_param(DEV_ID)
    # time.sleep(1)
    
    whj30.enable_joint(DEV_ID)
    time.sleep(3)

    result_list = whj30.get_joint_state(DEV_ID)
    if(result_list):
        print(result_list)
        now_position = result_list[5]
    else:
        print("ID {} get state ERROR".format(DEV_ID))
        sys.exit()


    whj30.set_joint_operate_mode(DEV_ID, 0x03)


    # 获取当前位置
    result_list = whj30.get_joint_state(DEV_ID)
    now_position = result_list[5]

    # 从当前位置运动到50度，每次只能运动10度
    while now_position < positionA-5:
        # 将当前位置增加10度，不超过50度
        next_position = min(now_position + step, positionA)
        whj30.set_joint_position(DEV_ID, next_position)
        time.sleep(1)
        result_list = whj30.get_joint_state(DEV_ID)
        now_position = result_list[5]
        print("now: {:.2f} to: {:.2f}".format(now_position, next_position))

    # 从50度运动到-50度，每次只能运动10度
    while now_position > positionB+5:
        # 将当前位置减少10度，不小于-50度
        next_position = max(now_position - step, positionB)
        whj30.set_joint_position(DEV_ID, next_position)
        time.sleep(1)
        result_list = whj30.get_joint_state(DEV_ID)
        now_position = result_list[5]
        print("now: {:.2f} to: {:.2f}".format(now_position, next_position))




    whj30.clear_joint_error(DEV_ID)
    whj30.get_joint_state(DEV_ID)
    whj30.enable_joint(DEV_ID)
    time.sleep(2)

    whj30.close_canbus()