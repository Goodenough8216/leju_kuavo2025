from SimpleSDK import WHJ30Tools
import time,sys

DEV_ID1 = 0x01
DEV_ID2 = 0x02
step = 9


def canbus_open():
    whj30 = WHJ30Tools()
    result = whj30.open_canbus()
    time.sleep(1)
    return result,whj30


def ruierman_position_set(id,can30,positionA,positionB):
    print(id,can30.skip_iap_update(id))
    time.sleep(2)
    result_list = can30.get_joint_state(id)
    if(result_list):
        print(result_list)
        now_position = result_list[5]
    else:
        print("ID {} get state ERROR".format(id))
        return
    
    can30.disable_joint(id)
    time.sleep(3)
    
    # can30.set_joint_min_position(id, -200)
    # can30.set_joint_max_position(id, 200)
    # can30.save_joint_param(id)
    # time.sleep(1)
    
    can30.enable_joint(id)
    time.sleep(3)



    can30.set_joint_operate_mode(id, 0x03)


    # 获取当前位置
    result_list = can30.get_joint_state(id)
    now_position = result_list[5]

    # 从当前位置运动到50度，每次只能运动10度
    while now_position < positionA-5:
        # 将当前位置增加10度，不超过50度
        next_position = min(now_position + step, positionA)
        can30.set_joint_position(id, next_position)
        time.sleep(1)
        result_list = can30.get_joint_state(id)
        now_position = result_list[5]

        now_str = "{:.2f}".format(now_position)
        next_str = "{:.2f}".format(next_position)
        print("now: {} to: {}".format(now_str.rjust(8), next_str.rjust(8)))

    # 从50度运动到-50度，每次只能运动10度
    while now_position > positionB+5:
        # 将当前位置减少10度，不小于-50度
        next_position = max(now_position - step, positionB)
        can30.set_joint_position(id, next_position)
        time.sleep(1)
        result_list = can30.get_joint_state(id)
        now_position = result_list[5]
        
        now_str = "{:.2f}".format(now_position)
        next_str = "{:.2f}".format(next_position)
        print("now: {} to: {}".format(now_str.rjust(8), next_str.rjust(8)))




    can30.clear_joint_error(id)
    can30.get_joint_state(id)
    can30.disable_joint(id)


def ruierman_mov():
    print("注意：请在设置零点之后运行程序")
    result,canBus = canbus_open()
    ruierman_position_set(DEV_ID1,canBus,20,-5)
    time.sleep(2)
    ruierman_position_set(DEV_ID2,canBus,5,-20)
    time.sleep(2)
    canBus.close_canbus()


# 新ID  旧ID
def ruierman_setId(n_id,o_id=3):
    # 创建WHJ30Tools对象实例
    whj30 = WHJ30Tools()

    # 开启CAN总线通讯
    whj30.open_canbus()
    new_id = n_id
    old_id = o_id
    print(new_id,old_id)
    # 跳过IAP更新操作，传入关节ID id
    whj30.skip_iap_update(old_id)


    # 失能关节，传入关节ID id
    whj30.disable_joint(old_id)

    # 使能关节，传入关节ID id
    whj30.enable_joint(old_id)
    time.sleep(4)

    whj30.set_joint_id(old_id,new_id)
    whj30.save_joint_param(old_id)
    # whj30.set_joint_zero_position(id)
    # whj30.save_joint_param(id)
    
    time.sleep(1)


    whj30.close_canbus()


def ruierman_setZero():
    # 创建WHJ30Tools对象实例
    whj30 = WHJ30Tools()

    # 开启CAN总线通讯
    whj30.open_canbus()
    joint_address_list = [0x01, 0x02]
    
    for address in joint_address_list:
        whj30.skip_iap_update(address)
        whj30.disable_joint(address)
        whj30.enable_joint(address)
        print(address)
    time.sleep(4)

    for address in joint_address_list:
        whj30.set_joint_zero_position(address)
        whj30.save_joint_param(address)
        print(whj30.get_joint_state(id))

    # 关闭CAN总线通讯
    whj30.close_canbus()


if __name__ == "__main__":
    # ruierman_mov()
    ruierman_setId(3,1)