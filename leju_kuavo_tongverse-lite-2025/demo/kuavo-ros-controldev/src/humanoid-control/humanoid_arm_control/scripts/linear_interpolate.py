def linear_interpolate_time(start, end, num_points):
        """
        在给定的起点和终点之间进行一维线性插值,插值个数由num_points指定
        
        参数:
        start (float): 起点值
        end (float): 终点值
        num_points (int): 插值个数
        
        返回:
        list: 包含起点、终点和插值点的列表
        """
        
        # 检查参数有效性
        if not isinstance(start, (int, float)):
            raise ValueError("起点值必须是整数或浮点数")
        if not isinstance(end, (int, float)):
            raise ValueError("终点值必须是整数或浮点数")
        if not isinstance(num_points, int) or num_points < 2:
            raise ValueError("插值个数必须是大于等于2的整数")
        
        # 计算步长
        step = (end - start) / (num_points - 1)
        
        # 初始化插值点列表
        points = [start]
        
        # 计算插值点并添加到列表中
        for i in range(1, num_points):
            point = start + i * step
            points.append(point)
        
        # 添加终点
        # points.append(end)
        return points

def linear_interpolate_joint(start_joints, end_joints, num_points):
    # print("joint_value: ", start_joints)
    """
    对一个14维的关节数组在起点和终点之间进行插值
    
    参数:
    start_joints (list): 起点关节值,长度为14
    end_joints (list): 终点关节值,长度为14
    num_points (int): 插值个数
    
    返回:
    list: 二维数组,每一行表示一次插值的所有关节值
    """
    
    # 检查参数有效性
    if len(start_joints) != 14:
        raise ValueError("起点关节值必须是长度为14的列表")
    if len(end_joints) != 14:
        raise ValueError("终点关节值必须是长度为14的列表")
    if num_points < 2:
        raise ValueError("插值个数必须是大于等于2的整数")
    
    # 初始化结果列表
    result = []
    
    # 对每一个关节进行插值
    for start, end in zip(start_joints, end_joints):
        # 计算步长
        step = (end - start) / (num_points - 1)
        # 初始化插值点列表
        points = [start]
        # 计算插值点并添加到列表中
        for i in range(1, num_points):
            point = start + i * step
            points.append(point)
        # 添加终点
        # points.append(end)
        # 将插值点列表作为一行添加到结果列表中
        result.append(points)
    
    # 将结果列表转置,每一行表示一次插值的所有关节值
    result = list(map(list, zip(*result)))
    return result