import numpy as np
import matplotlib.pyplot as plt


""" CubicBezier 贝塞尔曲线基础类

    包含插值，计算控制点等贝塞尔曲线基础操作
"""
class CubicBezier:
    def __init__(self):
        pass
    
    def _cubic_bezier_derivative(self, p0, p1, p2, p3, t):
        # 计算三阶贝塞尔曲线的一阶导数
        dx = -3 * (1 - t)**2 * (p0[0] - p1[0]) + 6 * (1 - t) * t * (p1[0] - 2 * p2[0] + p3[0]) + 3 * t**2 * (p3[0] - p2[0])
        dy = -3 * (1 - t)**2 * (p0[1] - p1[1]) + 6 * (1 - t) * t * (p1[1] - 2 * p2[1] + p3[1]) + 3 * t**2 * (p3[1] - p2[1])
        return np.array([dx, dy])

    def _cubic_bezier_second_derivative(self, p0, p1, p2, p3, t):
        # 计算三阶贝塞尔曲线的二阶导数
        dx = 6 * (1 - t) * (p0[0] - 2 * p1[0] + p2[0]) + 6 * t * (p3[0] - 2 * p2[0] + p1[0])
        dy = 6 * (1 - t) * (p0[1] - 2 * p1[1] + p2[1]) + 6 * t * (p3[1] - 2 * p2[1] + p1[1])
        return np.array([dx, dy])
    
    def calc_cubic_bezier_ctrl_point(self, p11, p12, p13, p14, start, end):
        # 1. 计算导数
        derivative_1 = self._cubic_bezier_derivative(p11, p12, p13, p14, 1)  # 起始点处的一阶导数
        derivative_2 = self._cubic_bezier_second_derivative(p11, p12, p13, p14, 1)  # 起始点处的一阶导数
        
        # 2. 求解 ctrl_point1
        ctrl_point1 = start + derivative_1 / 3

        # 3. 求解 ctrl_point2
        ctrl_point2 = 2 * ctrl_point1 - start + (1/6) * derivative_2
        
        return np.array([ctrl_point1, ctrl_point2])
        
    # 给定控制点，计算贝塞尔曲线上的点
    def _cubic_bezier(self, p0, p1, p2, p3, t):
        return (1-t)**3 * np.array(p0) + 3 * (1-t)**2 * t * np.array(p1) + 3 * (1-t) * t**2 * np.array(p2) + t**3 * np.array(p3)
    
    def show_cubic_bezier_line(self, points, num=100):
        # 生成t值
        t_values = np.linspace(0, 1, num)

        # 计算第一条曲线上的点
        i = 0
        for point in points:
            i = i + 1
            curve_points = [self._cubic_bezier(point[0], point[1], point[2], point[3], t) for t in t_values]
            plt.plot(*zip(*curve_points), 'r-', label='Bezier Curve {}'.format(i))
            sizes = np.abs(np.array([10, 20, 30, 40]))  # 使用绝对值确保大小为正
            # print(point)
            plt.scatter(point[0][0], point[0][1])
            plt.scatter(point[1][0], point[1][1])
            plt.scatter(point[2][0], point[2][1])
            plt.scatter(point[3][0], point[3][1])
        plt.legend()
        plt.show()
    
    """
        interpolate_cubic_bezier: 给定四个点，计算贝塞尔曲线上的插值点
        @ params:
            p0, p1, p2, p3: 四个点坐标
    """
    def interpolate_cubic_bezier(self, p0, p1, p2, p3, num_points=100):
        try:
            # 计算第一条曲线上的点
            points = []
            for t in np.linspace(0, 1, num_points+1):
                # 计算贝塞尔曲线的参数方程
                xt = ((1 - t)**3) * p0[0] + 3 * ((1 - t)**2) * t * p1[0] + 3 * (1 - t) * (t**2) * p2[0] + (t**3) * p3[0]
                yt = ((1 - t)**3) * p0[1] + 3 * ((1 - t)**2) * t * p1[1] + 3 * (1 - t) * (t**2) * p2[1] + (t**3) * p3[1]
                points.append([xt, yt])
            return points[1:]
        except Exception as e:
            print("error eccur in interpolate", e)
        
if __name__ == '__main__':
    # 创建 CubicBezier 实例
    cubic_bezier = CubicBezier()
    
    # 定义两组贝塞尔曲线的四个控制点
    p0_1, p1_1, p2_1, p3_1 = [0, 0], [0, 1], [1, 1], [1, 0]
    p0_2, p3_2 = [2, 0], [3, 0]
    
    # 计算新的控制点
    ctrl_points = cubic_bezier.calc_cubic_bezier_ctrl_point(
        p11=p0_1, p12=p1_1, p13=p2_1, p14=p3_1,
        start=p0_2, end=p3_2
    )
    
    # 显示贝塞尔曲线
    cubic_bezier.show_cubic_bezier_line([
        [p0_1, p1_1, p2_1, p3_1],
        [p0_2, ctrl_points[0], ctrl_points[1], p3_2]
    ])