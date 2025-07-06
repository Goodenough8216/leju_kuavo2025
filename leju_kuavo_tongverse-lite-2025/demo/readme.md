### Demo使用说明
> 注意：DEMO只提供了一些kuavo机器人自带开源控制器`kuavo-ros-control`的一些使用方法来完成部分比赛任务，完整的解决方案需要选手自行斟酌设计和编写实现。
> 想要学习和了解`kuavo-ros-control`控制器的使用，可以参考`demo/kuavo-ros-controldev/docs`目录下的文档。

1. 第一关
- 执行`python demo/solver_task1.py`
- 使用控制器中的键盘控制终端窗口，手动遥控机器人通过障碍物即可

2. 第二关
- 执行`python demo/solver_task2.py`
- 过不平路面和减速带
  - 使用控制器中的键盘控制终端窗口，手动遥控机器人通过障碍物即可
  - 在某些地方容易卡住脚，将速度降下来通过即可(建议20%的速度)
- 上台阶
  - 新开终端, 进入控制器的目录
    ```shell
        cd /TongVerse/biped_challenge/kuavo-ros-controldev
        source devel/setup.bash
        rosrun humanoid_controllers stairClimbPlanner.py
    ```
    - 即可启动爬楼梯程序
  - 或者执行脚本中的`controller.start_stair_climb()`函数，也可以开始走楼梯
- 下斜坡
  - 上完楼梯之后，使用控制器中的键盘控制终端窗口，手动遥控机器人通过
  - 关键点是：下斜坡时，机器人需要控制稍微下蹲的姿势(按k)

3. 第三关
- 执行`python demo/solver_task3.py`
- 新开一个终端，进入控制器的目录
```shell
cd /TongVerse/biped_challenge/demo/kuavo-ros-controldev
source devel/setup.bash
rosrun grab_box grab_box_demo 
```
- 即可启动搬箱子程序



