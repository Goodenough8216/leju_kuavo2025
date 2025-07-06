文件描述:
- `load_kuavo_mujoco_sim.launch`: mujoco仿真器的launch文件，加载了kuavo的模型和控制器。
- `load_kuavo_real.launch`: 实物控制的launch文件
- `robot_version_manager.launch`: 版本管理的launch文件，被其他launch文件调用。
- `play_back.launch`: 可视化回放log的luanch文件
   - 用法:
     - 一个终端运行`roslaunch humanoid_controllers play_back.launch`
     - 另外的一个终端运行`rosbag play <log文件路径>`
     - 可以用于回放rosbag文件, 排查问题。
  
- `play_back_mpc.launch`: 逐帧回放MPC求解过程的launch文件
   - 用法:
     - 一个终端运行`roslaunch humanoid_controllers play_back_mpc.launch rosbag_file:=<log文件路径>`
     - 没有可视化界面，可以通过输出的log文件观察MPC求解过程。
