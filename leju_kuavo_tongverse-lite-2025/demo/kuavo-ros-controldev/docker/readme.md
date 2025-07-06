# Build docker image&container for Kuavo-MPC-WBC
## 1. Install Docker
Follow the instructions on the official Docker website to install Docker on your system. 
## 2. Pull ROS Noetic Desktop Full
By default, the Docker image for Kuavo-MPC-WBC will be based on the ROS Noetic Desktop Full image. Use the command below to pull the image. 
```bash
docker pull osrf/ros:noetic-desktop-full
```
## 3.Build Docker Image for Kuavo-MPC-WBC
We provide two type of dockerfile for Kuavo-MPC-WBC. If U have GPU in your computer, please refer to 3.2 to build docker image with GPU support. Otherwise, please refer to 3.1 to build docker image without GPU support.
### 3.1 No GPU Version
> Note: There is a docker image file stored in [google drive](https://drive.google.com/file/d/16CUtP8OSp4cNLFeJd-tEhhlYZhZ2t_KU/view?usp=sharing). You can download it and skip the following steps.

Use the command below to build docker image for Kuavo-MPC-WBC. 
Note:
- If you are not in mainland China, maybe U need to comment out the following code:
```dockerfile
RUN apt-get update -y && apt-get install ca-certificates -y && \
    sh -c 'echo "deb https://mirrors.ustc.edu.cn/ubuntu/ focal main restricted universe multiverse\ndeb https://mirrors.ustc.edu.cn/ubuntu/ focal-updates main restricted universe multiverse\ndeb https://mirrors.ustc.edu.cn/ubuntu/ focal-backports main restricted universe multiverse\ndeb https://mirrors.ustc.edu.cn/ubuntu/ focal-security main restricted universe multiverse" > /etc/apt/sources.list' && \
    apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata && \
    apt-get install -y dirmngr gnupg2 && \
    sh -c 'echo "deb http://mirrors.ustc.edu.cn/ros/ubuntu/ focal main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 &&\
    rm -rf /var/lib/apt/lists/*
```
- change timezone by your location if U are not in mainland China.
```dockerfile
ENV TZ=Asia/Shanghai
```
- Due to some reasons, I can't install crocoddyl in my computer, so I comment it out. But U could try to uncomment it.
```dockerfile
RUN pip3 install --user crocoddyl \
    && pip3 install meshcat
```

- change docker image name and tag in `build.sh`
```bash
IMAGE_NAME="kuavo_mpc_wbc_img"
IMAGE_TAG=0.3
```
- run the build script
```bash
cd <path-to-dockerfile>
./build.sh
```
The docker image will be tagged as `kuavo_mpc_wbc_img:0.3`.
The whole process may take 10-20 minutes to complete. If U are in mainland China, maybe U need VPN to speed up the process.
### 3.2 GPU Version
> This dockerfile is provided by `Ivo Vatavuk`. However, I haven't tested it yet beacause there is no GPU in my computer. If you have any problem, please let me know.

To build the docker:
```bash
docker build -f Dockerfile.GPU -t humanoid_control_img:noetic .
```
To run the docker container:
```bash
   docker run -it --rm --net host --gpus all \
        -v /dev:/dev \
        --privileged \
        --group-add=dialout \
        --ulimit rtprio=99 \
        --cap-add=sys_nice \
        -e DISPLAY=$DISPLAY \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        humanoid_control_img:noetic \
        bash
```
## 4.Build Docker Container for Kuavo-MPC-WBC
Use following command to run docker container for Kuavo-MPC-WBC. We bind the current directory to the container's `/root/kuavo_ws` directory, which is the workspace for Kuavo-MPC-WBC.
```bash
CATKIN_WS=$HOME/kuavo_ws
mkdir -p $CATKIN_WS/src && cd $CATKIN_WS
docker run --privileged -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --mount type=bind,source=$CATKIN_WS,target=/root/kuavo_ws --workdir /root/kuavo_ws --name kuavo_mpc_wbc kuavo_mpc_wbc_img:0.2 /bin/zsh
```
Note: 
- We use `zsh`(with some useful plugins) instead of bash as default shell.

# Run Kuavo-MPC-WBC in Docker Container
## 1. Start Docker Container
Start the docker container by running the command in step 4.
```bash
xhost + # Allow container to access the local X server
docker start kuavo_mpc_wbc
docker exec -it kuavo_mpc_wbc /bin/zsh
``` 
## 2. Clone Kuavo-MPC-WBC Repository
clone the all repositories to the container's `/root/kuavo_ws/src` directory.
```bash
cd /root/kuavo_ws/src
#克隆以下仓库：
https://www.lejuhub.com/highlydynamic/kuavo
https://www.lejuhub.com/highlydynamic/kuavo_mpc
https://www.lejuhub.com/lisishu/my-crocoddyl-example
```
## 3. Build Kuavo-MPC-WBC
### Configure the workspace
Under the root folder execute the following command.
```bash
# Initialize the catkin workspace
catkin init
catkin config --extend /opt/ros/noetic
catkin config -DCMAKE_BUILD_TYPE=RelWithDebInfo # Important! 
```
### Build
#### Build Kuavo(ros version)
```bash
catkin build dynamic_biped
```
#### Build bipedal_robot_ros
Taking bipedal_robot_ros as an example. Under the root folder execute the following commands.
```bash
cd /root/kuavo_ws
source devel/setup.zsh
catkin build bipedal_robot_ros
echo "source /root/kuavo_ws/devel/setup.zsh" >> ~/.zshrc
```
Note:
- Pinocchio and hpp-fcl have been installed in the docker image, so we do `NOT` need to install them again😼.
- The default ROBOT_VERSION of kuavo is 34, U can change it in `~/.zshrc`. 
<!-- ### Run 
Please refer to other readme files for running the demo in detail.  -->

# Test
## 1. Mujoco simulator
By typing the following command, you can test if the mujoco is installed correctly. If successful, you will see a window pop up.
```bash
simulate
```
## 2. Kuavo with drake visualizer
First, make sure you have compiled kuavo (dynamic_biped). Then, start the drake visual interface:
```bash
drake-visualizer
```
Start the ROS version of the kuavo controller:
```bash
rosrun dynamic_biped highlyDynamicRobot_node
```
Pressing ‘r’ on the keyboard will enter the walking state, pressing ‘c’ will exit the walking state, for more usage please refer to the kuavo repository.
## 3. Ocs2 MPC
The follwing steps are based on the new dockerfile, if you are using the old one, please install the following dependencies in docker container. After that, you can start roslaunch normally.
```bash
apt-get update -y
apt-get install -y gnome-terminal \
dbus-x11 libcanberra-gtk-module libcanberra-gtk3-module
```
### 3.1 biao_mpc branch
In the biao_mpc branch of the [kuavo_mpc repository](https://www.lejuhub.com/highlydynamic/kuavo_mpc), we provide a demo of ocs2 mpc. 

After compiling, typing the following command can launch a mpc demo.
```bash
cd /root/kuavo_ws
source devel/setup.zsh
roslaunch ocs2_biped_robot_ros biped_robot_ddp.launch
```
You can type commands in these terminals to control the robot.
### 3.2 humanoid-control-wbc-mpc branch
In the `humanoid-control-wbc-mpc` branch of the [kuavo_mpc repository](https://www.lejuhub.com/highlydynamic/kuavo_mpc), We provide mpc-wbc control demo with access to mujoco simulator. 

Start mujoco simulator and controller in a terminal:
```bash
roslaunch humanoid_controllers load_cheat_controller.launch
```
Once the above launch is complete (mainly because the compilation of CppAdInterface is time-consuming, you can set `recompileLibrariesCppAd` in `task.info` to `false` to avoid recompilation), press the spacebar on the mujoco page to start the simulation (mujoco simulation is paused by default). If you do not see the mujoco interface, you can enter `xhost +` in the terminal of the host (outside the docker container).

You can set the gait according to the prompts in terminals.
