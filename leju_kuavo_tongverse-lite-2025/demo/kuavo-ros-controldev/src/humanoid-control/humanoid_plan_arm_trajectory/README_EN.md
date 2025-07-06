# Humanoid Plan Arm Trajectory

## 1. Overview

This package plans arm joint state trajectories for humanoid robots. It currently offers two interpolation methods:

1. Cubic Spline Interpolator
2. Bezier Curve Interpolator

## 2. Installation and Usage

### 2.1 Build

```bash
catkin build humanoid_plan_arm_trajectory
```

### 2.2 Run

As a ROS node:
```bash
roslaunch humanoid_plan_arm_trajectory humanoid_plan_arm_trajectory.launch
```

As a nodelet:
```bash
roslaunch humanoid_plan_arm_trajectory humanoid_plan_arm_trajectory.launch use_nodelet:=true
```

### 2.3 Integration

To integrate this package into your ROS launch file, add:

```xml
<include file="$(find humanoid_plan_arm_trajectory)/launch/humanoid_plan_arm_trajectory.launch"/>
```

For nodelet integration with an existing nodelet manager, comment out this line in the launch file:

```xml
<!-- <node pkg="nodelet" type="nodelet" name="nodelet_manager" args="manager" output="screen" /> -->
```

## 3. Interfaces

Each interpolation method provides the following interfaces:

- Service: `/<interpolate_type>/plan_arm_trajectory`
- Published Topics:
  - `/<interpolate_type>/arm_traj`: Planned arm trajectory
  - `/<interpolate_type>/arm_traj_state`: Current state of the arm trajectory

Where `<interpolate_type>` is either `cubic_spline` or `bezier`.

### 3.1  Service

**planArmTrajectoryBezierCurve Service:**

name: `/bezier/plan_arm_trajectory`

srv: `humanoid_plan_arm_trajectory/planArmTrajectoryBezierCurve`

Request Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| multi_joint_bezier_trajectory | jointBezierTrajectory[] | Array of Bezier trajectories for multiple joints |
| start_frame_time | float64 | Start time of the trajectory, unit is s(default is 0) |
| end_frame_time | float64 | End time of the trajectory, unit is s |
| joint_names | string[] | Names of the joints |

jointBezierTrajectory Message

| Field | Type | Description |
|-------|------|-------------|
| bezier_curve_points | bezierCurveCubicPoint[] | Array of Bezier curve control points |

bezierCurveCubicPoint Message

| Field | Type | Description |
|-------|------|-------------|
| end_point | float64[2] | End point of the Bezier curve segment [time, value] |
| left_control_point | float64[2] | Left control point of the Bezier curve segment [time, value] |
| right_control_point | float64[2] | Right control point of the Bezier curve segment [time, value] |

**NOTE:** A bezier curve consist of two bezierCurveCubicPoint, `p0` is the first bezierCurveCubicPoint's end point, `p1` is the first bezierCurveCubicPoint's right point, `p2` is the second bezierCurveCubicPoint's left point, `p3` is the second bezierCurveCubicPoint's end point.

Response

| Field | Type | Description |
|-------|------|-------------|
| success | bool | Indicates whether the trajectory planning was successful |

**planArmTrajectoryCubicSpline Service**

name: `/cubic_spline/plan_arm_trajectory`

srv: `humanoid_plan_arm_trajectory/planArmTrajectoryCubicSpline`

Request Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| joint_trajectory | trajectory_msgs/JointTrajectory | Joint trajectory specification |

JointTrajectory Message

| Field | Type | Description |
|-------|------|-------------|
| joint_names | string[] | Names of the joints |
| points | JointTrajectoryPoint[] | Array of trajectory points |

JointTrajectoryPoint Message

| Field | Type | Description |
|-------|------|-------------|
| positions | float64[] | Joint positions at this point |
| velocities | float64[] | Joint velocities at this point (not used) |
| accelerations | float64[] | Joint accelerations at this point (not used) |
| time_from_start | duration | Time from the start of the trajectory to this point |

Response

| Field | Type | Description |
|-------|------|-------------|
| success | bool | Indicates whether the trajectory planning was successful |

### 3.2 Published Topics

**arm_traj Topic:**

name: `<interpolate_type>/arm_traj`

msg type: `trajectory_msgs/JointTrajectory`

msg fields:

| Field | Type | Description |
|-------|------|-------------|
| header | std_msgs/Header | Header of the message |
| joint_names | string[] | Names of the joints |
| points | JointTrajectoryPoint[] | Array of trajectory points |

**NOTE:** 
Only points[0] contains the after joint values.

---

**arm_traj_state Topic:**

name: `<interpolate_type>/arm_traj_state`

msg type: `humanoid_plan_arm_trajectory/planArmState`

msg fields:

| Field | Type | Description |
|-------|------|-------------|
| progress | int32 | Progress of the trajectory, unit is ms |
| is_finished | bool | Indicates whether the trajectory is finished |

## 4. Examples

humanoid robot simulation(assume you have already built the workspace)

```bash
source <kuavo-ros-control>/devel/setup.bash
roslaunch humanoid_controllers load_normal_controller_mujoco_nodelet.launch 
```

plan arm trajectory service

```bash
roslaunch humanoid_plan_arm_trajectory humanoid_plan_arm_trajectory.launch
```

plan arm trajectory client

```bash
source <kuavo-ros-control>/devel/setup.bash
cd src/humanoid-control/humanoid_plan_arm_trajectory/scripts
python plan_arm_traj_bezier_demo.py # bezier curve
python plan_arm_traj_cubic_spline_demo.py # cubic spline
```

**FYI:**

You can write the client call service via imitating the above two demo scripts.

## 5. Notes

For improved code reusability, portability, and maintainability, users should:

- Subscribe to the result topics themselves
- Implement the final control logic based on their specific requirements
