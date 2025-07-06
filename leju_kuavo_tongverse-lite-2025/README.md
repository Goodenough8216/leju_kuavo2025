# 人工智能双足机器人仿真挑战赛

> Bipedal Robot Challenge powered by **TongVerse-Lite**.  
>
> Submission details and task evaluator will be released soon.

## Installation

### System Requirements

Ensure the following dependencies are installed before running the platform:

- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)


### Docker Image Download and Installation Guide


1. Download both split files to the same directory

    - [kuavo_tv-docker-release-v1.2.zip](https://kuavo.lejurobot.com/docker_images/kuavo_tv-docker-release-v1.2.zip)
    - [kuavo_tv-docker-release-v1.2.z01](https://kuavo.lejurobot.com/docker_images/kuavo_tv-docker-release-v1.2.z01)


2. Extract the files using one of these methods:

   ```bash
   # Method 1: Using zip command
   zip -F kuavo_tv-docker-release-v1.1.zip --out full.zip
   unzip full.zip

   # Method 2: Using 7z (recommended)
   7z x kuavo_tv-docker-release-v1.1.zip
   ```

3. After extraction, you'll get a tar.gz file. Load the Docker image:

   ```bash
   docker load -i kuavo_tv-docker-release-v1.1.tar.gz
   ```

    ## Notes
    - Make sure both split files are in the same directory
    - Ensure the filenames match exactly
    - If using zip command fails, try using 7z command instead
    

4. Launch the TongVerse-Lite environment:

   ```shell
   # Navigate to the biped_challenge/ folder
   # cd biped_challenge
   bash ./run_kuavo_isaac.sh
   ```

## Running a Demo

To start the demo inside the Docker environment, run:

```shell
python demo/solver_task1.py
```

For more details, see the [demo examples](demo/readme.md).

## Kuavo Native Controller Usage

Explore the example scripts in the [demo directory](demo/) to learn how to use the native Kuavo controller:

- `DemoController.py` demonstrates using the controller via the Python interface.
- `solver_task1/2/3.py` showcase tele-operation examples (keyboard-controlled) for solving tasks.

## API Usage

### Task Description:
-----------------
To retrieve task parameters:

```python
task_params = env.get_task_params()
```

Task parameters include:

- **task_goal** (*str*): The objective of the task.  
- **target_area_bbox** (*Tuple[float, float, float, float]*): Bounding box coordinates defining the target area.  
- **camera_intrinsics** (*np.ndarray*): Intrinsic matrix of the camera used in the environment.  
- **camera_resolution** (*Tuple[int]*): resolution of the camera used in the environment.  
- **other_task_specific_description** (*varied*): Additional details specific to the task.  

To retrieve robot parameters:

```python
robot_params = env.get_robot_params()
```

Robot parameters include:

```python
{
    "ordered_joint_name": List[str],  # Ordered joint names
    "arm_idx": List[int],  # Arm joint ordered indices
    "leg_idx": List[int],  # Leg joint ordered indices
    "head_idx": List[int],  # Head joint ordered indices
}
```
------

### Action Format

An action is structured as follows:

```python
action = {
    "arms": {
        "ctrl_mode": str,  # Control mode: "position" , "effort", "velocity"
        "joint_values": Optional[Union[np.ndarray, List]],  # Target joint values (shape: 14, ordered)
        "stiffness": Optional[Union[np.ndarray, List]],  # Stiffness values (shape: 14, ordered)
        "dampings": Optional[Union[np.ndarray, List]],  # Damping values (shape: 14, ordered)
    },
    "legs": {
        "ctrl_mode": str,  # Control mode: "position" , "effort", "velocity"
        "joint_values": Optional[Union[np.ndarray, List]],  # Target joint values (shape: 12, ordered)
        "stiffness": Optional[Union[np.ndarray, List]],  # Stiffness values (shape: 12, ordered)
        "dampings": Optional[Union[np.ndarray, List]],  # Damping values (shape: 12, ordered)
    },
    "head": {
        "ctrl_mode": str,  # Control mode: "position" , "effort", "velocity"
        "joint_values": Optional[Union[np.ndarray, List]],  # Target joint values (shape: 2, ordered)
        "stiffness": Optional[Union[np.ndarray, List]],  # Stiffness values (shape: 2, ordered)
        "dampings": Optional[Union[np.ndarray, List]],  # Damping values (shape: 2, ordered)
    }
}
```

**Parameters:**

- **ctrl_mode** (*str*): The control mode must be 'position', 'velocity', or 'effort' (case-sensitive).
- **joint_values** (*np.ndarray* or *List* or *None*): Target joint values, must follow joint order.
- **stiffness** (*np.ndarray* or *List* or *None*): Stiffness values, must follow joint order.
- **dampings** (*np.ndarray* or *List* or *None*): Damping values, must follow joint order.

**Notes:**

- The order of `joint_values`, `stiffness`, and `dampings` must match the joint index order.
- You can check the correct joint order using `env.get_robot_params()`.

**Example Usage:**

```python
    action = {
        "arms": {
            "ctrl_mode": "position",
            "joint_values": [0.5, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 14 arm joints
            "stiffness": [50.0, 0, 0, 0, 50.0, 0, 0, 0, 50.0, 0, 0, 0, 0, 0],
            "dampings": [0.0] * 14,
        },
        "legs": {
            "ctrl_mode": "effort",
            "joint_values": np.array([1.0, -1.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 12 leg joints
            "stiffness": None,  # Not setting stiffness
            "dampings": None,  # Not setting dampings
        },
        "head": {
            "ctrl_mode": "position",
            "joint_values": np.array([1.0, -1.0]),  # 2 head joints
            "stiffness": None,  # Not setting stiffness
            "dampings": None,  # Not setting dampings
        }
    }
```

-------

### Observation Format:

Observations are structured as follows:

```  python
    obs = {
        "Kuavo": {
            "body_state": {
                "world_position": np.ndarray,  # (3,) - Robot's world position
                "world_orient": np.ndarray,  # (4,) - World orientation (quaternion)
                "root_linear_velocity": np.ndarray,  # (3,) - Linear velocity of the robot’s root
                "root_angular_velocity": np.ndarray,  # (3,) - Angular velocity of the robot’s root
            },
            "joint_state": {
                "arms": {
                    "positions": np.ndarray,  # (14,) - Arm joint positions
                    "velocities": np.ndarray,  # (14,) - Arm joint velocities
                    "applied_effort": np.ndarray,  # (14,) - Applied effort to arms
                    "stiffness": np.ndarray,  # (14,) - Arm joint stiffness values
                    "dampings": np.ndarray,  # (14,) - Arm joint damping values
                },
                "legs": {
                    "positions": np.ndarray,  # (12,) - Leg joint positions
                    "velocities": np.ndarray,  # (12,) - Leg joint velocities
                    "applied_effort": np.ndarray,  # (12,) - Applied effort to legs
                    "stiffness": np.ndarray,  # (12,) - Leg joint stiffness values
                    "dampings": np.ndarray,  # (12,) - Leg joint damping values
                },
                "head": {
                    "positions": np.ndarray,  # (2,) - Head joint positions
                    "velocities": np.ndarray,  # (2,) - Head joint velocities
                    "applied_effort": np.ndarray,  # (2,) - Applied effort to head
                    "stiffness": np.ndarray,  # (2,) - Head joint stiffness values
                    "dampings": np.ndarray,  # (2,) - Head joint damping values
                }
            }
        },
        "camera": {
            "rgb": np.ndarray,  # (N, 3) - RGB color data per frame
            "depth": np.ndarray,  # (n, m, 1) - Depth data per frame
            "world_pose": Tuple[np.ndarray, np.ndarray],  # (position (3,), quaternion (4,))
        },
        "imu_data": {  # The IMU sensor is mounted on the base_link of Kuavo.
            "imu_time": float,
            "linear_acceleration": List[float],   
            "angular_velocity":  List[float],   
            "orientation":  List[float],   
        },
        "extras": {
            # Task-specific data (e.g., contact counts, time(minutes), hand pose, camera intrinsics)
        }
    }
```

## Submission Guidelines

### Getting Started

1. **Familiarize with Examples**
   Explore the [demo](./demo/) directory to understand how the environment works.
2. **Implement Your TaskSolver**
   Create your custom `TaskSolver` for each task in `submission/task_<id>_solver`. Please note:
   - Do not modify any code outside the `task_<id>_solver` directories.
   - Modifications to `task_launcher.py` or other core files are strictly prohibited.

### Other Python Package Dependencies

You could append other python packages you need at the end of file [docker-pip-install.sh](docker-pip-install.sh), and run `bash docker-pip-install.sh` **every time** after launching the docker image.

### Implementing Your Solver

In the [submission](./submission/) folder, we provide a solver template for each task. Implement your solver within the respective `task_<id>_solver/` folder. Below is the template provided:

```python
class TaskSolver(TaskSolverBase):
    def __init__(self) -> None:
        super().__init__()
        # Your TaskSolver implementation goes here
        raise NotImplementedError("Implement your own TaskSolver here")

    def next_action(self, obs: dict) -> dict:
        # Determine the next action based on the current observation (`obs`)
        # action = plan(obs)
        # return action
        raise NotImplementedError("Implement your own TaskSolver here")
```

- Use the `__init__()` function to initialize your solver with any necessary modules.
- Implement the `next_action()` function to determine and return the robot's next action based on the current observation `obs`. The specific formats for observations and actions are detailed in the sections that follow.

## Testing Your Solution

To test your `TaskSolver`, execute:
```bash
python submission/task_launcher.py <task-id>
```
Replace `<task-id>` with an integer from 1 to 3 corresponding to the task you are testing.

## Preparing for Submission

1. **Compress Your Work**
   Compress the entire [submission](./submission/) folder.

2. **Rename the File**
   Name the compressed file as `submission_<team-id>`.

3. **Submit to the Committee**
   Send your renamed submission file to our committee group for evaluation.

