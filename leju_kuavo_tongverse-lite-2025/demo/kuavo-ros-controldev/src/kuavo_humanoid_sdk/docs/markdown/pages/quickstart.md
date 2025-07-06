<a id="quickstart"></a>

# Quickstart

#### WARNING
Before running any code examples, make sure to start the robot first by executing either:

- For simulation: `roslaunch humanoid_controllers load_kuavo_mujoco_sim.launch` (Example command)
- For real robot: `roslaunch humanoid_controllers load_kuavo_real.launch` (Example command)

## Getting Started

Before using the Kuavo SDK, you must first initialize it by calling `KuavoSDK().Init()`. This is required before any other SDK functionality can be used.

Here’s a minimal example:

```python3
from kuavo_humanoid_sdk import KuavoSDK, KuavoRobot

# Initialize the SDK - this is required!
if not KuavoSDK().Init():
    print("Init KuavoSDK failed, exit!")
    exit(1)

# Create robot instance after successful initialization
robot = KuavoRobot()

# Now you can use the robot object...
```

#### WARNING
Calling SDK functions without first calling `KuavoSDK().Init()` or when it returns `False` is illegal and may lead to undefined behavior. The consequences are unpredictable and could be dangerous.

### Basic Example

Let’s break down this example:

1. First, we import the required modules:
   - `KuavoSDK` - The main SDK interface
   - `KuavoRobot` - The robot control interface
   - `time` - For timing control
2. In the `main()` function:
   - We initialize the SDK with `KuavoSDK().Init()` - this is a critical first step
   - Create a robot instance with `KuavoRobot()`
3. Basic robot control sequence:
   - `arm_reset()` - Moves the arms to their default position
   - `stance()` - Puts the robot in a stable standing position
   - `trot()` - Switches to a trotting gait mode
4. Walking control:
   - We set up a 4-second duration and walking speed of 0.3 m/s

This example demonstrates the basic workflow of initializing the SDK and controlling the robot’s movement.

Here’s a complete example showing basic robot control after initialization:

```python3
from kuavo_humanoid_sdk import KuavoSDK, KuavoRobot
import time

def main():
    # Initialize SDK
    if not KuavoSDK().Init():
        print("Init KuavoSDK failed, exit!")
        exit(1)

    robot = KuavoRobot()

    # Reset arm position
    robot.arm_reset()

    # Move to stance position
    robot.stance()

    # Switch to trot gait
    robot.trot()

    # Walk forward for 4 seconds
    duration = 4.0  # seconds
    speed = 0.3     # m/s
    start_time = time.time()
    while (time.time() - start_time < duration):
        robot.walk(linear_x=speed, linear_y=0.0, angular_z=0.0)
        time.sleep(0.1)

if __name__ == "__main__":
    main()
```
