import sys
import numpy as np

from tongverse.env import Env, app

def main(task_id):
    # Initialize the environment with task ID and seed.
    env = Env(task_id=task_id, seed=None)
    # Always call env.reset() to initialize the environment.
    env.reset()

    task_params = env.get_task_params()
    print("TASK INFO")
    print(f"{task_params}")
    agent_params = env.get_robot_params()
    print("AGENT INFO")
    print(f"{agent_params}")

    action = {
        "arms": {
            "ctrl_mode": "position",
            "joint_values": [0.0] * 14,
            "stiffness": [50.0, 0, 0, 0, 50.0, 0, 0, 0, 50.0, 0, 0, 0, 0, 0],
            "dampings": [0.0] * 14,
        },
        "legs": {
            "ctrl_mode": "effort",
            "joint_values": np.zeros(12),
            "stiffness": None,
            "dampings": None,
        },
        "head": {
            "ctrl_mode": "position",
            "joint_values": np.zeros(2),
            "stiffness": None,
            "dampings": None,
        }
    }

    # create solver for task 1
    # pylint: disable=import-outside-toplevel,exec-used
    exec(f"from task_{task_id}_solver.task_solver import TaskSolver", globals())
    # pylint: disable=undefined-variable
    solver = TaskSolver(task_params, agent_params)  # noqa: F821

    while app.is_running():
        # Apply the action to the robot and get the updated observation and task status.
        obs, is_done = env.step(action)

        # If the simulation is stopped or task is finished, then exit simulation.
        # Important: Please do not remove this line.
        if is_done:
            print(obs["extras"])
            break

        action = solver.next_action(obs)


if __name__ == "__main__":
    main(int(sys.argv[1]))
    app.close()