"""
Classical IK + PD trajectory tracking on the Franka Panda.
"""

import time

import mujoco
import mujoco.viewer
import numpy as np
from robot_descriptions import panda_mj_description

from ik import solve_ik


CIRCLE_CENTER = np.array([0.5, 0.0, 0.5])
CIRCLE_RADIUS = 0.65            # metres
CIRCLE_PERIOD = 1.80             # seconds per lap

IK_STEP_SIZE = 0.5
IK_MAX_ITERS = 100
IK_TOL = 1e-3

N_ARM_JOINTS = 7                # actuators 1..7 drive joints 1..7;
                                # actuator 8 is the gripper, left alone.


def trajectory(t: float) -> np.ndarray:
    """World-frame target position at time t: a circle in the y-z plane."""
    w = 2.0 * np.pi / CIRCLE_PERIOD
    return CIRCLE_CENTER + CIRCLE_RADIUS * np.array(
        [0.0, np.cos(w * t), np.sin(w * t)]
    )


def main() -> None:
    model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    # One scratch MjData reused by every IK solve so we are not
    # allocating inside the realtime loop.
    ik_scratch = mujoco.MjData(model)

    print(f"Circle center: {CIRCLE_CENTER}  radius: {CIRCLE_RADIUS} m  "
          f"period: {CIRCLE_PERIOD} s")
    print("Launching viewer (close window to exit).\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Render-only marker showing where the target currently is.
        viewer.user_scn.ngeom = 1
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.03, 0, 0]),
            pos=trajectory(0.0),
            mat=np.eye(3).flatten(),
            rgba=np.array([1.0, 0.0, 0.0, 1.0]),
        )

        while viewer.is_running():
            step_start = time.time()

            # 1. where should the tip be right now
            target = trajectory(data.time)

            # 2. geometry: joint angles that hit it, warm-started from
            #    the arm's actual current pose (cheap to converge).
            q_des = solve_ik(
                model,
                target,
                hand_id,
                q_seed=data.qpos.copy(),
                scratch=ik_scratch,
                step_size=IK_STEP_SIZE,
                max_iters=IK_MAX_ITERS,
                tol=IK_TOL,
            )

            # 3. hand the arm-joint targets to the position actuators.
            #    Their built-in PD does tau = Kp*(ctrl - q) - Kd*qvel.
            data.ctrl[:N_ARM_JOINTS] = q_des[:N_ARM_JOINTS]

            # 4. dynamics: physics integrates one step, actuators apply
            #    torque, gravity and inertia push back.
            mujoco.mj_step(model, data)

            # Move the target marker to the live target position.
            viewer.user_scn.geoms[0].pos[:] = target
            viewer.sync()

            # Pace the loop to wall-clock so the sim runs at real time.
            sleep_left = model.opt.timestep - (time.time() - step_start)
            if sleep_left > 0:
                time.sleep(sleep_left)


if __name__ == "__main__":
    main()
