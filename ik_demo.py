"""Drive the Franka Panda's hand body origin to a hardcoded 3D target
using iterative inverse kinematics (Jacobian pseudo-inverse).

Run with:
    uv run python ik_demo.py

Panda joint-limit gotcha (easy to forget):
    joint4 is constrained to negative angles only: [-3.07, -0.07] rad
    joint6 is constrained to positive angles only: [-0.02,  3.75] rad
The IK loop clamps qpos against model.jnt_range every iteration so we
can't drift past these.
"""

import time

import mujoco
import mujoco.viewer
import numpy as np
from robot_descriptions import panda_mj_description


TARGET = np.array([0.5, -0.5, 0.5]) # world-frame point: 50 cm righ, 50 cm up, 50cm forward
STEP_SIZE = 0.05                    # IK update magnitude per iteration (smaller = more frames)
MAX_ITERS = 200                     # safety cap
TOL = 1e-3                          # convergence threshold: 1 mm
VIEWER_SLEEP = 0.03                 # seconds between viewer updates (bigger = each frame held longer)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    # Pre-allocate Jacobian buffers; mj_jac writes into these in place.
    # jacr is required by the API but unused for position-only IK.
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    print(f"Target:        {TARGET}")
    print(f"Initial hand:  {data.xpos[hand_id]}")
    print("Launching viewer (close window to exit).\n")

    iteration = 0
    converged = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Draw the target as a render-only marker on top of the model.
        viewer.user_scn.ngeom = 1
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.05, 0, 0]),  # comically large for visibility debugging
            pos=TARGET,
            mat=np.eye(3).flatten(),
            rgba=np.array([1.0, 0.0, 0.0, 1.0]),
        )

        while viewer.is_running() and iteration < MAX_ITERS and not converged:
            mujoco.mj_forward(model, data)

            current = data.xpos[hand_id]
            error = TARGET - current
            err_norm = np.linalg.norm(error)
            print(f"  iter {iteration:3d}  ||error|| = {err_norm * 1000:7.3f} mm")

            if err_norm < TOL:
                converged = True
                break

            mujoco.mj_jac(model, data, jacp, jacr, current, hand_id)
            dq = np.linalg.pinv(jacp) @ error

            data.qpos += STEP_SIZE * dq
            np.clip(
                data.qpos,
                model.jnt_range[:, 0],
                model.jnt_range[:, 1],
                out=data.qpos,
            )

            viewer.sync()
            time.sleep(VIEWER_SLEEP)
            iteration += 1

        mujoco.mj_forward(model, data)
        final = data.xpos[hand_id]
        final_err_mm = float(np.linalg.norm(TARGET - final)) * 1000

        if converged:
            print(f"\nConverged in {iteration} iterations.")
        else:
            print(f"\nDid not converge within {MAX_ITERS} iterations.")
        print(f"  Final hand:   {final}")
        print(f"  Final error:  {final_err_mm:.3f} mm")
        print("\nClose the viewer window to exit.")

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
