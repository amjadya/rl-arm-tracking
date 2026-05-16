"""Drive the Franka Panda's hand body origin to a hardcoded 3D target
using iterative inverse kinematics (Jacobian pseudo-inverse).

Panda joint-limit gotcha (easy to forget):
    joint4 is constrained to negative angles only: [-3.07, -0.07] rad
    joint6 is constrained to positive angles only: [-0.02,  3.75] rad
solve_ik clamps qpos against model.jnt_range every iteration.
"""

import time

import mujoco
import mujoco.viewer
import numpy as np
from robot_descriptions import panda_mj_description

from ik import solve_ik


TARGET = np.array([0.5, -0.5, 0.5]) # world-frame point: 50 cm right, 50 cm up, 50cm forward
STEP_SIZE = 1.0                     # IK update magnitude per iteration (smaller = more frames)
MAX_ITERS = 200                     # safety cap
TOL = 1e-3                          # convergence threshold: 1 mm
VIEWER_SLEEP = 0.45                 # seconds between viewer updates (bigger = each frame held longer)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    print(f"Target:        {TARGET}")
    print(f"Initial hand:  {data.xpos[hand_id]}")
    print("Launching viewer (close window to exit).\n")

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

        # solve_ik runs on a scratch state; this callback mirrors each
        # iteration back onto the live arm so we can watch it converge
        def animate(i, err_norm, qpos):
            print(f"  iter {i:3d}  ||error|| = {err_norm * 1000:7.3f} mm")
            if not viewer.is_running():
                return
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(VIEWER_SLEEP)

        q_des = solve_ik(
            model,
            TARGET,
            hand_id,
            q_seed=data.qpos.copy(),
            step_size=STEP_SIZE,
            max_iters=MAX_ITERS,
            tol=TOL,
            on_iter=animate,
        )

        data.qpos[:] = q_des
        mujoco.mj_forward(model, data)
        final = data.xpos[hand_id]
        final_err_mm = float(np.linalg.norm(TARGET - final)) * 1000

        if final_err_mm < TOL * 1000:
            print(f"\nConverged. Final error: {final_err_mm:.3f} mm")
        else:
            print(f"\nDid not converge within {MAX_ITERS} iterations.")
            print(f"  Final error: {final_err_mm:.3f} mm")
        print(f"  Final hand:  {final}")
        print("\nClose the viewer window to exit.")

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
