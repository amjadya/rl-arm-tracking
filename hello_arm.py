"""Load Franka Panda in MuJoCo and visualize it.

Run with:
    uv run python hello_arm.py

Close the viewer window to exit.
"""

import mujoco
import mujoco.viewer
from robot_descriptions import panda_mj_description


def main() -> None:
    model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    print(f"Model:    {panda_mj_description.MJCF_PATH}")
    print(f"  njnt  = {model.njnt}  (joints)")
    print(f"  nq    = {model.nq}  (generalized coordinates)")
    print(f"  nv    = {model.nv}  (degrees of freedom)")
    print(f"  nu    = {model.nu}  (actuators)")
    print(f"  nbody = {model.nbody}")

    print("\nJoints:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"  [{i}] {name}")

    print("\nActuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  [{i}] {name}")

    print("\nLaunching viewer (close window to exit)...")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
