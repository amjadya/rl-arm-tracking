"""Measure the Panda hand's true reachable workspace by forward-kinematics
sampling: random valid joint angles -> where the hand lands, relative to
the base. Reports overall reach and reach in six cardinal cones (the
workspace is not a sphere).

    uv run python verify_length.py                # numbers only
    uv run python verify_length.py --plot         # matplotlib 3D scatter
    uv run python verify_length.py --show         # cloud in the sim viewer
    uv run python verify_length.py --n 1000000    # more samples
"""

import argparse
import time

import mujoco
import numpy as np
from robot_descriptions import panda_mj_description

ARM = 7  # joints 1..7; fingers ignored


def sample(model, data, n, rng):
    lo, hi = model.jnt_range[:ARM, 0], model.jnt_range[:ARM, 1]
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0")
    sh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint2")
    rel_base = np.zeros((n, 3))
    rel_sh = np.zeros((n, 3))
    for i in range(n):
        data.qpos[:ARM] = rng.uniform(lo, hi)
        mujoco.mj_forward(model, data)
        rel_base[i] = data.xpos[hand] - data.xpos[base]
        rel_sh[i] = data.xpos[hand] - data.xanchor[sh]
    return rel_base, rel_sh, data.xpos[base].copy()


def report(rel):
    r = np.linalg.norm(rel, axis=1)
    print(f"reach  min/mean/max : {r.min():.3f} / {r.mean():.3f} / {r.max():.3f} m")
    u = rel / r[:, None]
    cones = {"+x fwd": (0, 1), "-x back": (0, -1), "+y left": (1, 1),
             "-y right": (1, -1), "+z up": (2, 1), "-z down": (2, -1)}
    print("max reach by cone (within ~45 deg):")
    for name, (ax, s) in cones.items():
        m = r[s * u[:, ax] > 0.7]
        print(f"  {name:8s}: {m.max():.3f} m" if m.size else f"  {name:8s}: --")


def plot_mpl(rel, k=20_000):
    import matplotlib.pyplot as plt

    p = rel[np.random.default_rng(0).choice(len(rel), min(k, len(rel)), replace=False)]
    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1, alpha=0.2)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1)); ax.set_title(f"reachable hand positions (n={len(p)})")
    plt.show()


def show_viewer(model, data, rel, base_xpos):
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as v:
        k = min(rel.shape[0], v.user_scn.maxgeom)
        idx = np.random.default_rng(0).choice(len(rel), k, replace=False)
        if k < rel.shape[0]:
            print(f"viewer caps at {k} markers (of {rel.shape[0]}).")
        v.user_scn.ngeom = k
        for g, p in enumerate(rel[idx]):
            mujoco.mjv_initGeom(
                v.user_scn.geoms[g], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.004, 0, 0], pos=base_xpos + p,
                mat=np.eye(3).flatten(), rgba=[0.1, 0.6, 1.0, 0.35],
            )
        while v.is_running():
            v.sync()
            time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300_000)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
    data = mujoco.MjData(model)
    t = time.time()
    rel_base, rel_sh, base_xpos = sample(model, data, args.n, np.random.default_rng(0))
    print(f"samples: {args.n}  ({time.time() - t:.1f}s)")
    print("\n[from base link0]")
    report(rel_base)
    print("\n[from shoulder joint2 anchor]")
    report(rel_sh)
    if args.plot:
        plot_mpl(rel_base)
    if args.show:
        show_viewer(model, data, rel_base, base_xpos)


if __name__ == "__main__":
    main()
