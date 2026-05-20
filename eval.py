"""Watch the trained PPO policy track moving targets in the MuJoCo viewer.

Loads the best checkpoint, opens the viewer, and runs episodes back-to-back
with fresh random trajectories until you close the window.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from env import PandaTrackingEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./checkpoints/best/best_model.zip")
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed for reproducible trajectories; default random.")
    ap.add_argument("--episode_steps", type=int, default=500,
                    help="Steps per episode (longer than training for better viewing).")
    args = ap.parse_args()

    model = PPO.load(args.model)
    env = PandaTrackingEnv(seed=args.seed, max_steps=args.episode_steps)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            obs, _ = env.reset()

            path = env.trajectory.pos[:: max(1, len(env.trajectory.pos) // 200)]
            viewer.user_scn.ngeom = len(path) + 1
            for i, p in enumerate(path):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.005, 0, 0],
                    pos=p,
                    mat=np.eye(3).flatten(),
                    rgba=[0.4, 0.7, 1.0, 0.35],
                )

            target_idx = len(path)
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[target_idx],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.025, 0, 0],
                pos=env.trajectory.pos[0],
                mat=np.eye(3).flatten(),
                rgba=[1.0, 0.1, 0.1, 1.0],
            )

            done = False
            while not done and viewer.is_running():
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                viewer.user_scn.geoms[target_idx].pos[:] = env.trajectory.at(env.t)[0]
                viewer.sync()
                time.sleep(env.control_dt)


if __name__ == "__main__":
    main()
