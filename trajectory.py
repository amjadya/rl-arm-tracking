"""Trajectories for the arm to track.

A Trajectory is a precomputed array of (pos, vel) sampled at the sim
timestep for one episode. Shape is pluggable: only `make_circle` exists
now; a random-walk "fly" and other shapes are future generators that
return the same Trajectory type.

Reach model (verified in verify_length.py): the Panda's reachable space
is a sphere of radius ~0.85 m centered on the joint2 shoulder anchor.
"""

from dataclasses import dataclass

import numpy as np

SHOULDER = np.array([0.0, 0.0, 0.333])
R_REACH = 0.85

CENTER_BALL_R = 0.40   # circle centers spawn evenly in a ball this big about
                       # SHOULDER. Kept easy for the first de-risk run; widen
                       # later as a difficulty curriculum.
RADIUS_RANGE = (0.10, 0.60)   # size of the circle itself (independent of center placement)
PERIOD_RANGE = (1.5, 4.0)   # s per lap; this is the (tight) speed randomizer


@dataclass
class Trajectory:
    pos: np.ndarray   # (N, 3) world-frame target positions
    vel: np.ndarray   # (N, 3) target velocities
    dt: float
    reach_fraction: float  # fraction of samples within R_REACH of SHOULDER

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        i = min(max(int(round(t / self.dt)), 0), len(self.pos) - 1)
        return self.pos[i], self.vel[i]


def _reach_fraction(pos: np.ndarray) -> float:
    return float((np.linalg.norm(pos - SHOULDER, axis=1) <= R_REACH).mean())


def _rot(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def make_circle(
    rng: np.random.Generator, dt: float, duration: float
) -> Trajectory:
    """A circle with randomized center, radius, plane orientation
    (yaw/pitch/roll), speed, direction, and phase."""
    # center: a point spread evenly through a ball of radius CENTER_BALL_R
    # about the shoulder. u**(1/3) counteracts a ball's volume growing
    # outward, so points don't clump near the middle.
    dir_ = rng.normal(size=3)
    dir_ /= np.linalg.norm(dir_)
    center = SHOULDER + dir_ * CENTER_BALL_R * rng.random() ** (1 / 3)
    radius = rng.uniform(*RADIUS_RANGE)
    omega = (2 * np.pi / rng.uniform(*PERIOD_RANGE)) * rng.choice([-1.0, 1.0])
    phase = rng.uniform(0.0, 2 * np.pi)
    rot = _rot(*rng.uniform(0.0, 2 * np.pi, 3))  # random yaw, pitch, roll

    t = np.arange(0.0, duration, dt)
    a = omega * t + phase
    z = np.zeros_like(t)
    local_p = radius * np.stack([np.cos(a), np.sin(a), z], axis=1)
    local_v = radius * omega * np.stack([-np.sin(a), np.cos(a), z], axis=1)
    pos = center + local_p @ rot.T
    vel = local_v @ rot.T
    return Trajectory(pos, vel, dt, _reach_fraction(pos))


# TODO: make_fly(rng, dt, duration)  -> bounded random walk within a sphere,
#       speed sampled from a range; returns the same Trajectory type.
# TODO: make_figure8(...), make_line(...) as needed.


def _show(seed: int) -> None:
    """Visualize one trajectory in the MuJoCo viewer: faint dots for the
    whole path, a red marker moving along it, the Panda for scale."""
    import time

    import mujoco
    import mujoco.viewer
    from robot_descriptions import panda_mj_description

    model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    traj = make_circle(np.random.default_rng(seed), dt, duration=6.0)
    print(f"seed {seed}: reach_fraction = {traj.reach_fraction:.2f}, "
          f"{len(traj.pos)} samples")

    with mujoco.viewer.launch_passive(model, data) as v:
        path = traj.pos[:: max(1, len(traj.pos) // 400)]
        n = len(path)
        v.user_scn.ngeom = n + 1
        for g, p in enumerate(path):
            mujoco.mjv_initGeom(
                v.user_scn.geoms[g], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.006, 0, 0], pos=p, mat=np.eye(3).flatten(),
                rgba=[0.4, 0.7, 1.0, 0.35],
            )
        mujoco.mjv_initGeom(
            v.user_scn.geoms[n], type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.025, 0, 0], pos=traj.pos[0], mat=np.eye(3).flatten(),
            rgba=[1.0, 0.1, 0.1, 1.0],
        )
        t0 = time.time()
        while v.is_running():
            t = (time.time() - t0) % (len(traj.pos) * dt)
            v.user_scn.geoms[n].pos[:] = traj.at(t)[0]
            v.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if args.show:
        _show(args.seed)
    else:
        tr = make_circle(np.random.default_rng(args.seed), 0.002, 4.0)
        print(f"seed {args.seed}: reach_fraction = {tr.reach_fraction:.2f}, "
              f"pos {tr.pos.shape}, vel {tr.vel.shape}")
