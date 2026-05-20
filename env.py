"""Gymnasium env: track a moving target with the Panda arm.

Approach iv (RL + IK + PD): the policy outputs a small Cartesian lead
vector added to the current target. IK turns the leading point into joint
angles; the XML's PD actuators turn those into torques.
"""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.spaces import Box
from robot_descriptions import panda_mj_description

from ik import solve_ik
from trajectory import make_circle


class PandaTrackingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        control_hz: float = 100.0,
        max_steps: int = 200,
        max_lead: float = 0.15,     # metres: ±15 cm per-step lead (covers fast circles)
        alpha: float = 0.01,        # smoothing penalty weight
        ik_max_iters: int = 20,     # cheap IK; tracking absorbs slack
        seed: int | None = None,
    ):
        super().__init__()

        # Physics
        self.model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
        self.data = mujoco.MjData(self.model)
        self.ik_scratch = mujoco.MjData(self.model)   # reused inside step()
        self.hand_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand"
        )

        # Time discretization
        self.sim_dt = self.model.opt.timestep                        # 0.002 s for Panda
        self.control_dt = 1.0 / control_hz                           # 0.01 s @ 100 Hz
        self.substeps = int(round(self.control_dt / self.sim_dt))    # 5
        self.episode_duration = max_steps * self.control_dt

        # Hyperparameters
        self.max_steps = max_steps
        self.max_lead = max_lead
        self.alpha = alpha
        self.ik_max_iters = ik_max_iters

        # Spaces
        self.action_space = Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = Box(
            low=np.array(
                [-1.2]*3 + [-np.inf]*3 + [-1.2]*3 + [-np.inf]*3,
                dtype=np.float32,
            ),
            high=np.array(
                [+1.2]*3 + [+np.inf]*3 + [+1.2]*3 + [+np.inf]*3,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # RNG
        self.rng = np.random.default_rng(seed)

        # Per-episode state (populated in reset)
        self.trajectory = None
        self.t = 0.0
        self.step_count = 0
        self.prev_ee_pos = np.zeros(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Arm to home pose
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        # New random trajectory, slightly longer than the episode for safety
        self.trajectory = make_circle(
            self.rng,
            dt=self.sim_dt,
            duration=self.episode_duration + 0.1,
        )

        self.t = 0.0
        self.step_count = 0
        self.prev_ee_pos = self.data.xpos[self.hand_id].copy()

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # 1. Where to aim this step
        target_pos, _ = self.trajectory.at(self.t)
        ik_target = target_pos + action * self.max_lead

        # 2. IK to desired joint angles, fed to the XML's PD actuators
        q_des = solve_ik(
            self.model, ik_target, self.hand_id,
            q_seed=self.data.qpos,
            scratch=self.ik_scratch,
            max_iters=self.ik_max_iters,
        )
        self.data.ctrl[:7] = q_des[:7]   # 7 arm actuators; ignore gripper

        # 3. Advance physics by one control tick (= substeps sim ticks)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        # 4. Advance trajectory clock
        self.t += self.control_dt
        self.step_count += 1

        # 5. Reward: distance penalty + smoothness penalty
        ee_pos = self.data.xpos[self.hand_id]
        target_pos_now, _ = self.trajectory.at(self.t)
        distance = float(np.linalg.norm(target_pos_now - ee_pos))
        reward = -distance - self.alpha * float(np.linalg.norm(action))

        # 6. Termination flags
        terminated = False                            # no natural end condition
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {"distance": distance}

    def _get_obs(self) -> np.ndarray:
        ee_pos = self.data.xpos[self.hand_id].copy()
        ee_vel = (ee_pos - self.prev_ee_pos) / self.control_dt
        self.prev_ee_pos = ee_pos

        target_pos, target_vel = self.trajectory.at(self.t)

        return np.concatenate(
            [ee_pos, ee_vel, target_pos, target_vel]
        ).astype(np.float32)
