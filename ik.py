"""Reusable Jacobian pseudo-inverse inverse kinematics.

Extracted from ik_demo.py so it can be reused by:
  - ik_demo.py        (the Day 2 teleport demo, with per-iteration animation)
  - pd_demo.py         (the Day 3 classical IK + PD trajectory tracker)
  - the RL Cartesian action wrapper, later

"""

import mujoco
import numpy as np


def solve_ik(
    model,
    target,
    body_id,
    q_seed,
    *,
    scratch=None,
    step_size=0.5,
    max_iters=200,
    tol=1e-3,
    on_iter=None,
):
    """Find joint angles that put `body_id`'s origin at `target` (world xyz).

    Args:
        model:     MjModel.
        target:    (3,) world-frame point we want the body origin to reach.
        body_id:   body whose origin we are driving (e.g. the "hand").
        q_seed:    (nq,) starting joint configuration. Pass the arm's
                   current qpos for a free warm start.
        scratch:   optional MjData to iterate on. Allocated if None.
                   Pass one in from a realtime loop to avoid per-step
                   allocation.
        step_size: fraction of the Newton step taken per iteration.
                   Small = damped/smooth/slow; ~1 = fast but can
                   overshoot or thrash near singularities.
        max_iters: safety cap on iterations.
        tol:       convergence threshold on ||target - current|| (metres).
        on_iter:   optional callback(i, err_norm, qpos) invoked every
                   iteration *before* the step. ik_demo.py uses this to
                   animate the live viewer; pd_demo.py leaves it None.

    Returns:
        (nq,) copy of the converged joint configuration. The caller's
        live MjData is untouched.
    """
    data = scratch if scratch is not None else mujoco.MjData(model)
    data.qpos[:] = q_seed

    # mj_jac writes into these in place; jacr is required by the API but
    # unused for position-only IK.
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for i in range(max_iters):
        mujoco.mj_forward(model, data)

        current = data.xpos[body_id].copy()
        error = target - current
        err_norm = float(np.linalg.norm(error))

        if on_iter is not None:
            on_iter(i, err_norm, data.qpos)

        if err_norm < tol:
            break

        mujoco.mj_jac(model, data, jacp, jacr, current, body_id)
        dq = np.linalg.pinv(jacp) @ error

        data.qpos += step_size * dq
        np.clip(
            data.qpos,
            model.jnt_range[:, 0],
            model.jnt_range[:, 1],
            out=data.qpos,
        )

    return data.qpos.copy()
