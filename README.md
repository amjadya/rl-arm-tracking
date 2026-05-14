# rl-arm-tracking

A reinforcement learning system for tracking time-varying 3D Cartesian end-effector
trajectories on a simulated Franka Panda arm. Built with PPO (via Stable-Baselines3)
and MuJoCo.

Submitted as part of an internship application challenge.

## Status

Active build, May 14–25, 2026. Build log: [meej.ca](https://meej.ca/) (page coming soon).

## What it does

- Trains a PPO policy to drive a Franka Panda's end-effector along parametric
  trajectories (circle, figure-eight, and more as I learn)
- Benchmarks the learned policy against a classical IK + PD baseline
- Introduces one source of uncertainty: target positions may fall outside the arm's
  workspace, requiring graceful degradation

## Quick start

> **TODO:** run instructions will be added once training / evaluation entry points
> are implemented.

## Design choices

To be expanded at the end of the build: state / action / reward design, trajectory
representation, evaluation methodology, RL vs classical baseline analysis.

## Author

Amjad Yaghi -- UBC Engineering Physics. [meej.ca](https://meej.ca/) · [github.com/amjadya](https://github.com/amjadya)
