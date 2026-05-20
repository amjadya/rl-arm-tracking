"""Train a PPO policy to track moving targets with the Panda arm."""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import PandaTrackingEnv

SEED = 42
N_ENVS = 8
TOTAL_TIMESTEPS = 1_000_000


def make_env():
    return PandaTrackingEnv()


def main():
    vec_env = make_vec_env(
        make_env,
        n_envs=N_ENVS,
        seed=SEED,
        vec_env_cls=SubprocVecEnv,
    )

    eval_env = Monitor(make_env())

    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=SEED,
        verbose=1,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.0,
        tensorboard_log="./logs/tb/",
        device="cpu",
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path="./checkpoints/",
        name_prefix="ppo_panda",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/best/",
        log_path="./logs/",
        eval_freq=max(10_000 // N_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    callbacks = CallbackList([checkpoint_cb, eval_cb])

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
    )

    model.save("./checkpoints/ppo_panda_final.zip")


if __name__ == "__main__":
    main()
