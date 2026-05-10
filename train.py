"""
Training script for quadruped perturbation recovery.
Run: MUJOCO_GL=disabled python train.py
Checkpoints saved to ./checkpoints/, final model to recovery_policy.zip
"""

import os
# Must be set before ANY import that touches dm_control, because
# dm_control reads MUJOCO_GL at import time to select the renderer.
os.environ.setdefault("MUJOCO_GL", "disabled")

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

from recovery_env import QuadrupedRecoveryEnv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
FINAL_MODEL_PATH = "recovery_policy"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS = 20_000_000
N_ENVS = 8            # H200 node has 10 CPUs; use 8 for sim, leave 2 for OS
MAX_EP_STEPS = 1000   # episode length before truncation

# Resume from checkpoint: set to None to train from scratch.
# Load best_model + matching vecnorm, continue with lower lr to reduce std drift.
RESUME_MODEL    = "./checkpoints/best_model.zip"
RESUME_VECNORM  = "./checkpoints/recovery_vecnormalize_10000000_steps.pkl"
RESUME_LR       = 1e-4   # lower than original 3e-4 to slow entropy drift

PPO_KWARGS = dict(
    n_steps=2048,
    batch_size=64,
    n_epochs=4,
    learning_rate=3e-4,   # overridden by RESUME_LR when resuming
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.001,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    ),
    verbose=1,
    tensorboard_log=LOG_DIR,
)


# ---------------------------------------------------------------------------
# Curriculum callback: syncs total timestep count into each env so the
# curriculum threshold fires based on global steps, not per-env steps.
# Uses _on_rollout_end (once per rollout) rather than _on_step to avoid
# 16k IPC round-trips per rollout from calling env_method every step.
# ---------------------------------------------------------------------------
class CurriculumCallback(BaseCallback):
    def __init__(self, n_envs: int, verbose=0):
        super().__init__(verbose)
        self.n_envs = n_envs

    def _on_rollout_end(self) -> None:
        for i in range(self.n_envs):
            try:
                self.training_env.env_method("set_global_step", self.num_timesteps, indices=[i])
            except Exception:
                pass

    def _on_step(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_env(rank: int, seed: int = 0):
    def _init():
        env = QuadrupedRecoveryEnv(random_seed=seed + rank)
        env = TimeLimit(env, max_episode_steps=MAX_EP_STEPS)
        env = Monitor(env)
        return env
    return _init


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # start_method="fork" is required on Python 3.13 — the new default
    # (forkserver) rejects subprocess creation outside __main__ guards,
    # and SB3's SubprocVecEnv prefers forkserver when available regardless
    # of the global multiprocessing default.
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)], start_method="fork")
    eval_env_raw = SubprocVecEnv([make_env(99)], start_method="fork")

    resuming = RESUME_MODEL is not None and os.path.exists(RESUME_MODEL)

    if resuming:
        print(f"Resuming from {RESUME_MODEL} with lr={RESUME_LR}")
        env = VecNormalize.load(RESUME_VECNORM, env)
        env.training = True
        env.norm_reward = True
        eval_env = VecNormalize.load(RESUME_VECNORM, eval_env_raw)
        eval_env.training = False
        eval_env.norm_reward = False
        model = PPO.load(
            RESUME_MODEL,
            env=env,
            device=device,
            custom_objects={"learning_rate": RESUME_LR},
        )
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env = VecNormalize(
            eval_env_raw, norm_obs=True, norm_reward=False,
            clip_obs=10.0, training=False,
        )
        model = PPO("MlpPolicy", env, device=device, **PPO_KWARGS)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1_000_000 // N_ENVS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="recovery",
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=CHECKPOINT_DIR,
        log_path=LOG_DIR,
        eval_freq=max(500_000 // N_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
    )
    curriculum_cb = CurriculumCallback(n_envs=N_ENVS)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb, curriculum_cb],
        progress_bar=True,
        reset_num_timesteps=not resuming,
    )

    model.save(FINAL_MODEL_PATH)
    env.save(f"{FINAL_MODEL_PATH}_vecnorm.pkl")
    print(f"Saved final model → {FINAL_MODEL_PATH}.zip")


if __name__ == "__main__":
    main()
