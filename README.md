# Quadruped Perturbation Recovery

A quadruped robot trained with PPO to walk and recover from random external force perturbations using [dm_control](https://github.com/google-deepmind/dm_control) and [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3).

## Demo Videos

| Full Demo | Challenge Video |
|-----------|----------------|
| Side-by-side comparison across 3 episodes | Per-magnitude isolated challenges with fresh resets |

> Video links: see the [latest release](../../releases/latest)

## What It Does

The robot learns in two phases via a curriculum:

1. **Walking phase (0–500k steps):** No perturbations. The robot learns stable forward locomotion using dm_control's quadruped walk task reward.
2. **Recovery phase (500k–30M steps):** Random external forces (50–300N) are applied to the torso from random horizontal directions. The robot learns to absorb impacts and continue walking.

The final policy handles forces up to 300N — 50% beyond its training range — demonstrating genuine generalization rather than memorization.

## Results

| Metric | Value |
|--------|-------|
| Training steps | 30M |
| Final reward | 880 / ~1000 |
| Episode length | 1000 steps (never falls during eval) |
| Max perturbation handled | 300N (training range: 50–200N) |
| Policy std (final) | 1.02 |

## Repository Structure

```
├── recovery_env.py   # Gymnasium wrapper around dm_control quadruped
├── train.py          # PPO training script with curriculum callback
├── render_video.py   # Renders full demo, highlight reel, and challenge video
├── test_all.py       # Pre-training test suite (14 tests)
```

## Setup

```bash
pip install dm_control stable-baselines3 opencv-python gymnasium torch
```

## Training

```bash
MUJOCO_GL=disabled python train.py
```

Checkpoints saved to `./checkpoints/` every 1M steps.

To resume from a checkpoint with a reduced learning rate (recommended after 10M steps):
set `RESUME_MODEL` and `RESUME_VECNORM` in `train.py` and restart.

### Compute

This project was trained on the [NCSA Delta](https://www.ncsa.illinois.edu/research/project-highlights/delta/) cluster at the University of Illinois Urbana-Champaign, accessed via the [Campus Research Computing (CRN)](https://campuscluster.illinois.edu/) program. Training used a single H200 GPU node with 10 CPUs.

- 8 parallel environments via `SubprocVecEnv` (`start_method="fork"`, required on Python 3.13)
- `MUJOCO_GL=disabled` for headless simulation (no display stack on compute nodes)
- ~15 hours total wall-clock time across two training runs (~10 hours for the first 10M steps at lr=3e-4, ~4.5 hours for the resumed 20M steps at lr=1e-4)
- Rendering done locally on Windows after downloading the trained model — cluster lacked OpenGL/EGL/OSMesa stack

## Rendering

Requires a display or GPU render stack (EGL/OSMesa). On Windows, standard OpenGL drivers work out of the box.

```bash
python render_video.py --model recovery_policy.zip --vecnorm recovery_policy_vecnorm.pkl
```

Produces three files:
- `recovery_demo.mp4` — full 3-episode side-by-side (walker only vs recovery trained)
- `highlight_clip.mp4` — best recovery moment per force magnitude
- `challenge_video.mp4` — isolated per-magnitude challenges with fresh resets

## Environment Details

- **Observation space (46-dim):** joint positions, joint velocities, torso quaternion, linear/angular velocity, foot contacts
- **Action space (12-dim):** torque targets for 12 actuated joints, bounded [-1, 1]
- **Reward:** dm_control walk reward + energy penalty − fall penalty
- **Perturbations:** random horizontal force, 50–200N during training, applied for one physics step

## Training Details

| Hyperparameter | Phase 1 (0–10M) | Phase 2 (10M–30M) |
|----------------|-----------------|-------------------|
| Learning rate | 3e-4 | 1e-4 |
| n_epochs | 4 | 4 |
| ent_coef | 0.001 | 0.001 |
| batch_size | 64 | 64 |
| n_envs | 8 | 8 |

Learning rate was reduced at 10M steps to address entropy drift (std climbing from 0.9 to 1.43 without reward improvement). The reduction brought std back to 1.02 and reward climbed from 838 to 880.
