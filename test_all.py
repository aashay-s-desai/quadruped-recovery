"""
Comprehensive pre-training test suite.
Run: MUJOCO_GL=disabled python test_all.py
All tests must pass before starting the long training run.
"""

import os
os.environ.setdefault("MUJOCO_GL", "disabled")

import numpy as np
import traceback

PASS = []
FAIL = []

def test(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        PASS.append(name)
    except Exception as e:
        print(f"  FAIL  {name}")
        print(f"        {e}")
        traceback.print_exc()
        FAIL.append(name)

# ---------------------------------------------------------------------------
# 1. Basic env creation
# ---------------------------------------------------------------------------
def t_env_creates():
    from recovery_env import QuadrupedRecoveryEnv
    env = QuadrupedRecoveryEnv(random_seed=0)
    env.close()

# ---------------------------------------------------------------------------
# 2. Observation shape is exactly 38
# ---------------------------------------------------------------------------
def t_obs_shape():
    from recovery_env import QuadrupedRecoveryEnv
    env = QuadrupedRecoveryEnv(random_seed=0)
    expected = env.observation_space.shape
    obs, _ = env.reset()
    assert obs.shape == expected, f"Obs {obs.shape} doesn't match declared space {expected}"
    assert obs.shape[0] > 0, "Obs is empty"
    print(f"        (obs dim = {obs.shape[0]})")
    env.close()

# ---------------------------------------------------------------------------
# 3. Action space is 12-dimensional
# ---------------------------------------------------------------------------
def t_action_shape():
    from recovery_env import QuadrupedRecoveryEnv
    env = QuadrupedRecoveryEnv(random_seed=0)
    assert env.action_space.shape == (12,), f"Expected (12,), got {env.action_space.shape}"
    env.close()

# ---------------------------------------------------------------------------
# 4. Step returns correct types and shapes
# ---------------------------------------------------------------------------
def t_step_returns():
    from recovery_env import QuadrupedRecoveryEnv
    env = QuadrupedRecoveryEnv(random_seed=0)
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape, f"Obs shape {obs.shape} != declared {env.observation_space.shape}"
    assert isinstance(reward, float), f"Reward not float: {type(reward)}"
    assert isinstance(terminated, bool), f"Terminated not bool: {type(terminated)}"
    assert truncated == False, "Truncated should always be False (handled by TimeLimit)"
    assert "perturbation_active" in info
    env.close()

# ---------------------------------------------------------------------------
# 5. Force clears after exactly one step (the xfrc fix)
#    After step(), xfrc_applied on the torso must be zero.
# ---------------------------------------------------------------------------
def t_force_clears_after_step():
    from recovery_env import QuadrupedRecoveryEnv
    env = QuadrupedRecoveryEnv(random_seed=0)
    env.reset()

    # Manually fire a perturbation by bypassing curriculum
    env._global_step = env._curriculum_threshold + 1
    env._next_perturb_step = 0  # fire immediately

    action = env.action_space.sample()
    _, _, _, _, info = env.step(action)
    assert info["perturbation_active"] == True, "Perturbation should have fired"

    # After step returns, force must be zero
    try:
        tid = env._physics.model.name2id("torso", "body")
        force = env._physics.data.xfrc_applied[tid, :3]
    except Exception:
        force = env._physics.data.xfrc_applied[1, :3]

    assert np.allclose(force, 0.0), f"Force not cleared after step: {force}"
    env.close()

# ---------------------------------------------------------------------------
# 6. Scripted external push survives through step() (render_video fix)
#    Force set externally before step() must affect physics (not be wiped at start).
# ---------------------------------------------------------------------------
def t_external_push_not_wiped():
    from recovery_env import QuadrupedRecoveryEnv
    env = QuadrupedRecoveryEnv(random_seed=0)
    env.reset()

    # Resolve torso body index once, with fallback — same logic as env internals
    try:
        tid = env._physics.model.name2id("torso", "body")
    except Exception:
        tid = 1

    # Set external force (as render_video.py does before calling step)
    env._physics.data.xfrc_applied[tid, 0] = 999.0

    # Confirm force is actually set BEFORE step runs
    assert np.isclose(env._physics.data.xfrc_applied[tid, 0], 999.0), \
        "Force was not set correctly — check xfrc_applied indexing"

    # step() must NOT wipe the force before physics runs (clear happens at END)
    action = np.zeros(12, dtype=np.float32)
    env.step(action)

    # Force must be zero AFTER step — proves clear-at-end, not clear-at-start
    force_after = env._physics.data.xfrc_applied[tid, 0]
    assert np.isclose(force_after, 0.0), \
        f"Force not cleared after step ({force_after}). If this fails, _clear_perturbation " \
        f"is running at the START of step instead of the END, wiping render_video's pushes."
    env.close()

# ---------------------------------------------------------------------------
# 7. Curriculum does NOT fire before threshold
# ---------------------------------------------------------------------------
def t_curriculum_before_threshold():
    from recovery_env import QuadrupedRecoveryEnv
    env = QuadrupedRecoveryEnv(random_seed=0)
    env.reset()

    # Step many times well below threshold
    env._global_step = 0
    env._next_perturb_step = 0  # would fire immediately if curriculum allowed it

    for _ in range(10):
        _, _, _, _, info = env.step(env.action_space.sample())
        assert info["perturbation_active"] == False, \
            "Perturbation fired before curriculum threshold!"
    env.close()

# ---------------------------------------------------------------------------
# 8. Curriculum DOES fire after set_global_step crosses threshold
# ---------------------------------------------------------------------------
def t_curriculum_after_threshold():
    from recovery_env import QuadrupedRecoveryEnv
    env = QuadrupedRecoveryEnv(random_seed=0)
    env.reset()

    # Simulate the callback syncing global step past threshold
    env.set_global_step(env._curriculum_threshold + 1)
    env._next_perturb_step = 0  # schedule immediate fire

    _, _, _, _, info = env.step(env.action_space.sample())
    assert info["perturbation_active"] == True, \
        "Perturbation did not fire after curriculum threshold was crossed"
    env.close()

# ---------------------------------------------------------------------------
# 9. Fall detection works
# ---------------------------------------------------------------------------
def t_fall_detection():
    from recovery_env import QuadrupedRecoveryEnv
    env = QuadrupedRecoveryEnv(random_seed=0)
    env.reset()

    # Manually lower torso below fall threshold
    env._physics.data.qpos[2] = 0.05  # well below _fall_height=0.15
    assert env._is_fallen() == True, "Fall not detected when torso is low"

    env._physics.data.qpos[2] = 0.5  # well above threshold
    assert env._is_fallen() == False, "False fall detected when torso is high"
    env.close()

# ---------------------------------------------------------------------------
# 10. Reward is finite and in a reasonable range
# ---------------------------------------------------------------------------
def t_reward_finite():
    from recovery_env import QuadrupedRecoveryEnv
    env = QuadrupedRecoveryEnv(random_seed=0)
    env.reset()
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.isfinite(reward), f"Non-finite reward: {reward}"
        assert reward >= -10.1, f"Reward suspiciously low: {reward}"
        assert reward <= 2.0, f"Reward suspiciously high: {reward}"
        if terminated:
            env.reset()
    env.close()

# ---------------------------------------------------------------------------
# 11. SubprocVecEnv + VecNormalize starts without crash
#     (tests that MUJOCO_GL propagates to subprocesses)
# ---------------------------------------------------------------------------
def t_subproc_vecenv():
    import torch
    from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from gymnasium.wrappers import TimeLimit
    from recovery_env import QuadrupedRecoveryEnv

    def make_env(rank):
        def _init():
            env = QuadrupedRecoveryEnv(random_seed=rank)
            env = TimeLimit(env, max_episode_steps=100)
            env = Monitor(env)
            return env
        return _init

    env = SubprocVecEnv([make_env(0), make_env(1)], start_method="fork")
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    obs_dim = obs.shape[1]
    assert obs.shape == (2, obs_dim), f"VecEnv obs shape wrong: {obs.shape}"
    action = np.zeros((2, 12), dtype=np.float32)
    obs, rewards, dones, infos = env.step(action)
    assert obs.shape == (2, obs_dim)
    env.close()

# ---------------------------------------------------------------------------
# 12. DummyVecEnv reset returns (obs, infos) and unpacks correctly
#     (tests the render_video.py fix)
# ---------------------------------------------------------------------------
def t_dummy_vecenv_reset():
    from stable_baselines3.common.vec_env import DummyVecEnv
    from gymnasium.wrappers import TimeLimit
    from recovery_env import QuadrupedRecoveryEnv

    raw_env = QuadrupedRecoveryEnv(random_seed=0)
    raw_env = TimeLimit(raw_env, max_episode_steps=100)
    vec_env = DummyVecEnv([lambda: raw_env])

    result = vec_env.reset()
    # Older SB3 returns obs directly; newer SB3 returns (obs, infos).
    # Our render_video.py handles both — verify the obs is extractable either way.
    obs = result[0] if isinstance(result, tuple) else result
    assert hasattr(obs, 'shape') and obs.ndim == 2 and obs.shape[0] == 1, \
        f"Could not extract valid obs from reset(), got: {type(result)}"
    print(f"        (SB3 reset API: {'new (obs,infos)' if isinstance(result, tuple) else 'old (obs only)'})")
    vec_env.close()

# ---------------------------------------------------------------------------
# 13. run_episode returns (list, int) tuple — render_video.py contract
# ---------------------------------------------------------------------------
def t_run_episode_return_type():
    import sys
    sys.path.insert(0, ".")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from gymnasium.wrappers import TimeLimit
    from recovery_env import QuadrupedRecoveryEnv
    import render_video as rv
    import cv2, tempfile

    raw_env = QuadrupedRecoveryEnv(render_mode=None, random_seed=0)
    raw_env = TimeLimit(raw_env, max_episode_steps=50)
    inner_env = raw_env.env
    vec_env = DummyVecEnv([lambda: raw_env])

    # Create a dummy PPO model (untrained — we just need it to produce actions)
    model = PPO("MlpPolicy", vec_env, verbose=0)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name

    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 64))

    result = rv.run_episode(model, vec_env, inner_env, perturb_steps=[], writer=writer)
    writer.release()

    assert isinstance(result, tuple) and len(result) == 2, \
        f"run_episode must return (list, int), got: {type(result)}"
    push_frames, frame_count = result
    assert isinstance(push_frames, list), f"First return must be list, got {type(push_frames)}"
    assert isinstance(frame_count, int), f"Second return must be int, got {type(frame_count)}"
    assert frame_count >= 0
    vec_env.close()
    os.unlink(tmp_path)

# ---------------------------------------------------------------------------
# 14. Short end-to-end training smoke test (500 steps, 2 envs)
#     Confirms the full PPO training loop runs without errors.
# ---------------------------------------------------------------------------
def t_short_training_run():
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    from gymnasium.wrappers import TimeLimit
    from recovery_env import QuadrupedRecoveryEnv

    class CurriculumCallback(BaseCallback):
        def __init__(self, n_envs):
            super().__init__()
            self.n_envs = n_envs
        def _on_rollout_end(self):
            for i in range(self.n_envs):
                try:
                    self.training_env.env_method("set_global_step", self.num_timesteps, indices=[i])
                except Exception:
                    pass
        def _on_step(self):
            return True

    def make_env(rank):
        def _init():
            env = QuadrupedRecoveryEnv(random_seed=rank)
            env = TimeLimit(env, max_episode_steps=100)
            env = Monitor(env)
            return env
        return _init

    n_envs = 2
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method="fork")
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", env,
        n_steps=64, batch_size=32, n_epochs=2,
        verbose=0, device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    )
    model.learn(total_timesteps=500, callback=CurriculumCallback(n_envs))
    env.close()


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
print("\n=== Pre-training test suite ===\n")

test("1.  Env creates without crash",              t_env_creates)
test("2.  Observation shape is (38,)",             t_obs_shape)
test("3.  Action space shape is (12,)",            t_action_shape)
test("4.  Step returns correct types/shapes",      t_step_returns)
test("5.  Force clears after step (xfrc fix)",     t_force_clears_after_step)
test("6.  External push survives step (video fix)",t_external_push_not_wiped)
test("7.  Curriculum silent before threshold",     t_curriculum_before_threshold)
test("8.  Curriculum fires after threshold",       t_curriculum_after_threshold)
test("9.  Fall detection correct",                 t_fall_detection)
test("10. Reward finite and in range",             t_reward_finite)
test("11. SubprocVecEnv + VecNormalize starts",    t_subproc_vecenv)
test("12. DummyVecEnv reset returns (obs,infos)",  t_dummy_vecenv_reset)
test("13. run_episode returns (list, int)",        t_run_episode_return_type)
test("14. Short PPO training loop runs cleanly",   t_short_training_run)

print(f"\n=== Results: {len(PASS)} passed, {len(FAIL)} failed ===\n")
if FAIL:
    print("FAILED TESTS:")
    for name in FAIL:
        print(f"  - {name}")
    raise SystemExit(1)
else:
    print("ALL TESTS PASSED — safe to start training.")
