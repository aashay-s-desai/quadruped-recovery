import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control import suite


class QuadrupedRecoveryEnv(gym.Env):
    """
    Gymnasium wrapper around dm_control's quadruped walk task.
    Adds curriculum-based random force perturbations to the torso.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, random_seed=42):
        super().__init__()
        self.render_mode = render_mode
        self._seed = random_seed

        self._env = suite.load(
            domain_name="quadruped",
            task_name="walk",
            task_kwargs={"random": random_seed},
        )

        self._physics = self._env.physics
        self._task = self._env.task

        # Compute obs dim from actual model to avoid hardcoding assumptions.
        # The dm_control quadruped has passive toe joints beyond the 12 actuated
        # joints, so nq-7 and nv-6 may exceed 12.
        # Layout: joint_pos + joint_vel + torso_quat(4) + lin_vel(3) + ang_vel(3) + contacts(4)
        nq = self._physics.model.nq  # total qpos size
        nv = self._physics.model.nv  # total qvel size
        obs_dim = (nq - 7) + (nv - 6) + 4 + 3 + 3 + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space: 12 actuators bounded [-1, 1]
        n_act = self._physics.model.nu
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32
        )

        # Perturbation state
        self._global_step = 0
        self._ep_step = 0
        self._next_perturb_step = None
        self._perturb_active = False
        # No pushes until robot has had enough time to learn to walk
        self._curriculum_threshold = 500_000
        self._perturbation_log = []

        # Fall detection: torso z below this height = fallen
        self._fall_height = 0.15

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        timestep = self._env.reset()
        self._ep_step = 0
        self._perturb_active = False
        self._perturbation_log = []
        self._clear_perturbation()
        self._schedule_next_perturbation()
        obs = self._extract_obs(timestep)
        return obs.astype(np.float32), {}

    def step(self, action):
        self._ep_step += 1
        self._global_step += 1

        self._perturb_active = False
        if (
            self._global_step >= self._curriculum_threshold
            and self._next_perturb_step is not None
            and self._ep_step >= self._next_perturb_step
        ):
            self._apply_perturbation()
            self._perturb_active = True
            self._schedule_next_perturbation()

        timestep = self._env.step(action)

        obs = self._extract_obs(timestep)
        reward = self._compute_reward(timestep, action)
        terminated = self._is_fallen()

        # Clear force AFTER physics steps so it acts for exactly one step.
        # Clearing at start would break render_video.py's scripted pushes,
        # which apply force externally before calling step().
        self._clear_perturbation()

        return obs.astype(np.float32), reward, terminated, False, {"perturbation_active": self._perturb_active}

    def render(self):
        if self.render_mode == "rgb_array":
            # camera_id=0 is always valid; "back_close" is the back-view camera
            # in dm_control's quadruped if you want to switch to a named camera
            return self._physics.render(height=480, width=640, camera_id=0)
        return None

    def close(self):
        self._env.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_obs(self, timestep):
        phys = self._physics
        joint_pos = phys.data.qpos[7:].copy()     # 12 joint positions
        joint_vel = phys.data.qvel[6:].copy()     # 12 joint velocities
        torso_quat = phys.data.qpos[3:7].copy()   # 4: quaternion (w, x, y, z)
        torso_lin_vel = phys.data.qvel[:3].copy()  # 3: linear velocity
        torso_ang_vel = phys.data.qvel[3:6].copy() # 3: angular velocity
        foot_contacts = self._get_foot_contacts()  # 4: binary contact flags
        return np.concatenate([
            joint_pos, joint_vel, torso_quat,
            torso_lin_vel, torso_ang_vel, foot_contacts,
        ])

    def _get_foot_contacts(self):
        phys = self._physics
        foot_geom_names = ["foot_front_left", "foot_front_right",
                           "foot_back_left", "foot_back_right"]
        contacts = np.zeros(4, dtype=np.float32)
        contact_geom_ids = set()
        for i in range(phys.data.ncon):
            contact_geom_ids.add(phys.data.contact[i].geom1)
            contact_geom_ids.add(phys.data.contact[i].geom2)
        for i, name in enumerate(foot_geom_names):
            try:
                gid = phys.model.name2id(name, "geom")
                contacts[i] = float(gid in contact_geom_ids)
            except Exception:
                pass
        return contacts

    def _compute_reward(self, timestep, action):
        # Use dm_control's tuned walk reward as base (speed-matching signal)
        base_reward = float(timestep.reward) if timestep.reward is not None else 0.0

        # Penalize large actions → smoother, lower-energy gaits
        energy_penalty = -0.001 * float(np.sum(np.square(action)))

        # Hard penalty for falling
        fall_penalty = -5.0 if self._is_fallen() else 0.0

        return base_reward + energy_penalty + fall_penalty

    def _is_fallen(self):
        return float(self._physics.data.qpos[2]) < self._fall_height

    def _apply_perturbation(self):
        magnitude = float(np.random.uniform(50, 200))
        angle = float(np.random.uniform(0, 2 * np.pi))
        fx = magnitude * np.cos(angle)
        fy = magnitude * np.sin(angle)
        try:
            tid = self._physics.model.name2id("torso", "body")
            self._physics.data.xfrc_applied[tid, 0] = fx
            self._physics.data.xfrc_applied[tid, 1] = fy
        except Exception:
            self._physics.data.xfrc_applied[1, 0] = fx
            self._physics.data.xfrc_applied[1, 1] = fy
        self._perturbation_log.append((self._ep_step, magnitude, angle))

    def _clear_perturbation(self):
        try:
            tid = self._physics.model.name2id("torso", "body")
            self._physics.data.xfrc_applied[tid, :] = 0.0
        except Exception:
            self._physics.data.xfrc_applied[1, :] = 0.0

    def _schedule_next_perturbation(self):
        self._next_perturb_step = self._ep_step + int(np.random.randint(100, 301))

    def set_global_step(self, step: int):
        """Called by training script to sync curriculum counter."""
        self._global_step = step
