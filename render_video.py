"""
Render demo videos of the trained recovery policy.
Usage: python render_video.py [--model recovery_policy.zip] [--out recovery_demo.mp4]

Produces three files:
  recovery_demo.mp4    — full continuous side-by-side (walker | trained)
  highlight_clip.mp4   — best recovery per magnitude tier cut from full demo
  challenge_video.mp4  — isolated per-magnitude challenges, both robots reset fresh
"""

import argparse
import math
import os
import numpy as np
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from gymnasium.wrappers import TimeLimit

from recovery_env import QuadrupedRecoveryEnv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_EPISODES            = 3
FPS                   = 60
HIGHLIGHT_DURATION_S  = 30
PUSH_LABEL_FRAMES     = 90    # frames (~1.5s) to show push overlay
HIGHLIGHT_WINDOW_S    = 3     # seconds before/after push in highlight reel
RECOVERY_WINDOW_STEPS = 180   # steps after push to measure peak tilt for scoring

# Escalating magnitudes — start at training range, push beyond to show robustness
PERTURB_INTERVALS  = [150, 350, 550, 750, 950]
PERTURB_MAGNITUDES = [100.0, 150.0, 200.0, 250.0, 300.0]

# Challenge video config
CHALLENGE_INTRO_STEPS    = 90   # steps of clean walking before push (~1.5s)
CHALLENGE_RECOVERY_STEPS = 180  # steps to record after push (~3s)
CHALLENGE_TITLE_FRAMES   = 60   # black title card frames between challenges (~1s)

# Checkpoint paths
WALKER_MODEL   = "recovery_1000000_steps.zip"
WALKER_VECNORM = "recovery_vecnormalize_1000000_steps.pkl"

# Panel labels
LABEL_WALKER  = "WALKER ONLY  (1M steps)"
LABEL_TRAINED = "PERTURBATION TRAINED  (30M steps)"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     default="recovery_policy.zip")
    p.add_argument("--vecnorm",   default="recovery_policy_vecnorm.pkl")
    p.add_argument("--out",       default="recovery_demo.mp4")
    p.add_argument("--highlight", default="highlight_clip.mp4")
    p.add_argument("--challenge", default="challenge_video.mp4")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------

def apply_fixed_perturbation(env_unwrapped, magnitude, angle):
    fx = magnitude * math.cos(angle)
    fy = magnitude * math.sin(angle)
    phys = env_unwrapped._physics
    try:
        tid = phys.model.name2id("torso", "body")
        phys.data.xfrc_applied[tid, 0] = fx
        phys.data.xfrc_applied[tid, 1] = fy
    except Exception:
        phys.data.xfrc_applied[1, 0] = fx
        phys.data.xfrc_applied[1, 1] = fy


# ---------------------------------------------------------------------------
# Frame annotation helpers
# ---------------------------------------------------------------------------

def draw_push_label(frame: np.ndarray, magnitude: float) -> np.ndarray:
    frame = frame.copy()
    text  = f"PUSH  {int(magnitude)}N"
    font  = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.3
    thick = 2
    (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
    x = (frame.shape[1] - tw) // 2
    y = 60
    cv2.putText(frame, text, (x + 2, y + 2), font, scale, (0, 0, 0),   thick + 2)
    cv2.putText(frame, text, (x,     y),     font, scale, (0, 80, 255), thick)
    return frame


def draw_compass(frame: np.ndarray, push_angle: float) -> np.ndarray:
    frame  = frame.copy()
    h, w   = frame.shape[:2]
    cx, cy = w - 70, h - 70
    r      = 45

    cv2.circle(frame, (cx, cy), r + 4, (0, 0, 0),      -1)
    cv2.circle(frame, (cx, cy), r,     (40, 40, 40),    -1)
    cv2.circle(frame, (cx, cy), r,     (120, 120, 120),  2)

    for deg in range(0, 360, 45):
        rad = math.radians(deg)
        ix = int(cx + (r - 6) * math.sin(rad))
        iy = int(cy - (r - 6) * math.cos(rad))
        ox = int(cx + r       * math.sin(rad))
        oy = int(cy - r       * math.cos(rad))
        cv2.line(frame, (ix, iy), (ox, oy), (160, 160, 160), 1)

    cv2.putText(frame, "N", (cx - 7, cy - r + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    CAM_AZIMUTH  = math.radians(135)
    screen_angle = push_angle - CAM_AZIMUTH
    arrow_len    = r - 10
    tip_x  = int(cx + arrow_len      * math.sin(screen_angle))
    tip_y  = int(cy - arrow_len      * math.cos(screen_angle))
    tail_x = int(cx - (arrow_len//2) * math.sin(screen_angle))
    tail_y = int(cy + (arrow_len//2) * math.cos(screen_angle))

    cv2.arrowedLine(frame, (tail_x, tail_y), (tip_x, tip_y),
                    (0, 80, 255), 3, tipLength=0.35)
    cv2.putText(frame, "PUSH DIR", (cx - 33, cy - r - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    return frame


def draw_panel_label(frame: np.ndarray, label: str, trained: bool) -> np.ndarray:
    frame = frame.copy()
    font  = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.7
    thick = 1
    color = (50, 205, 50) if trained else (50, 50, 220)
    (tw, _), _ = cv2.getTextSize(label, font, scale, thick)
    x = (frame.shape[1] - tw) // 2
    y = frame.shape[0] - 20
    cv2.putText(frame, label, (x + 1, y + 1), font, scale, (0, 0, 0), thick + 2)
    cv2.putText(frame, label, (x,     y),     font, scale, color,      thick)
    return frame


def make_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    divider = np.full((left.shape[0], 4, 3), 200, dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def make_title_card(width: int, height: int, magnitude: float) -> np.ndarray:
    """Black card with challenge title centred."""
    card = np.zeros((height, width, 3), dtype=np.uint8)
    font  = cv2.FONT_HERSHEY_DUPLEX
    # Main force text
    line1 = f"{int(magnitude)}N CHALLENGE"
    scale1 = 2.0
    thick1 = 3
    (tw, _), _ = cv2.getTextSize(line1, font, scale1, thick1)
    x1 = (width - tw) // 2
    cv2.putText(card, line1, (x1 + 2, height // 2 + 2), font, scale1, (0,  0,  0),   thick1 + 2)
    cv2.putText(card, line1, (x1,     height // 2),     font, scale1, (0, 80, 255),   thick1)
    # Sub-label
    line2 = "Walker Only  vs  Perturbation Trained"
    scale2 = 0.65
    thick2 = 1
    (tw2, _), _ = cv2.getTextSize(line2, font, scale2, thick2)
    x2 = (width - tw2) // 2
    cv2.putText(card, line2, (x2, height // 2 + 40), font, scale2, (180, 180, 180), thick2)
    return card


# ---------------------------------------------------------------------------
# Single-model episode runner
# ---------------------------------------------------------------------------

def collect_episode_frames(model, vec_env, inner_env,
                            perturb_steps, magnitudes, push_angles):
    """Run one full episode, return annotated frames and scored push events."""
    reset_result = vec_env.reset()
    obs  = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    done = False
    step = 0
    push_label_countdown   = 0
    current_push_magnitude = 0.0
    current_push_angle     = 0.0
    frames        = []
    push_events   = []
    tilt_trackers = {}
    perturb_step_set = set(perturb_steps)

    while not done:
        if step in perturb_step_set:
            idx = perturb_steps.index(step)
            current_push_magnitude = magnitudes[idx]
            current_push_angle     = push_angles[step]
            apply_fixed_perturbation(inner_env, current_push_magnitude, current_push_angle)
            push_label_countdown = PUSH_LABEL_FRAMES
            event_idx = len(push_events)
            push_events.append({
                "frame_idx":  len(frames),
                "magnitude":  current_push_magnitude,
                "angle":      current_push_angle,
                "tilt_score": 0.0,
            })
            tilt_trackers[step] = {
                "steps_remaining": RECOVERY_WINDOW_STEPS,
                "peak_tilt":       0.0,
                "event_idx":       event_idx,
            }

        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = vec_env.step(action)
        done = bool(dones[0])

        try:
            qw   = float(inner_env._physics.data.qpos[3])
            tilt = 2.0 * math.acos(max(-1.0, min(1.0, abs(qw))))
        except Exception:
            tilt = 0.0

        for key in list(tilt_trackers.keys()):
            tr = tilt_trackers[key]
            tr["peak_tilt"] = max(tr["peak_tilt"], tilt)
            tr["steps_remaining"] -= 1
            if tr["steps_remaining"] <= 0:
                push_events[tr["event_idx"]]["tilt_score"] = tr["peak_tilt"]
                del tilt_trackers[key]

        frame = inner_env.render()
        if frame is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if push_label_countdown > 0:
                frame_bgr = draw_push_label(frame_bgr, current_push_magnitude)
                frame_bgr = draw_compass(frame_bgr,    current_push_angle)
                push_label_countdown -= 1
            frames.append(frame_bgr)

        step += 1

    for tr in tilt_trackers.values():
        push_events[tr["event_idx"]]["tilt_score"] = tr["peak_tilt"]

    return frames, push_events


# ---------------------------------------------------------------------------
# Challenge runner — isolated reset per magnitude
# ---------------------------------------------------------------------------

def collect_challenge_frames(model, vec_env, inner_env, magnitude, angle):
    """
    Reset env, walk for CHALLENGE_INTRO_STEPS, apply one push,
    record CHALLENGE_RECOVERY_STEPS. Returns list of annotated frames.
    Episode terminates early on fall — recording continues with last frame frozen.
    """
    reset_result = vec_env.reset()
    obs  = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    frames = []
    push_fired = False
    push_label_countdown = 0
    last_frame = None

    for step in range(CHALLENGE_INTRO_STEPS + CHALLENGE_RECOVERY_STEPS):
        # Fire push at the intro boundary
        if step == CHALLENGE_INTRO_STEPS and not push_fired:
            apply_fixed_perturbation(inner_env, magnitude, angle)
            push_label_countdown = PUSH_LABEL_FRAMES
            push_fired = True

        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = vec_env.step(action)

        frame = inner_env.render()
        if frame is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if push_label_countdown > 0:
                frame_bgr = draw_push_label(frame_bgr, magnitude)
                frame_bgr = draw_compass(frame_bgr,    angle)
                push_label_countdown -= 1
            last_frame = frame_bgr
            frames.append(frame_bgr)
        elif last_frame is not None:
            frames.append(last_frame)

        # If fallen, freeze on last frame for remainder
        if bool(dones[0]):
            while len(frames) < CHALLENGE_INTRO_STEPS + CHALLENGE_RECOVERY_STEPS:
                frames.append(last_frame if last_frame is not None else
                               np.zeros_like(frames[0]))
            break

    return frames


def make_fresh_env(vecnorm_path, model_path, seed=7):
    """Create a brand-new env+vecnorm+model triple to avoid stale state."""
    raw   = QuadrupedRecoveryEnv(render_mode="rgb_array", random_seed=seed)
    raw   = TimeLimit(raw, max_episode_steps=CHALLENGE_INTRO_STEPS + CHALLENGE_RECOVERY_STEPS + 10)
    inner = raw.env
    vec   = DummyVecEnv([lambda: raw])
    vec   = VecNormalize.load(vecnorm_path, vec)
    vec.training    = False
    vec.norm_reward = False
    model = PPO.load(model_path, env=vec)
    return model, vec, inner


def build_challenge_video(trained_model_path, trained_vecnorm_path,
                           walker_model_path,  walker_vecnorm_path,
                           out_path: str, frame_w: int, frame_h: int):
    """
    For each magnitude: title card → fresh env reset → walk → push → reaction.
    Creates new env instances per challenge to avoid stale DummyVecEnv state.
    Side-by-side, walker left / trained right.
    """
    combined_w = frame_w * 2 + 4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (combined_w, frame_h))

    push_angle = math.pi / 2  # lateral — most visible to camera

    for magnitude in PERTURB_MAGNITUDES:
        print(f"  Challenge {int(magnitude)}N ...")

        # Title card
        card = make_title_card(combined_w, frame_h, magnitude)
        for _ in range(CHALLENGE_TITLE_FRAMES):
            writer.write(card)

        # Fresh envs per challenge — avoids Windows DummyVecEnv reset timeout
        t_model, t_vec, t_inner = make_fresh_env(trained_vecnorm_path, trained_model_path)
        w_model, w_vec, w_inner = make_fresh_env(walker_vecnorm_path,  walker_model_path)

        trained_frames = collect_challenge_frames(t_model, t_vec, t_inner, magnitude, push_angle)
        walker_frames  = collect_challenge_frames(w_model, w_vec, w_inner, magnitude, push_angle)

        t_vec.close()
        w_vec.close()

        for t_frame, w_frame in zip(trained_frames, walker_frames):
            left  = draw_panel_label(w_frame, LABEL_WALKER,  trained=False)
            right = draw_panel_label(t_frame, LABEL_TRAINED, trained=True)
            writer.write(make_side_by_side(left, right))

    writer.release()
    print(f"Challenge video -> {out_path}")


# ---------------------------------------------------------------------------
# Highlight builder
# ---------------------------------------------------------------------------

def build_highlight(full_video_path: str, all_push_events: list, fps: int,
                    out_path: str, duration_s: int):
    """Best recovery per magnitude tier from the full demo, ordered light->heavy."""
    tiers = {}
    for event in all_push_events:
        mag = event["magnitude"]
        if mag not in tiers or event["tilt_score"] > tiers[mag]["tilt_score"]:
            tiers[mag] = event

    selected = sorted(tiers.values(), key=lambda e: e["magnitude"])

    print("\nHighlight selection (best recovery per tier):")
    for e in selected:
        print(f"  {int(e['magnitude']):3d}N  tilt={e['tilt_score']:.3f} rad"
              f"  frame={e['frame_idx']}")

    cap          = cv2.VideoCapture(full_video_path)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    window     = fps * HIGHLIGHT_WINDOW_S
    max_frames = fps * duration_s
    written    = 0

    for event in selected:
        if written >= max_frames:
            break
        push_f = event["frame_idx"]
        start  = max(0, push_f - window)
        end    = min(total_frames - 1, push_f + window)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(end - start):
            if written >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            written += 1

    cap.release()
    out.release()
    print(f"Highlight clip -> {out_path}  ({written / fps:.1f}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.exists(WALKER_MODEL):
        raise FileNotFoundError(f"Walker checkpoint not found: {WALKER_MODEL}")

    # Trained model
    trained_raw   = QuadrupedRecoveryEnv(render_mode="rgb_array", random_seed=7)
    trained_raw   = TimeLimit(trained_raw, max_episode_steps=1200)
    trained_inner = trained_raw.env
    trained_vec   = DummyVecEnv([lambda: trained_raw])
    if os.path.exists(args.vecnorm):
        trained_vec = VecNormalize.load(args.vecnorm, trained_vec)
        trained_vec.training    = False
        trained_vec.norm_reward = False
        print("Loaded trained VecNormalize stats.")
    trained_model = PPO.load(args.model, env=trained_vec)
    print(f"Loaded trained model: {args.model}")

    # Walker-only model
    walker_raw   = QuadrupedRecoveryEnv(render_mode="rgb_array", random_seed=7)
    walker_raw   = TimeLimit(walker_raw, max_episode_steps=1200)
    walker_inner = walker_raw.env
    walker_vec   = DummyVecEnv([lambda: walker_raw])
    walker_vec   = VecNormalize.load(WALKER_VECNORM, walker_vec)
    walker_vec.training    = False
    walker_vec.norm_reward = False
    walker_model = PPO.load(WALKER_MODEL, env=walker_vec)
    print(f"Loaded walker model: {WALKER_MODEL}")

    # Frame size
    sample = trained_inner.render()
    h, w   = (480, 640) if sample is None else sample.shape[:2]
    combined_w = w * 2 + 4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, FPS, (combined_w, h))

    all_push_events = []
    frame_offset    = 0
    rng = np.random.default_rng(seed=42)

    # Full continuous demo
    for ep in range(N_EPISODES):
        print(f"\nEpisode {ep + 1}/{N_EPISODES}")

        push_angles = {}
        for i, s in enumerate(PERTURB_INTERVALS):
            base = math.pi / 2 if i % 2 == 0 else 3 * math.pi / 2
            push_angles[s] = base + float(rng.uniform(-0.3, 0.3))

        trained_frames, trained_events = collect_episode_frames(
            trained_model, trained_vec, trained_inner,
            PERTURB_INTERVALS, PERTURB_MAGNITUDES, push_angles,
        )
        walker_frames, _ = collect_episode_frames(
            walker_model, walker_vec, walker_inner,
            PERTURB_INTERVALS, PERTURB_MAGNITUDES, push_angles,
        )

        max_len = max(len(trained_frames), len(walker_frames))
        if len(trained_frames) < max_len:
            trained_frames += [trained_frames[-1]] * (max_len - len(trained_frames))
        if len(walker_frames) < max_len:
            walker_frames  += [walker_frames[-1]]  * (max_len - len(walker_frames))

        for t_frame, wk_frame in zip(trained_frames, walker_frames):
            left  = draw_panel_label(wk_frame, LABEL_WALKER,  trained=False)
            right = draw_panel_label(t_frame,  LABEL_TRAINED, trained=True)
            writer.write(make_side_by_side(left, right))

        for event in trained_events:
            event["frame_idx"] += frame_offset
        all_push_events.extend(trained_events)
        frame_offset += max_len

    writer.release()
    print(f"\nFull demo -> {args.out}")

    # Highlight reel from full demo
    build_highlight(args.out, all_push_events, FPS, args.highlight, HIGHLIGHT_DURATION_S)

    # Challenge video — isolated resets per magnitude, fresh envs to avoid timeout
    print("\nBuilding challenge video...")
    build_challenge_video(
        args.model,    args.vecnorm,
        WALKER_MODEL,  WALKER_VECNORM,
        args.challenge, w, h,
    )


if __name__ == "__main__":
    main()
