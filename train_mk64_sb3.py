from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import gym
import numpy as np

try:
    from mk64_flow_env import register_menu_restricted_env
except Exception:
    register_menu_restricted_env = None

from mk64_common import require_gym_mupen64plus, list_registered_env_ids, cleanup_stale_x_sockets


def require_sb3():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor, VecTransposeImage
    except Exception as exc:
        raise RuntimeError(
            "Stable Baselines3 + torch are required for training. "
            "Install them inside mk64-venv (choose versions compatible with your CUDA/CPU stack)."
        ) from exc

    return PPO, CheckpointCallback, DummyVecEnv, VecFrameStack, VecMonitor, VecTransposeImage


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on Mario Kart 64 (gym_mupen64plus).")
    p.add_argument(
        "--env-id",
        type=str,
        default="Mario-Kart-Menu-Restricted-v0",
        help="Gym env id for a single track (use --list-envs to discover available ids).",
    )
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--frame-skip", type=int, default=1, help="If your env already frameskips, leave at 1.")
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--models-dir", type=str, default="models_mk64")
    p.add_argument("--logs-dir", type=str, default="logs_mk64")
    p.add_argument("--checkpoints", type=int, default=100_000, help="Checkpoint frequency in timesteps.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--progress-bar", action="store_true", help="Enable SB3 progress bar (requires tqdm and rich).")
    p.add_argument("--list-envs", action="store_true", help="List available env ids and exit.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Ensure MK64 training backend is present.
    require_gym_mupen64plus()
    # Clear stale X lock files that can prevent Xvfb from starting.
    cleanup_stale_x_sockets()

    # Register the custom flow env if available so gym.make can resolve it.
    if register_menu_restricted_env is not None:
        try:
            register_menu_restricted_env(args.env_id)
        except Exception:
            # Non-fatal: fall back to whatever is already registered.
            pass

    if args.list_envs:
        envs = list_registered_env_ids(keyword="Mario")
        print("Registered env ids containing 'Mario':")
        for e in envs:
            print("  ", e)
        return 0

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)

    PPO, CheckpointCallback, DummyVecEnv, VecFrameStack, VecMonitor, VecTransposeImage = require_sb3()

    # If the user asked for CUDA but it is unavailable (e.g., no driver/libcuda),
    # fall back to CPU so training can proceed instead of crashing.
    try:
        import torch

        if str(args.device).startswith("cuda") and not torch.cuda.is_available():
            print("CUDA requested but not available; falling back to CPU.")
            args.device = "cpu"
    except Exception:
        # If torch import itself fails, SB3 will raise later; ignore here.
        pass

    def make_env():
        env = gym.make(args.env_id)

        # Some gym_mupen64plus builds (or wrappers) can return (obs, info) on reset
        # or the newer 5-tuple on step; SB3 expects the legacy 4-tuple/obs only.
        original_reset = env.reset
        original_step = env.step

        def _normalize_obs(obs):
            if isinstance(obs, tuple):
                obs = obs[0]
            return np.asarray(obs)

        def _reset(*args, **kwargs):
            obs = original_reset(*args, **kwargs)
            return _normalize_obs(obs)

        def _step(action):
            result = original_step(action)
            if isinstance(result, tuple) and len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
                return _normalize_obs(obs), reward, done, info
            obs, reward, done, info = result
            return _normalize_obs(obs), reward, done, info

        env.reset = _reset
        env.step = _step
        # If your env supports seeding, this will apply; otherwise it is harmless.
        try:
            env.reset(seed=args.seed)
        except Exception:
            pass
        return env

    vec = DummyVecEnv([make_env])
    vec = VecMonitor(vec)

    # If observations are images in HWC (common), transpose to CHW for SB3 CNN.
    # If your env already returns CHW, this still typically works; if not, disable this line.
    vec = VecTransposeImage(vec)

    if args.frame_stack and args.frame_stack > 1:
        vec = VecFrameStack(vec, n_stack=args.frame_stack)

    model = PPO(
        "CnnPolicy",
        vec,
        verbose=1,
        tensorboard_log=args.logs_dir,
        # High-resolution observations make the PPO buffer huge; use smaller rollout length to limit RAM.
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        learning_rate=2.5e-4,
        device=args.device,
        seed=args.seed,
    )

    ckpt = CheckpointCallback(
        save_freq=max(1, args.checkpoints),
        save_path=args.models_dir,
        name_prefix="mk64_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(total_timesteps=args.timesteps, callback=ckpt, progress_bar=args.progress_bar)

    out = Path(args.models_dir) / "mk64_ppo_latest.zip"
    model.save(out)
    print("Saved model:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
