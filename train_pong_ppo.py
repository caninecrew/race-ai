import argparse
import concurrent.futures
import csv
import json
import os
import traceback
import random
import shutil
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, List, Sequence, Tuple, Dict, Union, cast

import gymnasium as gym
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dep
    yaml = None



def _tensorboard_available() -> bool:
    """Return True if tensorboard is installed; avoid hard dependency at runtime."""
    try:
        import tensorboard  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def _parse_resolution(res_str: str) -> Tuple[int, int]:
    if "x" not in res_str:
        raise ValueError("Resolution must be in <width>x<height> format.")
    w_str, h_str = res_str.lower().split("x", 1)
    return int(w_str), int(h_str)


def _resolve_affinity_list(cpu_affinity: Optional[str], n_envs: int) -> Optional[List[int]]:
    if not cpu_affinity:
        return None
    if cpu_affinity.lower() == "auto":
        try:
            cpus = list(range(os.cpu_count() or 1))
            stride = max(1, len(cpus) // max(1, n_envs))
            return cpus[::stride] or cpus
        except Exception:
            return None
    try:
        return [int(x) for x in cpu_affinity.split(",") if x.strip()]
    except Exception:
        return None


from pong import (
    PongEnv,
    simple_tracking_policy,
    Action,
    STAY,
)


class SB3PongEnv(gym.Env):
    """
    Gymnasium wrapper around the custom PongEnv.
    The learning agent controls the left paddle; the right paddle uses a fixed policy.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        opponent_policy: Optional[Callable[[tuple, bool], Action]] = None,
        render_mode: Optional[str] = None,
        ball_color: Optional[Tuple[int, int, int]] = None,
    ):
        super().__init__()
        self.env = PongEnv(render_mode=render_mode, ball_color=ball_color)
        self.opponent_policy = opponent_policy or (lambda obs, is_left: STAY)
        self.last_obs: Optional[tuple] = None

        # Observations are normalized: [bx, by, bvx, bvy, ly, ry]
        low = np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
        obs, info = self.env.reset()
        self.last_obs = obs
        return np.array(obs, dtype=np.float32), info

    def step(self, action: Union[int, np.integer, np.ndarray]):
        if self.last_obs is None:
            raise RuntimeError("Call reset() before step().")

        action_int = int(action.item()) if isinstance(action, np.ndarray) else int(action)
        right_action = self.opponent_policy(self.last_obs, False)
        obs, reward, done, info = self.env.step(action_int, right_action)
        self.last_obs = obs

        terminated = False  # Episodes end only by truncation (step cap) in PongEnv.
        truncated = bool(done)
        return np.array(obs, dtype=np.float32), float(reward), terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def _add_overlay(frame: np.ndarray, text: str) -> np.ndarray:
    if not text:
        return frame
    img = Image.fromarray(frame)
    drawer = ImageDraw.Draw(img)
    drawer.rectangle([(0, 0), (img.width, 26)], fill=(0, 0, 0))
    drawer.text((6, 6), text, fill=(255, 255, 255))
    return np.array(img)


def record_video_segment(
    model: PPO,
    ball_color: Tuple[int, int, int],
    steps: int = 400,
    overlay_text: str = "",
    resolution: Tuple[int, int] = (320, 192),
) -> Tuple[List[np.ndarray], bool]:
    """
    Roll out a short episode with the trained model and return frames plus whether the ball was successfully returned.
    """
    env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode="rgb_array", ball_color=ball_color)
    obs, _ = env.reset()
    frames: List[np.ndarray] = []
    target_size = resolution  # divisible by 16 to keep codecs happy
    ponged = False

    frame = env.render()
    if frame is not None:
        frames.append(_add_overlay(np.array(Image.fromarray(frame).resize(target_size)), overlay_text))

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, terminated, truncated, _ = env.step(action)
        if rew > 0:
            ponged = True
        frame = env.render()
        if frame is not None:
            frames.append(_add_overlay(np.array(Image.fromarray(frame).resize(target_size)), overlay_text))
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    return frames, ponged


def build_grid_frames(segments: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    Arrange per-model segments into a grid per timestep.
    """
    if not segments or not any(segments):
        return []

    max_len = max(len(seg) for seg in segments)
    num_models = len(segments)
    cols = int(np.ceil(np.sqrt(num_models)))
    rows = int(np.ceil(num_models / cols))

    placeholder = None
    for seg in segments:
        if seg:
            placeholder = np.zeros_like(seg[0])
            break

    grid_frames: List[np.ndarray] = []

    for i in range(max_len):
        row_images = []
        for r in range(rows):
            row_tiles = []
            for c in range(cols):
                idx = r * cols + c
                if idx >= num_models:
                    continue
                seg = segments[idx]
                if seg:
                    if i < len(seg):
                        row_tiles.append(seg[i])
                    else:
                        row_tiles.append(seg[-1])  # hold last frame if shorter
                elif placeholder is not None:
                    row_tiles.append(placeholder)
            if row_tiles:
                row_images.append(np.concatenate(row_tiles, axis=1))
        if row_images:
            grid_frame = np.concatenate(row_images, axis=0)
            grid_frames.append(grid_frame)

    return grid_frames


def _safe_write_video(frames: List[np.ndarray], path: Path, fps: int, final_overlay: str = "") -> bool:
    """
    Write frames to mp4 using ffmpeg if available. Returns True on success.
    """
    if not frames:
        print("No frames to write; skipping video.")
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as iio  # use v2 API for stable ffmpeg handling

        writer = cast(Any, iio.get_writer(path, format="ffmpeg", fps=fps))  # type: ignore[arg-type]
        with writer:
            for idx, frame in enumerate(frames):
                if final_overlay and idx == len(frames) - 1:
                    frame = _add_overlay(frame, final_overlay)
                writer.append_data(frame)
        return True
    except Exception as exc:
        print(f"Video write failed for {path.name}: {exc}")
        return False


@dataclass
class TrainConfig:
    train_timesteps: int = 300_000
    n_steps: int = 256
    batch_size: int = 512
    n_epochs: int = 4
    gamma: float = 0.99
    learning_rate: float = 2.5e-4
    device: str = "auto"
    target_fps: int = 30
    max_video_seconds: int = 30  # total seconds per cycle video
    video_steps: int = 600  # total frames per cycle video
    max_cycles: int = 1
    checkpoint_interval: int = 1  # cycles between timestamped checkpoints
    iterations_per_set: int = 6  # how many parallel model lines to train each cycle
    n_envs: int = 8  # vectorized envs per PPO learner
    seed: int = 0
    deterministic: bool = False
    base_seed: int = 0
    early_stop_patience: int = 3
    improvement_threshold: float = 0.05
    eval_episodes: int = 3
    eval_video_steps: int = 0
    long_eval_video_steps: int = 0
    top_k_checkpoints: int = 3
    no_checkpoint: bool = False
    individual_videos: bool = False
    cpu_affinity: Optional[str] = None  # e.g. "0,1,2" or "auto"
    num_threads: Optional[int] = None
    video_dir: str = "videos"
    model_dir: str = "models"
    log_dir: str = "logs"
    metrics_csv: str = "logs/metrics.csv"
    metrics_deltas: bool = True
    stream_tensorboard: bool = False
    status: bool = False
    resume_from: Optional[str] = None
    config_path: Optional[str] = None
    profile: Optional[str] = None
    dry_run: bool = False
    video_resolution: str = "320x192"
    eval_overlay: bool = True
    eval_deterministic: bool = False
    worker_watchdog: bool = True

    @property
    def max_video_frames(self) -> int:
        return self.target_fps * self.max_video_seconds


def _load_config_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if cfg_path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAML not installed; cannot read YAML configs.")
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    else:
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping of keys to values.")
    defaults = data.get("defaults", data)
    profiles = data.get("profiles", {})
    if not isinstance(defaults, dict):
        raise ValueError("Config defaults must be a mapping.")
    if profiles and not isinstance(profiles, dict):
        raise ValueError("Config profiles must be a mapping of profile -> overrides.")
    return {"defaults": defaults, "profiles": profiles}


def _merge_profile_config(file_cfg: Dict[str, Any], profile: Optional[str]) -> Dict[str, Any]:
    defaults = file_cfg.get("defaults", {})
    profiles = file_cfg.get("profiles", {})
    merged = dict(defaults)
    if profile and isinstance(profiles, dict) and profile in profiles:
        overrides = profiles[profile]
        if isinstance(overrides, dict):
            merged.update(overrides)
    return merged


def _apply_profile(cfg: TrainConfig) -> None:
    """
    Apply preset overrides for common workflows.
    """
    if not cfg.profile:
        return
    profiles: Dict[str, Dict[str, Any]] = {
        "quick": {
            "train_timesteps": 10_000,
            "iterations_per_set": 1,
            "max_cycles": 1,
            "eval_episodes": 1,
            "video_steps": 300,
            "max_video_seconds": 20,
            "checkpoint_interval": 0,
            "no_checkpoint": True,
        },
        "single": {
            "iterations_per_set": 1,
            "n_envs": 4,
            "eval_episodes": max(1, cfg.eval_episodes),
        },
        "gpu": {
            "device": "cuda",
            "batch_size": max(cfg.batch_size, 1024),
            "n_envs": max(cfg.n_envs, 16),
            "train_timesteps": max(cfg.train_timesteps, 500_000),
        },
    }
    overrides = profiles.get(cfg.profile)
    if overrides:
        for key, value in overrides.items():
            setattr(cfg, key, value)
        print(f"Applied profile '{cfg.profile}' overrides: {overrides}")


_progress_bar_checked = False
_progress_bar_available = False


def _progress_bar_ready(suppress_log: bool = False) -> bool:
    """
    Check if optional progress bar deps are present. Prints a hint only once.
    """
    global _progress_bar_checked, _progress_bar_available
    if _progress_bar_checked:
        return _progress_bar_available
    _progress_bar_checked = True
    try:
        import rich  # type: ignore  # noqa: F401
        import tqdm  # type: ignore  # noqa: F401

        _progress_bar_available = True
    except Exception:
        if not suppress_log:
            print("Progress bar disabled (install 'rich' and 'tqdm' or stable-baselines3[extra] to enable).")
    return _progress_bar_available


def parse_args() -> TrainConfig:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, help="Path to YAML/JSON config to merge.")
    base_parser.add_argument("--profile", choices=["quick", "single", "gpu"], help="Preset overrides for common workflows.")
    known, remaining = base_parser.parse_known_args()
    file_cfg = _load_config_file(known.config)
    defaults_from_file = _merge_profile_config(file_cfg, known.profile)

    parser = argparse.ArgumentParser(description="Train PPO agents for custom Pong.", parents=[base_parser])
    parser.add_argument("--status", action="store_true", default=defaults_from_file.get("status", False), help="Show current best status and exit.")
    parser.add_argument("--train-timesteps", type=int, default=defaults_from_file.get("train_timesteps", TrainConfig.train_timesteps))
    parser.add_argument("--n-steps", type=int, default=defaults_from_file.get("n_steps", TrainConfig.n_steps))
    parser.add_argument("--batch-size", type=int, default=defaults_from_file.get("batch_size", TrainConfig.batch_size))
    parser.add_argument("--n-epochs", type=int, default=defaults_from_file.get("n_epochs", TrainConfig.n_epochs))
    parser.add_argument("--gamma", type=float, default=defaults_from_file.get("gamma", TrainConfig.gamma))
    parser.add_argument("--learning-rate", type=float, default=defaults_from_file.get("learning_rate", TrainConfig.learning_rate))
    parser.add_argument("--device", type=str, default=defaults_from_file.get("device", TrainConfig.device))
    parser.add_argument("--target-fps", type=int, default=defaults_from_file.get("target_fps", TrainConfig.target_fps))
    parser.add_argument("--max-video-seconds", type=int, default=defaults_from_file.get("max_video_seconds", TrainConfig.max_video_seconds))
    parser.add_argument("--video-steps", type=int, default=defaults_from_file.get("video_steps", TrainConfig.video_steps), help="Frames captured per clip; e.g., 1800 ~= 60s at 30 fps, 3600 ~= 2min.")
    parser.add_argument("--max-cycles", type=int, default=defaults_from_file.get("max_cycles", TrainConfig.max_cycles))
    parser.add_argument("--checkpoint-interval", type=int, default=defaults_from_file.get("checkpoint_interval", TrainConfig.checkpoint_interval))
    parser.add_argument("--iterations-per-set", type=int, default=defaults_from_file.get("iterations_per_set", TrainConfig.iterations_per_set))
    parser.add_argument("--n-envs", type=int, default=defaults_from_file.get("n_envs", TrainConfig.n_envs))
    parser.add_argument("--seed", type=int, default=defaults_from_file.get("seed", TrainConfig.seed))
    parser.add_argument("--base-seed", type=int, default=defaults_from_file.get("base_seed", TrainConfig.base_seed))
    parser.add_argument("--deterministic", action="store_true", default=defaults_from_file.get("deterministic", TrainConfig.deterministic))
    parser.add_argument("--early-stop-patience", type=int, default=defaults_from_file.get("early_stop_patience", TrainConfig.early_stop_patience))
    parser.add_argument("--improvement-threshold", type=float, default=defaults_from_file.get("improvement_threshold", TrainConfig.improvement_threshold))
    parser.add_argument("--eval-episodes", type=int, default=defaults_from_file.get("eval_episodes", TrainConfig.eval_episodes))
    parser.add_argument("--eval-video-steps", type=int, default=defaults_from_file.get("eval_video_steps", TrainConfig.eval_video_steps), help="Optional extra eval video steps.")
    parser.add_argument("--long-eval-video-steps", type=int, default=defaults_from_file.get("long_eval_video_steps", TrainConfig.long_eval_video_steps), help="Capture a longer evaluation match separately.")
    parser.add_argument("--top-k-checkpoints", type=int, default=defaults_from_file.get("top_k_checkpoints", TrainConfig.top_k_checkpoints))
    parser.add_argument("--no-checkpoint", action="store_true", default=defaults_from_file.get("no_checkpoint", TrainConfig.no_checkpoint))
    parser.add_argument("--individual-videos", action="store_true", default=defaults_from_file.get("individual_videos", TrainConfig.individual_videos))
    parser.add_argument("--cpu-affinity", type=str, default=defaults_from_file.get("cpu_affinity", TrainConfig.cpu_affinity))
    parser.add_argument("--num-threads", type=int, default=defaults_from_file.get("num_threads", TrainConfig.num_threads) or None, help="Override torch.num_threads.")
    parser.add_argument("--video-dir", type=str, default=defaults_from_file.get("video_dir", TrainConfig.video_dir))
    parser.add_argument("--model-dir", type=str, default=defaults_from_file.get("model_dir", TrainConfig.model_dir))
    parser.add_argument("--log-dir", type=str, default=defaults_from_file.get("log_dir", TrainConfig.log_dir))
    parser.add_argument("--metrics-csv", type=str, default=defaults_from_file.get("metrics_csv", TrainConfig.metrics_csv))
    parser.add_argument("--metrics-deltas", action="store_true", default=defaults_from_file.get("metrics_deltas", TrainConfig.metrics_deltas))
    parser.add_argument("--stream-tensorboard", action="store_true", default=defaults_from_file.get("stream_tensorboard", TrainConfig.stream_tensorboard))
    parser.add_argument("--resume-from", type=str, default=defaults_from_file.get("resume_from", TrainConfig.resume_from), help="Checkpoint to resume from instead of *_latest.")
    parser.add_argument("--video-resolution", type=str, default=defaults_from_file.get("video_resolution", TrainConfig.video_resolution))
    parser.add_argument("--eval-deterministic", action="store_true", default=defaults_from_file.get("eval_deterministic", TrainConfig.eval_deterministic), help="Deterministic eval to reduce variance.")
    parser.add_argument("--watchdog", dest="worker_watchdog", action="store_true", default=defaults_from_file.get("worker_watchdog", TrainConfig.worker_watchdog), help="Keep training when a worker fails.")
    parser.add_argument("--dry-run", action="store_true", help="Parse config, build envs, and exit without training.")
    args = parser.parse_args(remaining)
    cfg = TrainConfig(
        train_timesteps=args.train_timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        device=args.device,
        target_fps=args.target_fps,
        max_video_seconds=args.max_video_seconds,
        video_steps=args.video_steps,
        max_cycles=args.max_cycles,
        checkpoint_interval=args.checkpoint_interval,
        iterations_per_set=args.iterations_per_set,
        n_envs=args.n_envs,
        seed=args.seed,
        base_seed=args.base_seed,
        deterministic=args.deterministic,
        early_stop_patience=args.early_stop_patience,
        improvement_threshold=args.improvement_threshold,
        eval_episodes=args.eval_episodes,
        eval_video_steps=args.eval_video_steps,
        long_eval_video_steps=args.long_eval_video_steps,
        top_k_checkpoints=args.top_k_checkpoints,
        no_checkpoint=args.no_checkpoint,
        individual_videos=args.individual_videos,
        cpu_affinity=args.cpu_affinity,
        num_threads=args.num_threads,
        video_dir=args.video_dir,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        metrics_csv=args.metrics_csv,
        metrics_deltas=args.metrics_deltas,
        stream_tensorboard=args.stream_tensorboard,
        status=args.status,
        resume_from=args.resume_from,
        config_path=known.config,
        profile=args.profile,
        dry_run=args.dry_run,
        video_resolution=args.video_resolution,
        eval_deterministic=args.eval_deterministic,
        worker_watchdog=args.worker_watchdog,
    )
    _apply_profile(cfg)
    if cfg.video_steps > cfg.max_video_frames:
        print(
            f"Warning: video_steps ({cfg.video_steps}) exceeds max frames from max_video_seconds*target_fps ({cfg.max_video_frames}). "
            "Consider lowering --video-steps or raising --max-video-seconds/--target-fps."
        )
    return cfg


def _print_status(cfg: TrainConfig) -> None:
    metrics_path = Path(cfg.metrics_csv)
    if not metrics_path.exists():
        print(f"No metrics CSV found at {metrics_path}; run training first.")
        return
    import csv

    best_row = None
    with metrics_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                reward = float(row.get("avg_reward", 0.0))
            except Exception:
                reward = float("-inf")
            if best_row is None or reward > float(best_row.get("avg_reward", float("-inf"))):
                best_row = row
    if not best_row:
        print("Metrics file is empty; no status to report.")
        return
    print("=== Current Status ===")
    print(f"Best model id: {best_row.get('model_id')} | avg_reward={best_row.get('avg_reward')} | win_rate={best_row.get('win_rate')}")
    print(f"Last recorded at cycle {best_row.get('cycle')} on {best_row.get('timestamp')}")

def evaluate_model(model: PPO, episodes: int, deterministic: bool = True) -> Dict[str, float]:
    eval_env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode=None)
    rewards: List[float] = []
    returns: List[int] = []
    rally_lengths: List[int] = []
    wins = 0
    for _ in range(episodes):
        obs, info = eval_env.reset()
        left_score = info.get("left_score", 0)
        right_score = info.get("right_score", 0)
        done = False
        ep_rew = 0.0
        steps = 0
        last_left = 0
        last_right = 0
        rally_steps = 0
        while not done and steps < eval_env.env.cfg.max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rew, terminated, truncated, info = eval_env.step(action)
            rally_steps += 1
            left_score = info.get("left_score", left_score)
            right_score = info.get("right_score", right_score)
            if left_score != last_left or right_score != last_right:
                rally_lengths.append(rally_steps)
                rally_steps = 0
                last_left, last_right = left_score, right_score
            if rew > 0:
                returns.append(1)
            ep_rew += rew
            steps += 1
            done = terminated or truncated
        rewards.append(ep_rew)
        if left_score > right_score:
            wins += 1
    eval_env.close()

    def _ci(arr: Sequence[Union[float, int]]) -> float:
        if len(arr) < 2:
            return 0.0
        std = float(np.std(arr, ddof=1))
        return 1.96 * std / np.sqrt(len(arr))

    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    avg_return_rate = float(np.mean(returns)) if returns else 0.0
    avg_rally = float(np.mean(rally_lengths)) if rally_lengths else 0.0
    return {
        "avg_reward": avg_reward,
        "avg_reward_ci": _ci(rewards),
        "avg_return_rate": avg_return_rate,
        "avg_return_rate_ci": _ci(returns),
        "avg_rally_length": avg_rally,
        "win_rate": wins / episodes if episodes else 0.0,
    }


def _train_single(
    model_id: str,
    color: Tuple[int, int, int],
    cfg: TrainConfig,
    seed: int,
) -> Tuple[str, Dict[str, float], List[np.ndarray], bool, str, str, Optional[str]]:
    """Train one model line in isolation (separate process-friendly)."""
    set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if cfg.deterministic:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)

    affinity = _resolve_affinity_list(cfg.cpu_affinity, cfg.n_envs)
    if affinity:
        try:
            if hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, affinity)  # type: ignore[attr-defined]
        except Exception:
            pass

    def _make_env_with_retry(attempts: int = 3):
        last_err: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                env = make_vec_env(
                    lambda: SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode=None),
                    n_envs=cfg.n_envs,
                    seed=seed,
                )
                # Prewarm to smooth jitters during initial rollout.
                env.reset()
                try:
                    if hasattr(env, "num_envs"):
                        actions = np.asarray([env.action_space.sample() for _ in range(env.num_envs)])
                    else:
                        actions = np.asarray(env.action_space.sample())
                    env.step(actions)
                except Exception:
                    env.close()
                    raise
                return env
            except Exception as exc:  # pragma: no cover - only on flaky init
                last_err = exc
                time.sleep(0.5 * (attempt + 1))
        if last_err is not None:
            raise last_err
        raise RuntimeError("Failed to create environments but no error captured.")

    env = _make_env_with_retry()
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    latest_path = os.path.join(cfg.model_dir, f"{model_id}_latest.zip")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_log_dir = os.path.join(cfg.log_dir, model_id) if _tensorboard_available() else None
    if tb_log_dir is None:
        print(f"[{model_id}] TensorBoard not installed; disabling tensorboard logging.")

    resume_path = cfg.resume_from or (latest_path if os.path.exists(latest_path) else None)
    if resume_path and os.path.exists(resume_path):
        print(f"[{model_id}] Loading existing model from {resume_path} to continue training...")
        model = PPO.load(resume_path, env=env, device=cfg.device)
        if tb_log_dir is None:
            model.tensorboard_log = None
    else:
        print(f"[{model_id}] No existing model found; starting a fresh PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tb_log_dir,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            learning_rate=cfg.learning_rate,
            device=cfg.device,
        )

    model.learn(
        total_timesteps=cfg.train_timesteps,
        reset_num_timesteps=False,
        progress_bar=_progress_bar_ready(suppress_log=True),
    )

    stamped_model_path: Optional[str] = None
    if cfg.checkpoint_interval > 0 and not cfg.no_checkpoint:
        stamped_model_path = os.path.join(cfg.model_dir, f"{model_id}_{timestamp}.zip")
        model.save(stamped_model_path)
        print(f"[{model_id}] Saved timestamped model: {stamped_model_path}")

    model.save(latest_path)
    print(f"[{model_id}] Updated latest model: {latest_path}")

    metrics = evaluate_model(model, cfg.eval_episodes, deterministic=cfg.eval_deterministic or cfg.deterministic)
    print(f"[{model_id}] Avg reward over {cfg.eval_episodes} eval episodes: {metrics['avg_reward']:.3f}")

    segment, ponged = record_video_segment(
        model,
        ball_color=color,
        steps=cfg.video_steps,
        overlay_text=f"{model_id} | r {metrics['avg_reward']:.2f} | win {metrics['win_rate']:.2f}",
        resolution=_parse_resolution(cfg.video_resolution),
    )
    env.close()
    return model_id, metrics, segment, ponged, timestamp, latest_path, stamped_model_path


def main():
    cfg = parse_args()
    base_seed = cfg.base_seed or cfg.seed
    set_random_seed(base_seed)
    random.seed(base_seed)
    np.random.seed(base_seed)

    print(
        f"Config: profile={cfg.profile or 'none'}, train_steps={cfg.train_timesteps}, "
        f"iters_per_set={cfg.iterations_per_set}, max_cycles={cfg.max_cycles}, "
        f"video_steps={cfg.video_steps} (~{cfg.video_steps / cfg.target_fps:.1f}s @ {cfg.target_fps} fps)"
    )
    if cfg.status:
        _print_status(cfg)
        return
    _progress_bar_ready()

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.video_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(cfg.log_dir, f"train_run_{run_timestamp}.jsonl")
    metrics_csv_path = Path(cfg.metrics_csv)
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    tb_writer = None
    if cfg.stream_tensorboard and _tensorboard_available():
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            tb_writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, "cycle_metrics", run_timestamp))
        except Exception:
            tb_writer = None

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "config", **asdict(cfg), "resolved_at": run_timestamp}) + "\n")
    if cfg.dry_run:
        env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode=None)
        obs, info = env.reset()
        env.close()
        print("Dry run: environment initialized successfully. Exiting before training.")
        return

    model_ids = [f"ppo_pong_custom_{i}" for i in range(cfg.iterations_per_set)]
    ball_colors = [
        (255, 0, 0),
        (0, 200, 255),
        (255, 200, 0),
        (0, 255, 120),
    ]

    failure_detected = False
    best_score = float("-inf")
    best_overall_id: Optional[str] = None
    best_overall_score = float("-inf")
    last_combined_video: Optional[str] = None
    last_eval_video: Optional[str] = None
    no_improve_cycles = 0
    max_video_frames = cfg.max_video_frames
    top_checkpoints: List[Tuple[float, str]] = []
    stop_reason = "max_cycles"
    last_avg_by_model: Dict[str, float] = {}
    cycle_reports: List[Dict[str, Any]] = []
    best_checkpoint_path: Optional[str] = None

    cycle = 0
    while not failure_detected and cycle < cfg.max_cycles:
        combined_frames_per_model: List[List[np.ndarray]] = []
        segments_by_model: Dict[str, List[np.ndarray]] = {}
        all_grid_frames: List[np.ndarray] = []
        scores: List[Tuple[str, float]] = []
        metrics_list: List[Tuple[str, Dict[str, float], str]] = []
        pong_flags: List[bool] = []
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        cycle += 1
        print(f"\n=== Cycle {cycle} / {cfg.max_cycles} ===")
        start_cycle = time.time()

        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.iterations_per_set) as executor:
            futures = []
            seed_by_future: Dict[concurrent.futures.Future, int] = {}
            for idx, model_id in enumerate(model_ids):
                color = ball_colors[idx % len(ball_colors)]
                derived_seed = base_seed + idx + cycle
                future = executor.submit(
                    _train_single,
                    model_id=model_id,
                    color=color,
                    cfg=cfg,
                    seed=derived_seed,
                )
                futures.append(future)
                seed_by_future[future] = derived_seed

            for future in concurrent.futures.as_completed(futures):
                try:
                    model_id, metrics, segment, ponged, stamp, latest_path, stamped_path = future.result()
                except Exception as exc:
                    if cfg.worker_watchdog:
                        print(f"[watchdog] Worker failed: {exc}; continuing without this model.")
                        tb = future.exception()
                        if tb:
                            print(traceback.format_exc())
                        continue
                    raise
                scores.append((model_id, metrics["avg_reward"]))
                metrics_list.append((model_id, metrics, latest_path))
                if stamped_path:
                    metrics_list[-1] = (model_id, metrics, stamped_path)
                if segment:
                    combined_frames_per_model.append(segment)
                    segments_by_model[model_id] = segment
                pong_flags.append(ponged)
                timestamp = stamp  # use last reported for video naming
                seed_used = seed_by_future.get(future, base_seed)
                print(f"[{model_id}] Added {len(segment)} frames; Ponged: {ponged}; seed={seed_used}")

        if combined_frames_per_model and any(combined_frames_per_model):
            grid_frames = build_grid_frames(combined_frames_per_model)
            for frame in grid_frames:
                if len(all_grid_frames) >= max_video_frames:
                    break
                all_grid_frames.append(frame)
            print(f"Accumulated {len(all_grid_frames)} frames toward combined video (max {max_video_frames}).")
        else:
            print("No frames captured this cycle; skipping video accumulation.")

        if scores:
            best_id, best_score_cycle = max(scores, key=lambda t: t[1])
            best_latest = os.path.join(cfg.model_dir, f"{best_id}_latest.zip")
            best_checkpoint_path = next((p for mid, _, p in metrics_list if mid == best_id), best_latest)
            best_metrics = next((metrics for mid, metrics, _ in metrics_list if mid == best_id), {})
            for model_id in model_ids:
                target_latest = os.path.join(cfg.model_dir, f"{model_id}_latest.zip")
                if os.path.exists(best_latest) and best_latest != target_latest:
                    shutil.copy2(best_latest, target_latest)
            # checkpoint pruning
            top_checkpoints.append((best_score_cycle, best_checkpoint_path))
            top_checkpoints = sorted(top_checkpoints, key=lambda t: t[0], reverse=True)
            for idx, (_, path) in enumerate(top_checkpoints):
                if idx >= cfg.top_k_checkpoints and os.path.exists(path) and not path.endswith("_latest.zip"):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            top_checkpoints = top_checkpoints[: cfg.top_k_checkpoints]

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "event": "cycle",
                            "cycle": cycle,
                            "scores": scores,
                            "frames": len(all_grid_frames),
                            "timestamp": timestamp,
                            "seeds": [base_seed + i + cycle for i in range(len(model_ids))],
                        }
                    )
                    + "\n"
                )

            metrics_csv_exists = metrics_csv_path.exists()
            with metrics_csv_path.open("a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=[
                        "cycle",
                        "model_id",
                        "avg_reward",
                        "avg_reward_ci",
                        "win_rate",
                        "avg_return_rate",
                        "avg_return_rate_ci",
                        "avg_rally_length",
                        "delta_reward",
                        "timestamp",
                    ],
                )
                if not metrics_csv_exists:
                    writer.writeheader()
                for model_id, metrics, _ in metrics_list:
                    delta_reward = None
                    if cfg.metrics_deltas and model_id in last_avg_by_model:
                        delta_reward = metrics.get("avg_reward", 0.0) - last_avg_by_model[model_id]
                    last_avg_by_model[model_id] = metrics.get("avg_reward", 0.0)
                    writer.writerow(
                        {
                            "cycle": cycle,
                            "model_id": model_id,
                            **metrics,
                            "delta_reward": delta_reward if delta_reward is not None else "",
                            "timestamp": timestamp,
                        }
                    )

            if best_score_cycle > best_score + cfg.improvement_threshold:
                best_score = best_score_cycle
                best_overall_id = best_id
                best_overall_score = best_score_cycle
                no_improve_cycles = 0
            else:
                no_improve_cycles += 1
            if all_grid_frames:
                annotated = _add_overlay(all_grid_frames[-1], f"Next base model: {best_id}")
                all_grid_frames.append(annotated)
            elapsed = time.time() - start_cycle
            print(
                f"Best model this cycle: {best_id} (avg reward {best_score_cycle:.3f}); "
                f"elapsed {elapsed:.1f}s; no_improve={no_improve_cycles}"
            )
            cycle_reports.append(
                {
                    "cycle": cycle,
                    "scores": scores,
                    "best_id": best_id,
                    "best_score": best_score_cycle,
                    "timestamp": timestamp,
                    "seeds": [base_seed + i + cycle for i in range(len(model_ids))],
                }
            )
        else:
            print("No scores recorded; cannot propagate best model.")

        if pong_flags and not any(pong_flags):
            print("Failure detected: no model returned the ball this cycle.")
            failure_detected = True
            stop_reason = "no_pong"
        if no_improve_cycles >= cfg.early_stop_patience:
            print(f"No improvement for {no_improve_cycles} cycles; stopping early.")
            stop_reason = "early_stop"
            break

        if all_grid_frames:
            overlay_text = ""
            if scores:
                overlay_text = f"Cycle {cycle} best {best_id} | win {best_metrics.get('win_rate',0):.2f} | rally {best_metrics.get('avg_rally_length',0):.1f}"
            combined_video_path = Path(cfg.video_dir) / f"ppo_pong_combined_{timestamp}_seed{base_seed}.mp4"
            if _safe_write_video(all_grid_frames, combined_video_path, cfg.target_fps, final_overlay=overlay_text):
                print(f"Saved combined video with all models in a grid: {combined_video_path} (frames: {len(all_grid_frames)})")
                last_combined_video = str(combined_video_path)
        else:
            print("No frames captured; combined video not written.")

        if cfg.long_eval_video_steps and best_checkpoint_path and os.path.exists(best_checkpoint_path):
            try:
                best_model = PPO.load(best_checkpoint_path, device=cfg.device)
                frames, _ = record_video_segment(
                    best_model,
                    ball_color=(255, 255, 255),
                    steps=cfg.long_eval_video_steps,
                    overlay_text=f"{best_id} long eval",
                    resolution=_parse_resolution(cfg.video_resolution),
                )
                if frames:
                    eval_path = Path(cfg.video_dir) / f"{best_id}_eval_{timestamp}_seed{base_seed}.mp4"
                    if _safe_write_video(frames, eval_path, cfg.target_fps, final_overlay="Long eval clip"):
                        print(f"[{best_id}] Saved extended eval video: {eval_path}")
                        last_eval_video = str(eval_path)
            except Exception as exc:  # pragma: no cover - non-critical
                print(f"Could not record extended eval video: {exc}")

        if cfg.individual_videos and segments_by_model:
            for model_id, frames in segments_by_model.items():
                if not frames:
                    continue
                indiv_path = Path(cfg.video_dir) / f"{model_id}_{timestamp}_seed{base_seed}.mp4"
                if _safe_write_video(frames, indiv_path, cfg.target_fps):
                    print(f"[{model_id}] Saved individual video: {indiv_path}")

        if tb_writer:
            for model_id, metrics, _ in metrics_list:
                tb_writer.add_scalar(f"{model_id}/avg_reward", metrics.get("avg_reward", 0.0), cycle)
                tb_writer.add_scalar(f"{model_id}/win_rate", metrics.get("win_rate", 0.0), cycle)

    if tb_writer:
        tb_writer.close()

    summary = {
        "event": "summary",
        "best_model": best_overall_id,
        "best_score": best_overall_score,
        "metrics_csv": cfg.metrics_csv,
        "video_dir": cfg.video_dir,
        "model_dir": cfg.model_dir,
        "last_combined_video": last_combined_video,
        "last_eval_video": last_eval_video,
        "run_timestamp": run_timestamp,
        "stop_reason": stop_reason,
        "cycles": cycle_reports,
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    report_json = Path(cfg.log_dir) / f"run_report_{run_timestamp}.json"
    report_json.write_text(json.dumps({"config": asdict(cfg), "summary": summary}, indent=2))
    report_html = Path(cfg.log_dir) / f"run_report_{run_timestamp}.html"
    report_html.write_text(
        f"<html><body><h1>Pong PPO Run {run_timestamp}</h1>"
        f"<p>Best model: {best_overall_id or 'n/a'} (avg reward {best_overall_score:.3f})</p>"
        f"<p>Stop reason: {stop_reason}</p>"
        f"<p>Metrics CSV: {cfg.metrics_csv}</p>"
        f"<p>Last combined video: {last_combined_video or 'n/a'}</p>"
        f"<h2>Cycles</h2>"
        + "".join(
            f"<div><strong>Cycle {c['cycle']}</strong>: best {c['best_id']} @ {c['best_score']:.3f} (ts {c['timestamp']})</div>"
            for c in cycle_reports
        )
        + "</body></html>"
    )
    print("\n=== Run Summary ===")
    print(f"Best model: {best_overall_id or 'n/a'} (avg reward {best_overall_score:.3f})")
    print(f"Stop reason: {stop_reason}")
    print(f"Metrics CSV: {cfg.metrics_csv}")
    if last_combined_video:
        print(f"Last combined video: {last_combined_video}")
    print(f"Models dir: {cfg.model_dir} | Videos dir: {cfg.video_dir}")
    if _tensorboard_available():
        print(f"TensorBoard available: run `tensorboard --logdir {cfg.log_dir}`")
    else:
        print("TensorBoard not installed; install tensorboard to view training curves.")

if __name__ == "__main__":
    main()
