from __future__ import annotations

import argparse
import importlib.util
import concurrent.futures
import csv
import json
import os
import random
import shutil
import tempfile
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, List, Tuple, Dict, Any, Sequence

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

from pong import (
    PongEnv,
    simple_tracking_policy,
    Action,
    STAY,
)

try:
    import retro  # type: ignore
except Exception:  # pragma: no cover - optional dep
    retro = None


def _tensorboard_available() -> bool:
    return importlib.util.find_spec("tensorboard") is not None


def _progress_bar_available() -> bool:
    has_tqdm = importlib.util.find_spec("tqdm") is not None
    has_rich = importlib.util.find_spec("rich") is not None
    return has_tqdm and has_rich


@dataclass
class TrainConfig:
    env_kind: str = "pong"
    train_timesteps: int = 300_000
    n_steps: int = 256
    batch_size: int = 512
    n_epochs: int = 4
    gamma: float = 0.99
    learning_rate: float = 2.5e-4
    device: str = "auto"
    target_fps: int = 30
    max_video_seconds: int = 120
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
    top_k_checkpoints: int = 3
    no_checkpoint: bool = False
    individual_videos: bool = False
    cpu_affinity: Optional[str] = None  # e.g. "0,1,2"
    video_dir: str = "videos"
    model_dir: str = "models"
    log_dir: str = "logs"
    metrics_csv: str = "logs/metrics.csv"
    kart_env_id: str = "SuperMarioKart-Snes"
    kart_state: Optional[str] = None
    kart_action_set: str = "drift-lite"
    kart_frame_size: int = 84
    kart_frame_stack: int = 4
    kart_frame_skip: int = 4
    kart_grayscale: bool = True
    kart_use_ram: bool = False
    kart_progress_scale: float = 0.02
    kart_speed_scale: float = 0.001
    kart_lap_bonus: float = 1.0
    kart_offtrack_penalty: float = -0.05
    kart_crash_penalty: float = -0.25
    kart_max_no_progress_steps: int = 600
    kart_video_steps: int = 800
    config_path: Optional[str] = None

    @property
    def max_video_frames(self) -> int:
        return self.target_fps * self.max_video_seconds


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

    def step(self, action: int):
        if self.last_obs is None:
            raise RuntimeError("Call reset() before step().")

        right_action = self.opponent_policy(self.last_obs, is_left=False)
        obs, reward, done, info = self.env.step(action, right_action)
        self.last_obs = obs

        terminated = False  # Episodes end only by truncation (step cap) in PongEnv.
        truncated = bool(done)
        return np.array(obs, dtype=np.float32), float(reward), terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


DEFAULT_KART_BUTTONS: Sequence[str] = (
    "B",
    "Y",
    "Select",
    "Start",
    "Up",
    "Down",
    "Left",
    "Right",
    "A",
    "X",
    "L",
    "R",
)

KART_ACTION_SETS: Dict[str, List[List[str]]] = {
    "drift-lite": [
        [],
        ["B"],  # throttle
        ["B", "Right"],
        ["B", "Left"],
        ["B", "A", "Right"],  # hop + throttle
        ["B", "A", "Left"],
        ["Right"],
        ["Left"],
        ["Y"],  # brake
        ["Y", "Right"],
        ["Y", "Left"],
    ],
    "steer-basic": [
        [],
        ["B"],
        ["B", "Right"],
        ["B", "Left"],
        ["Right"],
        ["Left"],
        ["Y"],
        ["Y", "Right"],
        ["Y", "Left"],
    ],
}


def _build_action_map(buttons: Sequence[str], combos: List[List[str]]) -> List[np.ndarray]:
    action_map: List[np.ndarray] = []
    for combo in combos:
        arr = np.zeros(len(buttons), dtype=np.int8)
        for btn in combo:
            if btn not in buttons:
                continue
            arr[buttons.index(btn)] = 1
        action_map.append(arr)
    return action_map


def _preprocess_kart_frame(frame: np.ndarray, size: int, grayscale: bool = True) -> np.ndarray:
    img = Image.fromarray(frame)
    if grayscale:
        img = img.convert("L")
    img = img.resize((size, size))
    arr = np.array(img, dtype=np.uint8)
    if grayscale:
        return arr  # H x W
    return arr.transpose(2, 0, 1)  # C x H x W if keeping color


class SB3MarioKartEnv(gym.Env):
    """
    Gymnasium wrapper for Retro Mario Kart (SNES) exposing a compact discrete action set
    and lightweight reward shaping for progress and speed.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_id: str = "SuperMarioKart-Snes",
        state: Optional[str] = None,
        action_set: str = "drift-lite",
        frame_size: int = 84,
        frame_stack: int = 4,
        frame_skip: int = 4,
        grayscale: bool = True,
        use_ram_features: bool = False,
        render_mode: Optional[str] = None,
        progress_scale: float = 0.02,
        speed_scale: float = 0.001,
        lap_bonus: float = 1.0,
        offtrack_penalty: float = -0.05,
        crash_penalty: float = -0.25,
        max_no_progress_steps: int = 600,
    ):
        super().__init__()
        if retro is None:
            raise RuntimeError(
                "Gym Retro is required for Mario Kart. Install with `pip install stable-retro` and import the ROM via "
                "`python -m retro.import <path-to-roms>`."
            )
        self.env = retro.make(game=env_id, state=state, render_mode=render_mode)
        self.button_names: Sequence[str] = getattr(self.env, "buttons", DEFAULT_KART_BUTTONS)
        if action_set not in KART_ACTION_SETS:
            raise ValueError(f"Unknown action set '{action_set}'. Available: {list(KART_ACTION_SETS)}")
        self.action_map = _build_action_map(self.button_names, KART_ACTION_SETS[action_set])
        self.frame_size = frame_size
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.grayscale = grayscale
        self.use_ram_features = use_ram_features
        self.frames: deque[np.ndarray] = deque(maxlen=frame_stack)
        self.last_progress = 0.0
        self.last_speed = 0.0
        self.last_lap = 0.0
        self.progress_scale = progress_scale
        self.speed_scale = speed_scale
        self.lap_bonus = lap_bonus
        self.offtrack_penalty = offtrack_penalty
        self.crash_penalty = crash_penalty
        self.max_no_progress_steps = max_no_progress_steps
        self.steps_since_progress = 0
        self.last_obs_raw: Optional[np.ndarray] = None

        channels = 1 if grayscale else 3
        if self.use_ram_features:
            # progress, speed, lap, section/position
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(channels * frame_stack, frame_size, frame_size), dtype=np.float32
            )
        self.action_space = gym.spaces.Discrete(len(self.action_map))

    def _get_progress(self, info: Dict[str, float]) -> float:
        return float(info.get("x", info.get("pos", info.get("position", info.get("trackPos", 0.0)))))

    def _get_speed(self, info: Dict[str, float]) -> float:
        return float(info.get("speed", info.get("vx", info.get("vel", 0.0))))

    def _get_lap(self, info: Dict[str, float]) -> float:
        return float(info.get("lap", info.get("lap_count", info.get("laps", 0.0))))

    def _stack_obs(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(frame)
        while len(self.frames) < self.frame_stack:
            self.frames.append(frame)
        stacked = np.stack(list(self.frames), axis=0).astype(np.float32) / 255.0
        if self.grayscale:
            return stacked
        return stacked.reshape(-1, self.frame_size, self.frame_size)

    def _build_feature_obs(self, progress: float, speed: float, info: Dict[str, float]) -> np.ndarray:
        section = float(info.get("section", info.get("section_idx", 0.0)))
        lap = self._get_lap(info)
        features = np.array(
            [
                progress / 1000.0,
                speed / 120.0,
                lap / 10.0,
                section / 20.0,
            ],
            dtype=np.float32,
        )
        return np.clip(features, 0.0, 1.0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            set_random_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        reset_out = self.env.reset(seed=seed)
        if isinstance(reset_out, tuple):
            raw_obs, info = reset_out
        else:
            raw_obs, info = reset_out, {}
        frame = _preprocess_kart_frame(raw_obs, self.frame_size, grayscale=self.grayscale)
        self.frames.clear()
        self.last_progress = 0.0
        self.last_speed = 0.0
        self.last_lap = 0.0
        self.steps_since_progress = 0
        self.last_obs_raw = raw_obs
        obs = self._build_feature_obs(0.0, 0.0, info) if self.use_ram_features else self._stack_obs(frame)
        return obs, info

    def step(self, action: int):
        raw_action = self.action_map[int(action)]
        total_reward = 0.0
        info: Dict[str, float] = {}
        done = False
        for _ in range(self.frame_skip):
            step_out = self.env.step(raw_action)
            if len(step_out) == 5:
                raw_obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                raw_obs, reward, done, info = step_out
            total_reward += float(reward)
            if done:
                break

        progress = self._get_progress(info)
        speed = self._get_speed(info)
        lap = self._get_lap(info)
        offtrack = bool(info.get("offroad", info.get("off_track", info.get("grass", False))))
        crashed = bool(info.get("crash", info.get("collision", False)))

        shaped_reward = total_reward
        delta_progress = progress - self.last_progress
        delta_speed = speed - self.last_speed
        if delta_progress > 0:
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1
        shaped_reward += delta_progress * self.progress_scale
        shaped_reward += speed * self.speed_scale
        if lap > self.last_lap:
            shaped_reward += self.lap_bonus
        if offtrack:
            shaped_reward += self.offtrack_penalty
        if crashed:
            shaped_reward += self.crash_penalty

        self.last_progress = progress
        self.last_speed = speed
        self.last_lap = lap
        self.last_obs_raw = raw_obs

        frame = _preprocess_kart_frame(raw_obs, self.frame_size, grayscale=self.grayscale)
        obs = self._build_feature_obs(progress, speed, info) if self.use_ram_features else self._stack_obs(frame)
        terminated = bool(done)
        truncated = self.steps_since_progress >= self.max_no_progress_steps
        info.update(
            {
                "__kart_progress": progress,
                "__kart_speed": speed,
                "__kart_lap": lap,
                "__kart_offtrack": offtrack,
            }
        )
        return obs, float(shaped_reward), terminated, truncated, info

    def render(self):
        if self.last_obs_raw is not None:
            return self.last_obs_raw
        try:
            return self.env.render()
        except Exception:
            return None

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
    cfg: TrainConfig,
    ball_color: Tuple[int, int, int],
    overlay_text: str = "",
) -> Tuple[List[np.ndarray], bool]:
    """
    Roll out a short episode with the trained model and return frames plus a success flag.
    """
    env = _make_env(cfg, render_mode="rgb_array", ball_color=ball_color)
    obs, _ = env.reset()
    frames: List[np.ndarray] = []
    target_size = (320, 192) if cfg.env_kind == "pong" else (320, 240)  # divisible by 16 to keep codecs happy
    success = False
    max_steps = cfg.kart_video_steps if cfg.env_kind == "kart" else 400

    frame = env.render()
    if frame is not None:
        frames.append(_add_overlay(np.array(Image.fromarray(frame).resize(target_size)), overlay_text))

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, terminated, truncated, info = env.step(action)
        if cfg.env_kind == "pong" and rew > 0:
            success = True
        elif cfg.env_kind == "kart" and float(info.get("__kart_progress", 0.0)) > 0.0:
            success = True
        frame = env.render()
        if frame is not None:
            frames.append(_add_overlay(np.array(Image.fromarray(frame).resize(target_size)), overlay_text))
        if terminated or truncated:
            break

    env.close()
    return frames, success


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


def _safe_write_video(frames: List[np.ndarray], path: Path, fps: int) -> bool:
    """
    Write frames to mp4 using ffmpeg if available. Returns True on success.
    """
    if not frames:
        print("No frames to write; skipping video.")
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as iio  # use v2 API for stable ffmpeg handling

        with iio.get_writer(path, format="FFMPEG", fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        return True
    except Exception as exc:
        print(f"Video write failed for {path.name}: {exc}")
        return False


ENV_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "pong": {
        "train_timesteps": 300_000,
        "n_steps": 256,
        "batch_size": 512,
        "iterations_per_set": 6,
        "n_envs": 8,
        "target_fps": 30,
        "max_video_seconds": 120,
    },
    "kart": {
        "train_timesteps": 400_000,
        "n_steps": 256,
        "batch_size": 256,
        "iterations_per_set": 2,
        "n_envs": 4,
        "target_fps": 20,
        "max_video_seconds": 90,
    },
}


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
    return data


def parse_args() -> TrainConfig:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, help="Path to YAML/JSON config to merge.")
    known, remaining = base_parser.parse_known_args()
    file_cfg = _load_config_file(known.config)
    env_kind = (file_cfg.get("env_kind") or TrainConfig.env_kind).lower()
    env_defaults = ENV_DEFAULTS.get(env_kind, ENV_DEFAULTS["pong"])

    parser = argparse.ArgumentParser(description="Train PPO agents for Pong or Mario Kart.", parents=[base_parser])
    parser.add_argument("--env-kind", type=str, choices=["pong", "kart"], default=env_kind)
    parser.add_argument("--train-timesteps", type=int, default=file_cfg.get("train_timesteps", env_defaults["train_timesteps"]))
    parser.add_argument("--n-steps", type=int, default=file_cfg.get("n_steps", env_defaults["n_steps"]))
    parser.add_argument("--batch-size", type=int, default=file_cfg.get("batch_size", env_defaults["batch_size"]))
    parser.add_argument("--n-epochs", type=int, default=file_cfg.get("n_epochs", TrainConfig.n_epochs))
    parser.add_argument("--gamma", type=float, default=file_cfg.get("gamma", TrainConfig.gamma))
    parser.add_argument("--learning-rate", type=float, default=file_cfg.get("learning_rate", TrainConfig.learning_rate))
    parser.add_argument("--device", type=str, default=file_cfg.get("device", TrainConfig.device))
    parser.add_argument("--target-fps", type=int, default=file_cfg.get("target_fps", env_defaults["target_fps"]))
    parser.add_argument("--max-video-seconds", type=int, default=file_cfg.get("max_video_seconds", env_defaults["max_video_seconds"]))
    parser.add_argument("--max-cycles", type=int, default=file_cfg.get("max_cycles", TrainConfig.max_cycles))
    parser.add_argument("--checkpoint-interval", type=int, default=file_cfg.get("checkpoint_interval", TrainConfig.checkpoint_interval))
    parser.add_argument("--iterations-per-set", type=int, default=file_cfg.get("iterations_per_set", env_defaults["iterations_per_set"]))
    parser.add_argument("--n-envs", type=int, default=file_cfg.get("n_envs", env_defaults["n_envs"]))
    parser.add_argument("--seed", type=int, default=file_cfg.get("seed", TrainConfig.seed))
    parser.add_argument("--base-seed", type=int, default=file_cfg.get("base_seed", TrainConfig.base_seed))
    parser.add_argument("--deterministic", action="store_true", default=file_cfg.get("deterministic", TrainConfig.deterministic))
    parser.add_argument("--early-stop-patience", type=int, default=file_cfg.get("early_stop_patience", TrainConfig.early_stop_patience))
    parser.add_argument("--improvement-threshold", type=float, default=file_cfg.get("improvement_threshold", TrainConfig.improvement_threshold))
    parser.add_argument("--eval-episodes", type=int, default=file_cfg.get("eval_episodes", TrainConfig.eval_episodes))
    parser.add_argument("--top-k-checkpoints", type=int, default=file_cfg.get("top_k_checkpoints", TrainConfig.top_k_checkpoints))
    parser.add_argument("--no-checkpoint", action="store_true", default=file_cfg.get("no_checkpoint", TrainConfig.no_checkpoint))
    parser.add_argument("--individual-videos", action="store_true", default=file_cfg.get("individual_videos", TrainConfig.individual_videos))
    parser.add_argument("--cpu-affinity", type=str, default=file_cfg.get("cpu_affinity", TrainConfig.cpu_affinity))
    parser.add_argument("--video-dir", type=str, default=file_cfg.get("video_dir", TrainConfig.video_dir))
    parser.add_argument("--model-dir", type=str, default=file_cfg.get("model_dir", TrainConfig.model_dir))
    parser.add_argument("--log-dir", type=str, default=file_cfg.get("log_dir", TrainConfig.log_dir))
    parser.add_argument("--metrics-csv", type=str, default=file_cfg.get("metrics_csv", TrainConfig.metrics_csv))
    parser.add_argument("--kart-env-id", type=str, default=file_cfg.get("kart_env_id", TrainConfig.kart_env_id))
    parser.add_argument("--kart-state", type=str, default=file_cfg.get("kart_state", TrainConfig.kart_state))
    parser.add_argument(
        "--kart-action-set",
        type=str,
        choices=list(KART_ACTION_SETS),
        default=file_cfg.get("kart_action_set", TrainConfig.kart_action_set),
    )
    parser.add_argument("--kart-frame-size", type=int, default=file_cfg.get("kart_frame_size", TrainConfig.kart_frame_size))
    parser.add_argument("--kart-frame-stack", type=int, default=file_cfg.get("kart_frame_stack", TrainConfig.kart_frame_stack))
    parser.add_argument("--kart-frame-skip", type=int, default=file_cfg.get("kart_frame_skip", TrainConfig.kart_frame_skip))
    parser.add_argument("--kart-no-grayscale", action="store_true", default=file_cfg.get("kart_no_grayscale", False))
    parser.add_argument("--kart-use-ram", action="store_true", default=file_cfg.get("kart_use_ram", TrainConfig.kart_use_ram))
    parser.add_argument(
        "--kart-progress-scale",
        type=float,
        default=file_cfg.get("kart_progress_scale", TrainConfig.kart_progress_scale),
    )
    parser.add_argument(
        "--kart-speed-scale",
        type=float,
        default=file_cfg.get("kart_speed_scale", TrainConfig.kart_speed_scale),
    )
    parser.add_argument("--kart-lap-bonus", type=float, default=file_cfg.get("kart_lap_bonus", TrainConfig.kart_lap_bonus))
    parser.add_argument(
        "--kart-offtrack-penalty",
        type=float,
        default=file_cfg.get("kart_offtrack_penalty", TrainConfig.kart_offtrack_penalty),
    )
    parser.add_argument(
        "--kart-crash-penalty",
        type=float,
        default=file_cfg.get("kart_crash_penalty", TrainConfig.kart_crash_penalty),
    )
    parser.add_argument(
        "--kart-max-no-progress-steps",
        type=int,
        default=file_cfg.get("kart_max_no_progress_steps", TrainConfig.kart_max_no_progress_steps),
    )
    parser.add_argument(
        "--kart-video-steps",
        type=int,
        default=file_cfg.get("kart_video_steps", TrainConfig.kart_video_steps),
    )
    args = parser.parse_args(remaining)
    cfg = TrainConfig(
        env_kind=args.env_kind,
        train_timesteps=args.train_timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        device=args.device,
        target_fps=args.target_fps,
        max_video_seconds=args.max_video_seconds,
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
        top_k_checkpoints=args.top_k_checkpoints,
        no_checkpoint=args.no_checkpoint,
        individual_videos=args.individual_videos,
        cpu_affinity=args.cpu_affinity,
        video_dir=args.video_dir,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        metrics_csv=args.metrics_csv,
        kart_env_id=args.kart_env_id,
        kart_state=args.kart_state,
        kart_action_set=args.kart_action_set,
        kart_frame_size=args.kart_frame_size,
        kart_frame_stack=args.kart_frame_stack,
        kart_frame_skip=args.kart_frame_skip,
        kart_grayscale=not args.kart_no_grayscale,
        kart_use_ram=args.kart_use_ram,
        kart_progress_scale=args.kart_progress_scale,
        kart_speed_scale=args.kart_speed_scale,
        kart_lap_bonus=args.kart_lap_bonus,
        kart_offtrack_penalty=args.kart_offtrack_penalty,
        kart_crash_penalty=args.kart_crash_penalty,
        kart_max_no_progress_steps=args.kart_max_no_progress_steps,
        kart_video_steps=args.kart_video_steps,
        config_path=known.config,
    )
    return cfg


def _make_env(cfg: TrainConfig, render_mode: Optional[str] = None, ball_color: Optional[Tuple[int, int, int]] = None):
    if cfg.env_kind == "kart":
        return SB3MarioKartEnv(
            env_id=cfg.kart_env_id,
            state=cfg.kart_state,
            action_set=cfg.kart_action_set,
            frame_size=cfg.kart_frame_size,
            frame_stack=cfg.kart_frame_stack,
            frame_skip=cfg.kart_frame_skip,
            grayscale=cfg.kart_grayscale,
            use_ram_features=cfg.kart_use_ram,
            render_mode=render_mode,
            progress_scale=cfg.kart_progress_scale,
            speed_scale=cfg.kart_speed_scale,
            lap_bonus=cfg.kart_lap_bonus,
            offtrack_penalty=cfg.kart_offtrack_penalty,
            crash_penalty=cfg.kart_crash_penalty,
            max_no_progress_steps=cfg.kart_max_no_progress_steps,
        )
    return SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode=render_mode, ball_color=ball_color)


def evaluate_model(model: PPO, cfg: TrainConfig) -> Dict[str, float]:
    if cfg.env_kind == "kart":
        eval_env = _make_env(cfg, render_mode=None)
        rewards: List[float] = []
        max_progress: List[float] = []
        max_speed: List[float] = []
        laps: List[float] = []
        for _ in range(cfg.eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            ep_rew = 0.0
            ep_progress = 0.0
            ep_speed = 0.0
            ep_lap = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rew, terminated, truncated, info = eval_env.step(action)
                ep_rew += rew
                ep_progress = max(ep_progress, float(info.get("__kart_progress", 0.0)))
                ep_speed = max(ep_speed, float(info.get("__kart_speed", 0.0)))
                ep_lap = max(ep_lap, float(info.get("__kart_lap", 0.0)))
                done = terminated or truncated
            rewards.append(ep_rew)
            max_progress.append(ep_progress)
            max_speed.append(ep_speed)
            laps.append(ep_lap)
        eval_env.close()
        return {
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "reward_std": float(np.std(rewards)) if rewards else 0.0,
            "avg_progress": float(np.mean(max_progress)) if max_progress else 0.0,
            "avg_speed": float(np.mean(max_speed)) if max_speed else 0.0,
            "avg_lap": float(np.mean(laps)) if laps else 0.0,
        }

    eval_env = _make_env(cfg, render_mode=None)
    rewards: List[float] = []
    returns: List[int] = []
    rally_lengths: List[int] = []
    wins = 0
    for _ in range(cfg.eval_episodes):
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
            action, _ = model.predict(obs, deterministic=True)
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

    def _ci(arr: List[float]) -> float:
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
        "win_rate": wins / cfg.eval_episodes if cfg.eval_episodes else 0.0,
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

    if cfg.cpu_affinity:
        try:
            cpu_ids = [int(x) for x in cfg.cpu_affinity.split(",") if x.strip()]
            if hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, cpu_ids)
        except Exception:
            pass

    def _make_env_with_retry(attempts: int = 3):
        last_err = None
        for _ in range(attempts):
            try:
                return make_vec_env(lambda: _make_env(cfg, render_mode=None), n_envs=cfg.n_envs, seed=seed)
            except Exception as exc:  # pragma: no cover - only on flaky init
                last_err = exc
                time.sleep(0.5)
        raise last_err

    env = _make_env_with_retry()
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    latest_path = os.path.join(cfg.model_dir, f"{model_id}_latest.zip")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_log = cfg.log_dir if _tensorboard_available() else None
    use_progress_bar = _progress_bar_available()

    if os.path.exists(latest_path):
        print(f"[{model_id}] Loading existing model from {latest_path} to continue training...")
        model = PPO.load(latest_path, env=env, device=cfg.device)
        model.tensorboard_log = tb_log
    else:
        print(f"[{model_id}] No existing model found; starting a fresh PPO model...")
        policy = "CnnPolicy" if cfg.env_kind == "kart" and not cfg.kart_use_ram else "MlpPolicy"
        model = PPO(
            policy,
            env,
            verbose=1,
            tensorboard_log=tb_log,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            learning_rate=cfg.learning_rate,
            device=cfg.device,
        )

    model.learn(total_timesteps=cfg.train_timesteps, reset_num_timesteps=False, progress_bar=use_progress_bar)

    stamped_model_path: Optional[str] = None
    if cfg.checkpoint_interval > 0 and not cfg.no_checkpoint:
        stamped_model_path = os.path.join(cfg.model_dir, f"{model_id}_{timestamp}.zip")
        model.save(stamped_model_path)
        print(f"[{model_id}] Saved timestamped model: {stamped_model_path}")

    model.save(latest_path)
    print(f"[{model_id}] Updated latest model: {latest_path}")

    metrics = evaluate_model(model, cfg)
    print(f"[{model_id}] Avg reward over {cfg.eval_episodes} eval episodes: {metrics['avg_reward']:.3f}")
    overlay_parts = [model_id, f"r {metrics['avg_reward']:.2f}"]
    if cfg.env_kind == "pong":
        overlay_parts.append(f"win {metrics.get('win_rate', 0.0):.2f}")
    else:
        overlay_parts.append(f"prog {metrics.get('avg_progress', 0.0):.0f}")
    segment, ponged = record_video_segment(model, cfg=cfg, ball_color=color, overlay_text=" | ".join(overlay_parts))
    env.close()
    return model_id, metrics, segment, ponged, timestamp, latest_path, stamped_model_path


def main():
    cfg = parse_args()
    base_seed = cfg.base_seed or cfg.seed
    set_random_seed(base_seed)
    random.seed(base_seed)
    np.random.seed(base_seed)

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.video_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(cfg.log_dir, f"train_run_{run_timestamp}.jsonl")
    metrics_csv_path = Path(cfg.metrics_csv)
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "config", **asdict(cfg), "resolved_at": run_timestamp}) + "\n")

    model_ids = [f"ppo_pong_custom_{i}" for i in range(cfg.iterations_per_set)]
    ball_colors = [
        (255, 0, 0),
        (0, 200, 255),
        (255, 200, 0),
        (0, 255, 120),
    ]

    failure_detected = False
    best_score = float("-inf")
    no_improve_cycles = 0
    max_video_frames = cfg.max_video_frames
    top_checkpoints: List[Tuple[float, str]] = []

    cycle = 0
    while not failure_detected and cycle < cfg.max_cycles:
        combined_frames_per_model: List[List[np.ndarray]] = []
        segments_by_model: Dict[str, List[np.ndarray]] = {}
        all_grid_frames: List[np.ndarray] = []
        scores: List[Tuple[str, float]] = []
        metrics_list: List[Tuple[str, Dict[str, float], str]] = []
        success_flags: List[bool] = []
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        cycle += 1
        print(f"\n=== Cycle {cycle} ===")
        start_cycle = time.time()

        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.iterations_per_set) as executor:
            futures = []
            for idx, model_id in enumerate(model_ids):
                color = ball_colors[idx % len(ball_colors)]
                derived_seed = base_seed + idx + cycle
                futures.append(
                    executor.submit(
                        _train_single,
                        model_id=model_id,
                        color=color,
                        cfg=cfg,
                        seed=derived_seed,
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                model_id, metrics, segment, ponged, stamp, latest_path, stamped_path = future.result()
                scores.append((model_id, metrics["avg_reward"]))
                metrics_list.append((model_id, metrics, latest_path))
                if stamped_path:
                    metrics_list[-1] = (model_id, metrics, stamped_path)
                if segment:
                    combined_frames_per_model.append(segment)
                    segments_by_model[model_id] = segment
                success_flags.append(ponged)
                timestamp = stamp  # use last reported for video naming
                print(f"[{model_id}] Added {len(segment)} frames; success={ponged}; seed={derived_seed}")

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
            metric_fields = (
                ["avg_reward", "reward_std", "avg_progress", "avg_speed", "avg_lap"]
                if cfg.env_kind == "kart"
                else ["avg_reward", "avg_reward_ci", "win_rate", "avg_return_rate", "avg_return_rate_ci", "avg_rally_length"]
            )
            with metrics_csv_path.open("a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=[
                        "cycle",
                        "model_id",
                        *metric_fields,
                        "timestamp",
                    ],
                )
                if not metrics_csv_exists:
                    writer.writeheader()
                for model_id, metrics, _ in metrics_list:
                    writer.writerow(
                        {
                            "cycle": cycle,
                            "model_id": model_id,
                            **metrics,
                            "timestamp": timestamp,
                        }
                    )

            if best_score_cycle > best_score + cfg.improvement_threshold:
                best_score = best_score_cycle
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
        else:
            print("No scores recorded; cannot propagate best model.")

        if success_flags and not any(success_flags):
            print("Failure detected: no successful rollouts this cycle.")
            failure_detected = True
        if no_improve_cycles >= cfg.early_stop_patience:
            print(f"No improvement for {no_improve_cycles} cycles; stopping early.")
            break

        if all_grid_frames:
            combined_video_path = Path(cfg.video_dir) / f"ppo_pong_combined_{timestamp}.mp4"
            if _safe_write_video(all_grid_frames, combined_video_path, cfg.target_fps):
                print(f"Saved combined video with all models in a grid: {combined_video_path} (frames: {len(all_grid_frames)})")
        else:
            print("No frames captured; combined video not written.")

        if cfg.individual_videos and segments_by_model:
            for model_id, frames in segments_by_model.items():
                if not frames:
                    continue
                indiv_path = Path(cfg.video_dir) / f"{model_id}_{timestamp}.mp4"
                if _safe_write_video(frames, indiv_path, cfg.target_fps):
                    print(f"[{model_id}] Saved individual video: {indiv_path}")

if __name__ == "__main__":
    main()
