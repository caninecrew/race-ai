import argparse
import concurrent.futures
import csv
import json
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, List, Tuple, Dict, Any

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
) -> Tuple[List[np.ndarray], bool]:
    """
    Roll out a short episode with the trained model and return frames plus whether the ball was successfully returned.
    """
    env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode="rgb_array", ball_color=ball_color)
    obs, _ = env.reset()
    frames: List[np.ndarray] = []
    target_size = (320, 192)  # divisible by 16 to keep codecs happy
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
    config_path: Optional[str] = None

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
    return data


def parse_args() -> TrainConfig:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, help="Path to YAML/JSON config to merge.")
    known, remaining = base_parser.parse_known_args()
    file_cfg = _load_config_file(known.config)

    parser = argparse.ArgumentParser(description="Train PPO agents for custom Pong.", parents=[base_parser])
    parser.add_argument("--train-timesteps", type=int, default=file_cfg.get("train_timesteps", TrainConfig.train_timesteps))
    parser.add_argument("--n-steps", type=int, default=file_cfg.get("n_steps", TrainConfig.n_steps))
    parser.add_argument("--batch-size", type=int, default=file_cfg.get("batch_size", TrainConfig.batch_size))
    parser.add_argument("--n-epochs", type=int, default=file_cfg.get("n_epochs", TrainConfig.n_epochs))
    parser.add_argument("--gamma", type=float, default=file_cfg.get("gamma", TrainConfig.gamma))
    parser.add_argument("--learning-rate", type=float, default=file_cfg.get("learning_rate", TrainConfig.learning_rate))
    parser.add_argument("--device", type=str, default=file_cfg.get("device", TrainConfig.device))
    parser.add_argument("--target-fps", type=int, default=file_cfg.get("target_fps", TrainConfig.target_fps))
    parser.add_argument("--max-video-seconds", type=int, default=file_cfg.get("max_video_seconds", TrainConfig.max_video_seconds))
    parser.add_argument("--max-cycles", type=int, default=file_cfg.get("max_cycles", TrainConfig.max_cycles))
    parser.add_argument("--checkpoint-interval", type=int, default=file_cfg.get("checkpoint_interval", TrainConfig.checkpoint_interval))
    parser.add_argument("--iterations-per-set", type=int, default=file_cfg.get("iterations_per_set", TrainConfig.iterations_per_set))
    parser.add_argument("--n-envs", type=int, default=file_cfg.get("n_envs", TrainConfig.n_envs))
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
        config_path=known.config,
    )
    return cfg


def evaluate_model(model: PPO, episodes: int) -> Dict[str, float]:
    eval_env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode=None)
    rewards: List[float] = []
    returns: List[int] = []
    rally_lengths: List[int] = []
    wins = 0
    for _ in range(episodes):
        obs, _ = eval_env.reset()
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
            if info["left_score"] != last_left or info["right_score"] != last_right:
                rally_lengths.append(rally_steps)
                rally_steps = 0
                last_left, last_right = info["left_score"], info["right_score"]
            if rew > 0:
                returns.append(1)
            ep_rew += rew
            steps += 1
            done = terminated or truncated
        rewards.append(ep_rew)
        if eval_env.left_score > eval_env.right_score:
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
                return make_vec_env(
                    lambda: SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode=None),
                    n_envs=cfg.n_envs,
                    seed=seed,
                )
            except Exception as exc:  # pragma: no cover - only on flaky init
                last_err = exc
                time.sleep(0.5)
        raise last_err

    env = _make_env_with_retry()
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    latest_path = os.path.join(cfg.model_dir, f"{model_id}_latest.zip")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if os.path.exists(latest_path):
        print(f"[{model_id}] Loading existing model from {latest_path} to continue training...")
        model = PPO.load(latest_path, env=env, device=cfg.device)
    else:
        print(f"[{model_id}] No existing model found; starting a fresh PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=cfg.log_dir,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            learning_rate=cfg.learning_rate,
            device=cfg.device,
        )

    model.learn(total_timesteps=cfg.train_timesteps, reset_num_timesteps=False, progress_bar=True)

    stamped_model_path: Optional[str] = None
    if cfg.checkpoint_interval > 0 and not cfg.no_checkpoint:
        stamped_model_path = os.path.join(cfg.model_dir, f"{model_id}_{timestamp}.zip")
        model.save(stamped_model_path)
        print(f"[{model_id}] Saved timestamped model: {stamped_model_path}")

    model.save(latest_path)
    print(f"[{model_id}] Updated latest model: {latest_path}")

    metrics = evaluate_model(model, cfg.eval_episodes)
    print(f"[{model_id}] Avg reward over {cfg.eval_episodes} eval episodes: {metrics['avg_reward']:.3f}")

    segment, ponged = record_video_segment(
        model,
        ball_color=color,
        steps=400,
        overlay_text=f"{model_id} | r {metrics['avg_reward']:.2f} | win {metrics['win_rate']:.2f}",
    )
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
        pong_flags: List[bool] = []
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
                pong_flags.append(ponged)
                timestamp = stamp  # use last reported for video naming
                print(f"[{model_id}] Added {len(segment)} frames; Ponged: {ponged}; seed={derived_seed}")

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

        if pong_flags and not any(pong_flags):
            print("Failure detected: no model returned the ball this cycle.")
            failure_detected = True
        if no_improve_cycles >= cfg.early_stop_patience:
            print(f"No improvement for {no_improve_cycles} cycles; stopping early.")
            break

        if all_grid_frames:
            combined_video_path = os.path.join(cfg.video_dir, f"ppo_pong_combined_{timestamp}.mp4")
            try:
                with tempfile.NamedTemporaryFile(delete=False, dir=Path(combined_video_path).parent) as tmp:
                    imageio.mimsave(tmp.name, all_grid_frames, fps=cfg.target_fps)
                    tmp.flush()
                os.replace(tmp.name, combined_video_path)
                print(f"Saved combined video with all models in a grid: {combined_video_path} (frames: {len(all_grid_frames)})")
            except ValueError:
                print("Video frames were empty or invalid; skipping video write.")
            except Exception as exc:  # pragma: no cover - defensive
                print(f"Video write failed: {exc}")
        else:
            print("No frames captured; combined video not written.")

        if cfg.individual_videos and segments_by_model:
            for model_id, frames in segments_by_model.items():
                if not frames:
                    continue
                indiv_path = Path(cfg.video_dir) / f"{model_id}_{timestamp}.mp4"
                try:
                    with tempfile.NamedTemporaryFile(delete=False, dir=indiv_path.parent) as tmp:
                        imageio.mimsave(tmp.name, frames, fps=cfg.target_fps)
                        tmp.flush()
                    os.replace(tmp.name, indiv_path)
                    print(f"[{model_id}] Saved individual video: {indiv_path}")
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"[{model_id}] Failed to write individual video: {exc}")

if __name__ == "__main__":
    main()
