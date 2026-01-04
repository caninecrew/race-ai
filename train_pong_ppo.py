import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Optional, List, Tuple

import imageio
import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

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
    drawer.rectangle([(0, 0), (img.width, 22)], fill=(0, 0, 0))
    drawer.text((5, 4), text, fill=(255, 255, 255))
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
    train_timesteps: int = 200_000
    n_steps: int = 256
    batch_size: int = 512
    n_epochs: int = 4
    gamma: float = 0.99
    learning_rate: float = 2.5e-4
    target_fps: int = 30
    max_video_seconds: int = 120
    max_cycles: int = 1
    checkpoint_interval: int = 1  # cycles between timestamped checkpoints
    seed: int = 0
    early_stop_patience: int = 3
    improvement_threshold: float = 0.05
    eval_episodes: int = 3

    @property
    def max_video_frames(self) -> int:
        return self.target_fps * self.max_video_seconds


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train PPO agents for custom Pong.")
    parser.add_argument("--train-timesteps", type=int, default=TrainConfig.train_timesteps)
    parser.add_argument("--n-steps", type=int, default=TrainConfig.n_steps)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--n-epochs", type=int, default=TrainConfig.n_epochs)
    parser.add_argument("--gamma", type=float, default=TrainConfig.gamma)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--target-fps", type=int, default=TrainConfig.target_fps)
    parser.add_argument("--max-video-seconds", type=int, default=TrainConfig.max_video_seconds)
    parser.add_argument("--max-cycles", type=int, default=TrainConfig.max_cycles)
    parser.add_argument("--checkpoint-interval", type=int, default=TrainConfig.checkpoint_interval)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--early-stop-patience", type=int, default=TrainConfig.early_stop_patience)
    parser.add_argument("--improvement-threshold", type=float, default=TrainConfig.improvement_threshold)
    parser.add_argument("--eval-episodes", type=int, default=TrainConfig.eval_episodes)
    args = parser.parse_args()
    return TrainConfig(
        train_timesteps=args.train_timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        target_fps=args.target_fps,
        max_video_seconds=args.max_video_seconds,
        max_cycles=args.max_cycles,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        early_stop_patience=args.early_stop_patience,
        improvement_threshold=args.improvement_threshold,
        eval_episodes=args.eval_episodes,
    )


def evaluate_model(model: PPO, episodes: int) -> float:
    eval_env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode=None)
    total = 0.0
    for _ in range(episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_rew = 0.0
        steps = 0
        while not done and steps < eval_env.env.cfg.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, _ = eval_env.step(action)
            ep_rew += rew
            steps += 1
            done = terminated or truncated
        total += ep_rew
    eval_env.close()
    return total / episodes if episodes else 0.0


def main():
    cfg = parse_args()
    set_random_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    log_file = os.path.join("logs", f"train_run_{run_timestamp}.jsonl")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "config", **asdict(cfg)}) + "\n")

    env = make_vec_env(
        lambda: SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode=None),
        n_envs=8,
        seed=cfg.seed,
    )

    model_ids = [
        "ppo_pong_custom",
        "ppo_pong_custom_b",
    ]

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

    cycle = 0
    while not failure_detected and cycle < cfg.max_cycles:
        combined_frames_per_model: List[List[np.ndarray]] = []
        all_grid_frames: List[np.ndarray] = []
        scores: List[Tuple[str, float]] = []
        pong_flags: List[bool] = []
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        cycle += 1

        for idx, model_id in enumerate(model_ids):
            latest_path = f"models/{model_id}_latest.zip"
            color = ball_colors[idx % len(ball_colors)]

            if os.path.exists(latest_path):
                print(f"[{model_id}] Loading existing model from {latest_path} to continue training...")
                model = PPO.load(latest_path, env=env)
            else:
                print(f"[{model_id}] No existing model found; starting a fresh PPO model...")
                model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    tensorboard_log="logs",
                    n_steps=cfg.n_steps,
                    batch_size=cfg.batch_size,
                    n_epochs=cfg.n_epochs,
                    gamma=cfg.gamma,
                    learning_rate=cfg.learning_rate,
                )

            model.learn(total_timesteps=cfg.train_timesteps, reset_num_timesteps=False, progress_bar=False)

            if cycle % cfg.checkpoint_interval == 0:
                stamped_model_path = f"models/{model_id}_{timestamp}.zip"
                model.save(stamped_model_path)
                print(f"[{model_id}] Saved timestamped model: {stamped_model_path}")

            model.save(latest_path)
            print(f"[{model_id}] Updated latest model: {latest_path}")

            avg_rew = evaluate_model(model, cfg.eval_episodes)
            scores.append((model_id, avg_rew))
            print(f"[{model_id}] Avg reward over {cfg.eval_episodes} eval episodes: {avg_rew:.3f}")

            segment, ponged = record_video_segment(
                model,
                ball_color=color,
                steps=400,
                overlay_text=f"{model_id} | avg {avg_rew:.2f}",
            )
            if segment:
                combined_frames_per_model.append(segment)
            pong_flags.append(ponged)
            print(f"[{model_id}] Added {len(segment)} frames with ball color {color} to combined video. Ponged: {ponged}")

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
            best_latest = f"models/{best_id}_latest.zip"
            for model_id in model_ids:
                target_latest = f"models/{model_id}_latest.zip"
                if os.path.exists(best_latest) and best_latest != target_latest:
                    shutil.copy2(best_latest, target_latest)
            print(f"Best model this cycle: {best_id} (avg reward {best_score_cycle:.3f}); propagated to all _latest checkpoints.")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "event": "cycle",
                            "cycle": cycle,
                            "scores": scores,
                            "frames": len(all_grid_frames),
                            "timestamp": timestamp,
                        }
                    )
                    + "\n"
                )
            if best_score_cycle > best_score + cfg.improvement_threshold:
                best_score = best_score_cycle
                no_improve_cycles = 0
            else:
                no_improve_cycles += 1
        else:
            print("No scores recorded; cannot propagate best model.")

        if pong_flags and not any(pong_flags):
            print("Failure detected: no model returned the ball this cycle.")
            failure_detected = True
        if no_improve_cycles >= cfg.early_stop_patience:
            print(f"No improvement for {no_improve_cycles} cycles; stopping early.")
            break

        if all_grid_frames:
            combined_video_path = f"videos/ppo_pong_combined_{timestamp}.mp4"
            try:
                imageio.mimsave(combined_video_path, all_grid_frames, fps=cfg.target_fps)
                print(f"Saved combined video with all models in a grid: {combined_video_path} (frames: {len(all_grid_frames)})")
            except ValueError:
                print("Video frames were empty or invalid; skipping video write.")
        else:
            print("No frames captured; combined video not written.")

    env.close()


if __name__ == "__main__":
    main()
