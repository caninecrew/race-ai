import argparse
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import imageio.v2 as iio
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecMonitor

try:
    import retro  # type: ignore
except Exception:  # pragma: no cover - optional dep
    retro = None


DEFAULT_BUTTONS: Sequence[str] = (
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

ACTION_SETS: Dict[str, List[List[str]]] = {
    "small": [
        [],
        ["B"],  # accelerate
        ["B", "Right"],
        ["B", "Left"],
        ["Right"],
        ["Left"],
        ["Y"],  # brake
        ["Y", "Right"],
        ["Y", "Left"],
    ],
    "medium": [
        [],
        ["B"],
        ["B", "Right"],
        ["B", "Left"],
        ["B", "A"],  # hop/use item while accelerating
        ["Right"],
        ["Left"],
        ["Y"],
        ["Y", "Right"],
        ["Y", "Left"],
    ],
}


def _require_retro() -> None:
    if retro is None:
        raise RuntimeError(
            "stable-retro is required for Mario Kart training. Install with `pip install stable-retro` "
            "and import the SNES ROM via `python -m retro.import <path-to-roms>`."
        )


def _validate_game_and_state(env_id: str, state: Optional[str]) -> None:
    _require_retro()
    try:
        integrations = getattr(retro.data, "Integrations", None)
        inttype = getattr(integrations, "ALL", None) if integrations else None
        games = set(retro.data.list_games(inttype=inttype) if inttype is not None else retro.data.list_games())
    except Exception:
        # If stable-retro listing fails, fall back to make() which will raise a clearer error.
        return

    if env_id not in games:
        raise RuntimeError(
            f"Game '{env_id}' not found in stable-retro data. Import the ROM via "
            "`python -m retro.import <path-to-roms>` and confirm the game id."
        )

    if state:
        try:
            states = set(retro.data.list_states(env_id))
        except Exception:
            states = set()
        if states and state not in states:
            raise RuntimeError(
                f"State '{state}' not found for '{env_id}'. Available: {sorted(states)}. "
                "Use `retro.data.list_states('<game>')` after importing the ROM."
            )


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


def _preprocess_frame(frame: np.ndarray, size: int, grayscale: bool = True) -> np.ndarray:
    img = Image.fromarray(frame)
    if grayscale:
        img = img.convert("L")
    img = img.resize((size, size))
    arr = np.array(img, dtype=np.uint8)
    if grayscale:
        return arr  # H x W
    return arr.transpose(2, 0, 1)  # C x H x W if keeping color


class RetroMarioKartEnv(gym.Env):
    """
    Gymnasium wrapper for stable-retro Mario Kart (SNES) with discrete action mapping,
    frame skipping, stacking, and lightweight reward shaping.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_id: str = "SuperMarioKart-Snes",
        state: Optional[str] = None,
        action_set: str = "small",
        frame_size: int = 96,
        frame_stack: int = 4,
        frame_skip: int = 4,
        reward_mode: str = "progress",
        progress_scale: float = 0.1,
        score_scale: float = 0.01,
        living_penalty: float = -0.001,
        grayscale: bool = True,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        _validate_game_and_state(env_id, state)
        make_kwargs = {"game": env_id, "state": state, "render_mode": render_mode}
        self.env = retro.make(**make_kwargs)
        self.button_names: Sequence[str] = getattr(self.env, "buttons", DEFAULT_BUTTONS)
        if action_set not in ACTION_SETS:
            raise ValueError(f"Unknown action set '{action_set}'. Available: {list(ACTION_SETS)}")
        self.action_map = _build_action_map(self.button_names, ACTION_SETS[action_set])
        self.frame_size = frame_size
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.reward_mode = reward_mode
        self.progress_scale = progress_scale
        self.score_scale = score_scale
        self.living_penalty = living_penalty
        self.grayscale = grayscale
        self.frames: Deque[np.ndarray] = deque(maxlen=frame_stack)
        self.last_progress = 0.0
        self.last_score = 0.0
        self.last_obs_raw: Optional[np.ndarray] = None

        channels = 1 if grayscale else 3
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(channels * frame_stack, frame_size, frame_size), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(self.action_map))

    def _get_progress(self, info: Dict[str, float]) -> float:
        return float(
            info.get("x", info.get("pos", info.get("position", info.get("trackPos", 0.0))))
        )

    def _get_score(self, info: Dict[str, float]) -> float:
        return float(info.get("score", info.get("points", 0.0)))

    def _stack_obs(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(frame)
        while len(self.frames) < self.frame_stack:
            self.frames.append(frame)
        stacked = np.stack(list(self.frames), axis=0).astype(np.float32) / 255.0
        if self.grayscale:
            return stacked
        # color stacking: frames already channel-first; flatten stack dimension into channel
        return stacked.reshape(-1, self.frame_size, self.frame_size)

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
        frame = _preprocess_frame(raw_obs, self.frame_size, grayscale=self.grayscale)
        self.frames.clear()
        self.last_progress = 0.0
        self.last_score = 0.0
        self.last_obs_raw = raw_obs
        return self._stack_obs(frame), info

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
        score = self._get_score(info)
        shaped_reward = total_reward
        if self.reward_mode == "progress":
            shaped_reward += (progress - self.last_progress) * self.progress_scale
            shaped_reward += self.living_penalty
        elif self.reward_mode == "mixed":
            shaped_reward += (progress - self.last_progress) * self.progress_scale
            shaped_reward += (score - self.last_score) * self.score_scale
            shaped_reward += self.living_penalty
        elif self.reward_mode == "score":
            shaped_reward += (score - self.last_score) * self.score_scale
            shaped_reward += self.living_penalty

        self.last_progress = progress
        self.last_score = score
        self.last_obs_raw = raw_obs

        frame = _preprocess_frame(raw_obs, self.frame_size, grayscale=self.grayscale)
        obs = self._stack_obs(frame)
        terminated = bool(done)
        truncated = False
        return obs, float(shaped_reward), terminated, truncated, info

    def render(self):
        if self.last_obs_raw is not None:
            return self.last_obs_raw
        try:
            obs = self.env.render()
        except Exception:
            obs = None
        return obs

    def close(self):
        self.env.close()


@dataclass
class TrainConfig:
    env_id: str = "SuperMarioKart-Snes"
    state: Optional[str] = None
    action_set: str = "small"
    frame_size: int = 96
    frame_stack: int = 4
    frame_skip: int = 4
    reward_mode: str = "progress"
    progress_scale: float = 0.1
    score_scale: float = 0.01
    living_penalty: float = -0.001
    grayscale: bool = True
    train_timesteps: int = 500_000
    n_steps: int = 128
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.99
    learning_rate: float = 2.5e-4
    device: str = "auto"
    n_envs: int = 8
    seed: int = 0
    eval_episodes: int = 5
    video_episodes: int = 1
    target_fps: int = 20
    video_dir: str = "videos"
    model_dir: str = "models"
    log_dir: str = "logs"
    model_prefix: str = "ppo_mariokart"
    checkpoint_interval: int = 1
    no_checkpoint: bool = False
    config_path: Optional[str] = None


def _load_config_file(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping of keys to values.")
    return data


def parse_args() -> TrainConfig:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, help="Path to JSON config to merge.")
    known, remaining = base_parser.parse_known_args()
    file_cfg = _load_config_file(known.config)

    parser = argparse.ArgumentParser(
        description="Train PPO on Super Mario Kart (stable-retro).", parents=[base_parser]
    )
    parser.add_argument("--env-id", type=str, default=file_cfg.get("env_id", TrainConfig.env_id))
    parser.add_argument("--state", type=str, default=file_cfg.get("state", TrainConfig.state))
    parser.add_argument("--action-set", type=str, choices=list(ACTION_SETS), default=file_cfg.get("action_set", TrainConfig.action_set))
    parser.add_argument("--frame-size", type=int, default=file_cfg.get("frame_size", TrainConfig.frame_size))
    parser.add_argument("--frame-stack", type=int, default=file_cfg.get("frame_stack", TrainConfig.frame_stack))
    parser.add_argument("--frame-skip", type=int, default=file_cfg.get("frame_skip", TrainConfig.frame_skip))
    parser.add_argument(
        "--reward-mode",
        type=str,
        choices=["progress", "score", "mixed"],
        default=file_cfg.get("reward_mode", TrainConfig.reward_mode),
    )
    parser.add_argument("--progress-scale", type=float, default=file_cfg.get("progress_scale", TrainConfig.progress_scale))
    parser.add_argument("--score-scale", type=float, default=file_cfg.get("score_scale", TrainConfig.score_scale))
    parser.add_argument("--living-penalty", type=float, default=file_cfg.get("living_penalty", TrainConfig.living_penalty))
    parser.add_argument("--no-grayscale", action="store_true", help="Keep color frames instead of grayscale.")
    parser.add_argument("--train-timesteps", type=int, default=file_cfg.get("train_timesteps", TrainConfig.train_timesteps))
    parser.add_argument("--n-steps", type=int, default=file_cfg.get("n_steps", TrainConfig.n_steps))
    parser.add_argument("--batch-size", type=int, default=file_cfg.get("batch_size", TrainConfig.batch_size))
    parser.add_argument("--n-epochs", type=int, default=file_cfg.get("n_epochs", TrainConfig.n_epochs))
    parser.add_argument("--gamma", type=float, default=file_cfg.get("gamma", TrainConfig.gamma))
    parser.add_argument("--learning-rate", type=float, default=file_cfg.get("learning_rate", TrainConfig.learning_rate))
    parser.add_argument("--device", type=str, default=file_cfg.get("device", TrainConfig.device))
    parser.add_argument("--n-envs", type=int, default=file_cfg.get("n_envs", TrainConfig.n_envs))
    parser.add_argument("--seed", type=int, default=file_cfg.get("seed", TrainConfig.seed))
    parser.add_argument("--eval-episodes", type=int, default=file_cfg.get("eval_episodes", TrainConfig.eval_episodes))
    parser.add_argument("--video-episodes", type=int, default=file_cfg.get("video_episodes", TrainConfig.video_episodes))
    parser.add_argument("--target-fps", type=int, default=file_cfg.get("target_fps", TrainConfig.target_fps))
    parser.add_argument("--video-dir", type=str, default=file_cfg.get("video_dir", TrainConfig.video_dir))
    parser.add_argument("--model-dir", type=str, default=file_cfg.get("model_dir", TrainConfig.model_dir))
    parser.add_argument("--log-dir", type=str, default=file_cfg.get("log_dir", TrainConfig.log_dir))
    parser.add_argument("--model-prefix", type=str, default=file_cfg.get("model_prefix", TrainConfig.model_prefix))
    parser.add_argument("--checkpoint-interval", type=int, default=file_cfg.get("checkpoint_interval", TrainConfig.checkpoint_interval))
    parser.add_argument("--no-checkpoint", action="store_true", default=file_cfg.get("no_checkpoint", TrainConfig.no_checkpoint))
    args = parser.parse_args(remaining)
    cfg = TrainConfig(
        env_id=args.env_id,
        state=args.state,
        action_set=args.action_set,
        frame_size=args.frame_size,
        frame_stack=args.frame_stack,
        frame_skip=args.frame_skip,
        reward_mode=args.reward_mode,
        progress_scale=args.progress_scale,
        score_scale=args.score_scale,
        living_penalty=args.living_penalty,
        grayscale=not args.no_grayscale,
        train_timesteps=args.train_timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        device=args.device,
        n_envs=args.n_envs,
        seed=args.seed,
        eval_episodes=args.eval_episodes,
        video_episodes=args.video_episodes,
        target_fps=args.target_fps,
        video_dir=args.video_dir,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        model_prefix=args.model_prefix,
        checkpoint_interval=args.checkpoint_interval,
        no_checkpoint=args.no_checkpoint,
        config_path=known.config,
    )
    return cfg


def evaluate_model(model: PPO, cfg: TrainConfig) -> Dict[str, float]:
    eval_env = RetroMarioKartEnv(
        env_id=cfg.env_id,
        state=cfg.state,
        action_set=cfg.action_set,
        frame_size=cfg.frame_size,
        frame_stack=cfg.frame_stack,
        frame_skip=cfg.frame_skip,
        reward_mode=cfg.reward_mode,
        progress_scale=cfg.progress_scale,
        score_scale=cfg.score_scale,
        living_penalty=cfg.living_penalty,
        grayscale=cfg.grayscale,
    )
    rewards: List[float] = []
    for _ in range(cfg.eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_rew = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, _ = eval_env.step(action)
            ep_rew += rew
            done = terminated or truncated
        rewards.append(ep_rew)
    eval_env.close()
    return {
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
    }


def record_video(model: PPO, cfg: TrainConfig, path: Path, steps: int = 2000) -> bool:
    env = RetroMarioKartEnv(
        env_id=cfg.env_id,
        state=cfg.state,
        action_set=cfg.action_set,
        frame_size=cfg.frame_size,
        frame_stack=cfg.frame_stack,
        frame_skip=cfg.frame_skip,
        reward_mode=cfg.reward_mode,
        progress_scale=cfg.progress_scale,
        score_scale=cfg.score_scale,
        living_penalty=cfg.living_penalty,
        grayscale=cfg.grayscale,
        render_mode="rgb_array",
    )
    obs, _ = env.reset()
    frames: List[np.ndarray] = []
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            break
    env.close()
    if not frames:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with iio.get_writer(path, format="FFMPEG", fps=cfg.target_fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        return True
    except Exception:
        return False


def main():
    cfg = parse_args()
    set_random_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.video_dir).mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log = Path(cfg.log_dir) / f"train_mariokart_{run_ts}.jsonl"
    with run_log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "config", **asdict(cfg), "timestamp": run_ts}) + "\n")

    def _make_env():
        return RetroMarioKartEnv(
            env_id=cfg.env_id,
            state=cfg.state,
            action_set=cfg.action_set,
            frame_size=cfg.frame_size,
            frame_stack=cfg.frame_stack,
            frame_skip=cfg.frame_skip,
            reward_mode=cfg.reward_mode,
            progress_scale=cfg.progress_scale,
            score_scale=cfg.score_scale,
            living_penalty=cfg.living_penalty,
            grayscale=cfg.grayscale,
        )

    vec_env = make_vec_env(_make_env, n_envs=cfg.n_envs, seed=cfg.seed)
    vec_env = VecMonitor(vec_env)
    latest_path = Path(cfg.model_dir) / f"{cfg.model_prefix}_latest.zip"

    if latest_path.exists():
        print(f"Loading existing model from {latest_path}...")
        model = PPO.load(latest_path, env=vec_env, device=cfg.device)
    else:
        print("Starting fresh PPO model for Mario Kart...")
        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=cfg.log_dir,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            learning_rate=cfg.learning_rate,
            device=cfg.device,
        )

    start = time.time()
    model.learn(total_timesteps=cfg.train_timesteps, progress_bar=True)
    elapsed = time.time() - start
    stamped_path: Optional[Path] = None
    if cfg.checkpoint_interval > 0 and not cfg.no_checkpoint:
        stamped_path = Path(cfg.model_dir) / f"{cfg.model_prefix}_{run_ts}.zip"
        model.save(stamped_path)
        print(f"Saved checkpoint to {stamped_path}")
    model.save(latest_path)
    print(f"Updated latest model at {latest_path}")

    metrics = evaluate_model(model, cfg)
    with run_log.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "event": "summary",
                    "timestamp": run_ts,
                    "elapsed_seconds": elapsed,
                    "checkpoint": str(stamped_path) if stamped_path else None,
                    **metrics,
                }
            )
            + "\n"
        )
    print(f"Eval avg reward: {metrics['avg_reward']:.3f} (std {metrics['reward_std']:.3f})")

    if cfg.video_episodes > 0:
        video_path = Path(cfg.video_dir) / f"{cfg.model_prefix}_{run_ts}.mp4"
        if record_video(model, cfg, video_path):
            print(f"Saved evaluation video to {video_path}")
        else:
            print("Video capture skipped (no frames rendered).")


if __name__ == "__main__":
    main()
