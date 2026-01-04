import json
from pathlib import Path

import numpy as np
import pytest

from train_pong_ppo import (
    SB3PongEnv,
    build_grid_frames,
    _add_overlay,
    parse_args,
    TrainConfig,
    _train_single,
)
from pong import simple_tracking_policy


def test_sb3_pong_env_rollout():
    env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode=None)
    obs, info = env.reset(seed=123)
    assert obs.shape == (6,)
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert next_obs.shape == (6,)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated in (True, False)
    env.close()


def test_build_grid_frames_holds_last_frame():
    a = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]
    b = [np.ones((2, 2, 3), dtype=np.uint8) for _ in range(3)]
    frames = build_grid_frames([a, b])
    assert len(frames) == 3
    assert np.array_equal(frames[-1][:2, :2, :], a[-1])


def test_add_overlay_marks_header():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    overlaid = _add_overlay(frame, "test")
    assert not np.array_equal(frame[0, 0], overlaid[0, 0])


def test_parse_args_reads_config(tmp_path, monkeypatch):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(json.dumps({"n_steps": 5, "iterations_per_set": 1}))
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    monkeypatch.setattr(
        "sys.argv",
        ["train_pong_ppo.py", "--config", str(cfg_file), "--n-steps", "7", "--max-cycles", "1"],
    )
    cfg = parse_args()
    assert cfg.n_steps == 7  # CLI override
    assert cfg.iterations_per_set == 1
    assert cfg.max_cycles == 1


@pytest.mark.slow
def test_train_single_smoke(tmp_path):
    cfg = TrainConfig(
        train_timesteps=8,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        n_envs=1,
        eval_episodes=1,
        iterations_per_set=1,
        max_cycles=1,
        checkpoint_interval=0,
        no_checkpoint=True,
        model_dir=str(tmp_path / "models"),
        log_dir=str(tmp_path / "logs"),
        video_dir=str(tmp_path / "videos"),
        metrics_csv=str(tmp_path / "logs" / "metrics.csv"),
    )
    model_id, metrics, segment, ponged, timestamp, latest_path, stamped = _train_single(
        "smoke_model", (200, 0, 0), cfg, seed=123
    )
    assert model_id == "smoke_model"
    assert "avg_reward" in metrics
    assert Path(latest_path).exists()
    assert stamped is None  # disabled checkpoints
    assert isinstance(ponged, bool)
    assert timestamp
    assert segment  # should have at least one frame
