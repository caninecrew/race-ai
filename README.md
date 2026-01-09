# Pong AI Trainer

Lightweight reinforcement learning setup for training PPO agents on a custom Pong environment built with pygame and Stable Baselines3.

## Requirements
- Python 3.9+
- Recommended: create a virtual environment and install dependencies with `pip install -r requirements.txt`.

## Usage
- Demo the environment: `python pong.py` (controls: W/S left, Up/Down right; auto-tracks when headless).
- Train agents: `python train_pong_ppo.py --config configs/example.yaml` or pass flags such as `--train-timesteps 200000 --checkpoint-interval 1 --seed 0 --n-envs 4 --iterations-per-set 2`.
  - Artifacts: models in `models/`, logs in `logs/`, combined videos in `videos/`. Resolved configs are emitted to `logs/train_run_<timestamp>.jsonl`, and per-cycle metrics are appended to `logs/metrics.csv`.
  - Config files: YAML/JSON supported with `--config`. CLI flags always override file values.
  - Checkpoints: `_latest` is always written; timestamped checkpoints can be disabled via `--no-checkpoint`. Top-K pruning keeps the best checkpoints by average reward.
  - Parallelism: control model parallelism via `--iterations-per-set` and vectorized envs with `--n-envs`. Use `--device cpu`/`cuda` and optional `--cpu-affinity` for pinning.
  - Determinism: set `--seed` and `--deterministic` to force deterministic torch ops; seeds per worker are derived from the base seed and recorded in logs.
  - Videos: enable per-model renders with `--individual-videos`. Video writes are atomic to avoid corruption.
- Evaluate a checkpoint: `python eval_pong.py --model-path models/ppo_pong_custom_latest.zip --episodes 5 [--render --output-csv logs/eval.csv]`.
  - Evaluation reports average reward (with CI), win rate, return rate, and rally length; missing models are handled gracefully.

## Common pitfalls
- Headless pygame: ensure `SDL_VIDEODRIVER=dummy` is respected (default when not rendering). On Linux servers install `libsdl2-dev` packages; on macOS use `brew install sdl2 sdl2_image`.
- Gymnasium/pygame versions: stick to recent gymnasium (>=0.29) and pygame (>=2.5) to avoid shape or surface issues.
- Stable Baselines3 and torch: CPU-only installs work; for GPU, set `--device cuda` and ensure CUDA-enabled torch is installed.
- Resume training: keep `_latest` checkpoints; rerun training with the same `model_dir` to continue. Passing `--no-checkpoint` skips timestamped saves but still updates `_latest`.

## Quickstart (venv)
```
python -m venv .venv
.\.venv\Scripts\activate   # or source .venv/bin/activate on *nix
pip install -r requirements.txt
```

## Smoke Tests
- Quick checks for shapes, deterministic seeds, and paddle bounds: `python -m pytest tests/test_pong_env.py`.
- Extended smoke / config tests: `python -m pytest tests/test_train_pipeline.py -m \"not slow\"` (or include `-m slow` to run the minimal training smoke test).
