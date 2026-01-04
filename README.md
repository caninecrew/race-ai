# Pong AI Trainer

Lightweight reinforcement learning setup for training PPO agents on a custom Pong environment built with pygame and Stable Baselines3.

## Requirements
- Python 3.9+
- `pip install -r requirements` equivalent packages: `pygame`, `gymnasium`, `stable-baselines3`, `imageio`, `Pillow`, `numpy`, `torch`, `PyYAML` (for YAML configs)
- For Mario Kart Retro training: `pip install stable-retro gymnasium stable-baselines3 imageio` and import the SNES ROM once via `python -m retro.import /path/to/roms`

## Usage
- Demo the environment: `python pong.py` (controls: W/S left, Up/Down right; auto-tracks when headless).
- Train agents: `python train_pong_ppo.py --env-kind pong --config configs/example.yaml` or pass flags such as `--train-timesteps 200000 --checkpoint-interval 1 --seed 0 --n-envs 4 --iterations-per-set 2`.
  - Switch to Mario Kart Retro with `--env-kind kart` and kart-specific knobs like `--kart-action-set drift-lite|steer-basic`, `--kart-frame-size 84`, `--kart-frame-stack 4`, `--kart-no-grayscale`, or `--kart-use-ram` for compact feature observations.
  - Artifacts: models in `models/`, logs in `logs/`, combined videos in `videos/`. Resolved configs are emitted to `logs/train_run_<timestamp>.jsonl`, and per-cycle metrics are appended to `logs/metrics.csv`.
  - Config files: YAML/JSON supported with `--config`. CLI flags always override file values.
  - Checkpoints: `_latest` is always written; timestamped checkpoints can be disabled via `--no-checkpoint`. Top-K pruning keeps the best checkpoints by average reward.
  - Parallelism: control model parallelism via `--iterations-per-set` and vectorized envs with `--n-envs`. Use `--device cpu`/`cuda` and optional `--cpu-affinity` for pinning.
  - Determinism: set `--seed` and `--deterministic` to force deterministic torch ops; seeds per worker are derived from the base seed and recorded in logs.
  - Videos: enable per-model renders with `--individual-videos`. Video writes are atomic to avoid corruption.
- Evaluate a checkpoint: `python eval_pong.py --model-path models/ppo_pong_custom_latest.zip --episodes 5 [--render --output-csv logs/eval.csv]`.
  - Evaluation reports average reward (with CI), win rate, return rate, and rally length; missing models are handled gracefully.

## Mario Kart (Gym Retro)
- Import the SNES ROM once: `python -m retro.import /path/to/roms` (expects `SuperMarioKart-Snes`).
- Train via the unified entrypoint: `python train_pong_ppo.py --env-kind kart --kart-state MarioCircuit1 --kart-action-set drift-lite --kart-frame-size 84 --train-timesteps 200000 --iterations-per-set 2 --n-envs 2 --checkpoint-interval 1`.
  - Reward shaping blends track progress, speed bonuses, lap rewards, and penalties for going off-track or crashing. Observations can be 84x84 grayscale stacks or normalized RAM/info features with `--kart-use-ram`.
  - Video rollouts resize frames and keep overlay text; checkpoints/logs land in `models/`, `logs/`, and `videos/` as with Pong.
  - Hardware: Retro envs are heavier; prefer GPU (`--device cuda`) and fewer workers (`--iterations-per-set 1-2`, `--n-envs 2-4`) for a smoke run before scaling up.

## Common pitfalls
- Headless pygame: ensure `SDL_VIDEODRIVER=dummy` is respected (default when not rendering). On Linux servers install `libsdl2-dev` packages; on macOS use `brew install sdl2 sdl2_image`.
- Gymnasium/pygame versions: stick to recent gymnasium (>=0.29) and pygame (>=2.5) to avoid shape or surface issues.
- Stable Baselines3 and torch: CPU-only installs work; for GPU, set `--device cuda` and ensure CUDA-enabled torch is installed.
- Resume training: keep `_latest` checkpoints; rerun training with the same `model_dir` to continue. Passing `--no-checkpoint` skips timestamped saves but still updates `_latest`.

## Smoke Tests
- Quick checks for shapes, deterministic seeds, and paddle bounds: `python -m pytest tests/test_pong_env.py`.
- Extended smoke / config tests: `python -m pytest tests/test_train_pipeline.py -m \"not slow\"` (or include `-m slow` to run the minimal training smoke test).
