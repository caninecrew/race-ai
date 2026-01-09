# Pong AI Trainer

Lightweight reinforcement learning setup for training PPO agents on a custom Pong environment built with pygame and Stable Baselines3.

## Requirements
- Python 3.9+
- Recommended: create a virtual environment and install dependencies with `pip install -r requirements.txt`.

## Usage
- Demo the environment: `python pong.py` (controls: W/S left, Up/Down right; auto-tracks when headless).
- Train agents: `python train_pong_ppo.py --config configs/example.yaml` or pass flags such as `--train-timesteps 200000 --checkpoint-interval 1 --seed 0 --n-envs 4 --iterations-per-set 2`.
  - Artifacts: models in `models/`, logs in `logs/`, combined videos in `videos/`. Resolved configs are emitted to `logs/train_run_<timestamp>.jsonl`, per-cycle metrics go to `logs/metrics.csv` (with deltas), and a JSON/HTML report is written per run.
  - Config files: YAML/JSON supported with `--config`; YAML can contain `defaults` and `profiles` sections. CLI flags always override file values, and `--profile` merges matching profile overrides.
  - Checkpoints: `_latest` is always written; timestamped checkpoints can be disabled via `--no-checkpoint`. Top-K pruning keeps the best checkpoints by average reward. Use `--resume-from <path>` to continue from a specific checkpoint.
  - Parallelism: control model parallelism via `--iterations-per-set` and vectorized envs with `--n-envs`. Use `--device cpu`/`cuda`, optional `--cpu-affinity auto|0,1` for pinning, and `--num-threads` to cap torch threads. Watchdog mode (`--watchdog`) skips failing workers instead of aborting the run.
  - Determinism: set `--seed` and `--deterministic` to force deterministic torch ops; seeds per worker are derived from the base seed and recorded in logs. Eval determinism can be toggled via `--eval-deterministic`.
  - Videos: enable per-model renders with `--individual-videos`. Adjust resolution via `--video-resolution 320x192`, add longer eval clips with `--long-eval-video-steps`, and seed-tagged filenames are emitted. A “status” flag reports the current best metrics without training.
- Evaluate a checkpoint: `python eval_pong.py --model-path models/ppo_pong_custom_latest.zip --episodes 5 [--render --output-csv logs/eval.csv --compare models/a.zip models/b.zip --plot-path logs/compare.png]`.
  - Evaluation reports average reward (with CI), win rate, return rate, and rally length; missing models are handled gracefully. `--compare` performs head-to-head comparisons and can optionally plot summaries.

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

If you prefer a one-liner setup after cloning, run the bootstrap scripts:
- Windows (PowerShell): `powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1`
- macOS/Linux: `bash scripts/setup_env.sh`

## Common recipes
- Quick smoke (headless, short): `python train_pong_ppo.py --profile quick --max-cycles 1 --dry-run` to validate env setup; drop `--dry-run` to train briefly.
- 1–2 minute videos: `python train_pong_ppo.py --video-steps 3600 --max-video-seconds 120 --target-fps 30 --individual-videos`.
- GPU profile: `python train_pong_ppo.py --profile gpu --iterations-per-set 2 --n-envs 16 --stream-tensorboard`.
- Status check: `python train_pong_ppo.py --status` prints the current best metrics from `logs/metrics.csv`.
- Live dashboard: `python dashboard.py` then open `http://127.0.0.1:8000` for live metrics, heatmap, and comparison panels.
- Arcade launcher: run `scripts/launch_arcade.bat` to open the dashboard or play human/model matchups without CLI flags.

## Troubleshooting
- Progress bars: install `pip install rich tqdm` (or `pip install stable-baselines3[extra]`) to enable the progress bar output.
- Headless pygame: ensure `SDL_VIDEODRIVER=dummy` is respected (default when not rendering). On Linux servers install `libsdl2-dev` packages; on macOS use `brew install sdl2 sdl2_image`.
- Permissions: if video or model writes fail, ensure the `videos/`, `models/`, and `logs/` directories are writable. HTML/JSON reports are created under `logs/`.

## Smoke Tests
- Quick checks for shapes, deterministic seeds, and paddle bounds: `python -m pytest tests/test_pong_env.py`.
- Extended smoke / config tests: `python -m pytest tests/test_train_pipeline.py -m \"not slow\"` (or include `-m slow` to run the minimal training smoke test).
