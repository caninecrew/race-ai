# Race AI (Mario Kart 64)

Mario Kart 64–only stack built around `gym-mupen64plus`, focused on launching MK64 for manual play and training PPO agents against the emulator.

## What’s here
- `play_mario_kart.py`: launch Mupen64Plus with sensible defaults and keyboard bindings. Batch wrappers (`play_mk64.bat`, `train_mk64.bat`) call into it from Windows/WSL.
- `gym_mupen64plus.py`: light PPO harness using Stable Baselines3 on `gym_mupen64plus` environments. Registers the custom menu-restricted env from `mk64_flow_env.py` when available.
- `mk64_flow_env.py`, `mk64_common.py`, `key_bindings_mk64.py`: helpers for scripted menus, restricted action spaces, and default input mappings.

## Prerequisites (WSL/Ubuntu)
- Python 3.8 with venv modules.
- System packages for Mupen64Plus and build tooling:
  ```bash
  sudo apt-get update && sudo apt-get install -y \
    python3.8 python3.8-venv python3.8-dev build-essential cmake git \
    mupen64plus-ui-console mupen64plus-input-all mupen64plus-video-rice mupen64plus-audio-sdl xvfb
  ```

## Setup
1) Create and activate the dedicated venv (keeps the legacy `gym` pinned and separate from anything modern):
   ```bash
   python3.8 -m venv mk64-venv
   source mk64-venv/bin/activate
   ```
2) Install Python deps (legacy gym/tooling): `pip install -r requirements.txt`  
   Then install the MK64 wrapper: `pip install --no-deps git+https://github.com/bzier/gym-mupen64plus.git`
3) Place your ROM where the env expects it (or override with `MK64_ROM`):
   `mk64-venv/lib/python3.8/site-packages/gym_mupen64plus/ROMs/marioKart.n64`
4) (Optional) Set plugin overrides if your Mupen64Plus install lives elsewhere:
   - `MK64_GFX_PLUGIN` (default: `/usr/lib/x86_64-linux-gnu/mupen64plus/mupen64plus-video-rice.so`)
   - `MK64_INPUT_DRIVER` (default: `/usr/lib/x86_64-linux-gnu/mupen64plus/mupen64plus-input-sdl.so`)
   - `MUPEN_CMD` if the executable is not on PATH

## Smoke test (imports + env)
```bash
source mk64-venv/bin/activate
python - <<'PY'
import gym, gym_mupen64plus  # noqa: F401
env = gym.make("Mario-Kart-Discrete-Luigi-Raceway-v0")
obs = env.reset()
print(type(obs), getattr(obs, "shape", None))
env.close()
PY
```

## Playing manually
- Activate the venv, ensure your ROM/plugins are reachable, then run:
  ```bash
  python play_mario_kart.py
  ```
- Windows/WSL: `play_mk64.bat` wraps the above (reads the repo path automatically).
- Default SDL input bindings (mupen64plus): Stick=Arrows, A(accel)=Z, B(brake)=X, R(hop/slide)=S, L(map)=A, Start=Enter, C-Up/Down/Left/Right=I/K/J/L.

## Training
- PPO harness (Stable Baselines3) against gym-mupen64plus:
  ```bash
  python gym_mupen64plus.py --env-id Mario-Kart-Menu-Restricted-v0 --timesteps 2_000_000 --device cuda
  ```
  - Use `--list-envs` to discover available env ids (includes the custom menu-restricted env when registered).
  - Artifacts land in `models_mk64/` and `logs_mk64/`.
  - The custom flow env (`MenuScriptedMarioKartEnv`) scripts menus up to driver select, restricts driving actions, and lives in `mk64_flow_env.py`.
- If `stable-baselines3` / `torch` are not installed in your mk64 venv, install them manually (version choice depends on your CUDA/CPU setup).

## Environment overrides
- `MK64_ROM`, `MK64_GFX_PLUGIN`, `MK64_INPUT_DRIVER`, `MUPEN_CMD` are honored by `mk64_common.py` and the launch/training scripts.
- The batch files use the repo path automatically; override `MK64_ROM` inside them if your ROM location differs.

## Known caveats
- This stack intentionally uses the legacy `gym` (0.7.x) required by `gym-mupen64plus`; keep it isolated from modern gymnasium installs.
- Stable Baselines3 expects more recent gym APIs; if you hit compatibility errors, pin SB3/torch to versions that work for your environment or adapt the training harness accordingly.
