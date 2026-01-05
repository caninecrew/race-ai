from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class MK64Paths:
    """
    Centralizes default paths used by both manual play (Mupen64Plus) and training.

    Defaults mirror your current play script assumptions: ROM stored under a local venv-ish path
    and Linux plugin locations, but all are overrideable via environment variables.
    """
    repo_root: Path
    rom_path: Path
    gfx_plugin: Path
    input_driver: Path
    mupen_cmd: str

    @staticmethod
    def from_env(repo_root: Optional[Path] = None) -> "MK64Paths":
        root = repo_root or Path(__file__).resolve().parent

        # Default ROM location (matches your current play script) :contentReference[oaicite:2]{index=2}
        default_rom = root / "mk64-venv" / "lib" / "python3.8" / "site-packages" / "gym_mupen64plus" / "ROMs" / "marioKart.n64"

        # Default plugins (Linux paths; override as needed)
        default_gfx = Path("/usr/lib/x86_64-linux-gnu/mupen64plus/mupen64plus-video-rice.so")
        default_input = Path("/usr/lib/x86_64-linux-gnu/mupen64plus/mupen64plus-input-sdl.so")

        rom = Path(os.environ.get("MK64_ROM", str(default_rom)))
        gfx = Path(os.environ.get("MK64_GFX_PLUGIN", str(default_gfx)))
        inp = Path(os.environ.get("MK64_INPUT_DRIVER", str(default_input)))
        mupen = os.environ.get("MUPEN_CMD", "mupen64plus")

        return MK64Paths(repo_root=root, rom_path=rom, gfx_plugin=gfx, input_driver=inp, mupen_cmd=mupen)


def validate_manual_play_paths(p: MK64Paths) -> List[Path]:
    """
    Returns a list of missing files required for manual play.
    """
    missing: List[Path] = []
    if not p.rom_path.exists():
        missing.append(p.rom_path)
    if not p.gfx_plugin.exists():
        missing.append(p.gfx_plugin)
    if not p.input_driver.exists():
        missing.append(p.input_driver)
    return missing


def find_mupen_executable(mupen_cmd: str) -> Optional[str]:
    """
    Returns a runnable executable path if it can be found, otherwise None.
    """
    exe = shutil.which(mupen_cmd)
    if exe:
        return exe
    # Common Windows name
    if not mupen_cmd.lower().endswith(".exe"):
        exe2 = shutil.which(mupen_cmd + ".exe")
        if exe2:
            return exe2
    return None


def build_mupen_command(p: MK64Paths) -> List[str]:
    """
    Builds the Mupen64Plus CLI command used by File 1.

    Notes:
    - Uses dummy audio to avoid Linux/WSL audio issues (same as your current script). :contentReference[oaicite:3]{index=3}
    - SDL input plugin handles keyboard mappings.
    """
    return [
        p.mupen_cmd,
        "--nospeedlimit",
        "--nosaveoptions",
        "--resolution",
        "640x480",
        "--gfx",
        str(p.gfx_plugin),
        "--audio",
        "dummy",
        "--input",
        str(p.input_driver),
        str(p.rom_path),
    ]


def require_gym_mupen64plus() -> None:
    """
    Training requires gym_mupen64plus (MK64 via Mupen64Plus + Gym env).
    If import fails, raise a clear error.
    """
    try:
        import gym_mupen64plus  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Training requires gym_mupen64plus to be installed and importable.\n"
            "If you can launch MK64 manually but cannot train, it usually means the Gym environment "
            "package is missing or not in the same venv you are using for training."
        ) from exc


def list_registered_env_ids(keyword: str = "Mario") -> List[str]:
    """
    Lists Gym environment IDs that contain the given keyword.
    Useful for discovering track-specific MK64 env ids.
    """
    try:
        import gym  # type: ignore
    except Exception:
        return []

    ids: List[str] = []
    try:
        specs = gym.envs.registry.all()  # type: ignore[attr-defined]
    except Exception:
        try:
            specs = gym.envs.registry.values()  # type: ignore[attr-defined]
        except Exception:
            specs = []

    for spec in specs:
        env_id = getattr(spec, "id", "")
        if keyword.lower() in env_id.lower():
            ids.append(env_id)
    return sorted(ids)
