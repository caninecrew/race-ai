from __future__ import annotations

import subprocess
import sys

from mk64_common import MK64Paths, build_mupen_command, find_mupen_executable, validate_manual_play_paths

# Default keyboard mappings for the SDL input plugin (mupen64plus-input-sdl.so):
#   Stick:    Arrow keys (Up/Down/Left/Right)
#   A:        Z
#   B:        X
#   L:        A
#   R:        S
#   Start:    Enter/Return
#   C-Buttons: I (Up), K (Down), J (Left), L (Right)
#   D-Pad:    default disabled unless configured; most play is via Stick + A/B + L/R + Start.
# These are the defaults shipped with mupen64plus SDL input; adjust via ~/.config/mupen64plus/InputAutoCfg.ini if needed.


def main() -> int:
    paths = MK64Paths.from_env()

    missing = validate_manual_play_paths(paths)
    if missing:
        for p in missing:
            sys.stderr.write(f"Missing file: {p}\n")
        sys.stderr.write(
            "Fix the missing paths or override via environment variables:\n"
            "  MK64_ROM, MK64_GFX_PLUGIN, MK64_INPUT_DRIVER\n"
        )
        return 1

    exe = find_mupen_executable(paths.mupen_cmd)
    if exe is None:
        sys.stderr.write(
            f"Could not find mupen64plus executable '{paths.mupen_cmd}' on PATH.\n"
            "Install Mupen64Plus and ensure it is on PATH, or set MUPEN_CMD to the full executable path.\n"
        )
        return 1

    cmd = build_mupen_command(paths)
    print("Launching Mupen64Plus. Close the emulator window to exit.")
    print("Command:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"mupen64plus exited with error code {exc.returncode}\n")
        return exc.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
