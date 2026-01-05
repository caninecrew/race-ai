from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from mk64_common import MK64Paths, build_mupen_command, find_mupen_executable, validate_manual_play_paths

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play Mario Kart 64 via mupen64plus.")
    p.add_argument("--headless", action="store_true", help="Run under Xvfb (no visible window).")
    p.add_argument(
        "--record",
        type=Path,
        default=None,
        help="Path to save an MP4 capture. Implies --headless. Example: videos/mk64_run.mp4",
    )
    return p.parse_args()


def _require_tool(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise RuntimeError(f"Required tool '{name}' not found on PATH.")
    return exe


def _start_xvfb(display: str = ":99", size: str = "640x480x24") -> subprocess.Popen:
    xvfb = _require_tool("Xvfb")
    cmd = [xvfb, display, "-screen", "0", size]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def _start_ffmpeg(display: str, video_size: str, output_path: Path) -> subprocess.Popen:
    ffmpeg = _require_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-video_size",
        video_size,
        "-f",
        "x11grab",
        "-i",
        display,
        "-codec:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def main() -> int:
    args = _parse_args()

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

    # Headless options
    display = os.environ.get("DISPLAY", ":0")
    xvfb_proc: subprocess.Popen | None = None
    ffmpeg_proc: subprocess.Popen | None = None
    video_size = "640x480"

    if args.headless or args.record:
        # Use a private Xvfb display for headless play/recording.
        display = ":99"
        try:
            xvfb_proc = _start_xvfb(display=display, size=f"{video_size}x24")
        except Exception as exc:  # pragma: no cover - system dep
            sys.stderr.write(f"Failed to start Xvfb: {exc}\n")
            return 1

    if args.record:
        try:
            ffmpeg_proc = _start_ffmpeg(display=display, video_size=video_size, output_path=args.record)
        except Exception as exc:  # pragma: no cover - system dep
            sys.stderr.write(f"Failed to start ffmpeg: {exc}\n")
            if xvfb_proc:
                xvfb_proc.terminate()
            return 1

    print("Launching Mupen64Plus. Close the emulator window to exit.")
    print("Command:", " ".join(cmd))

    env = os.environ.copy()
    env["DISPLAY"] = display

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"mupen64plus exited with error code {exc.returncode}\n")
        return exc.returncode
    finally:
        if ffmpeg_proc:
            ffmpeg_proc.terminate()
            ffmpeg_proc.wait(timeout=5)
        if xvfb_proc:
            xvfb_proc.terminate()
            xvfb_proc.wait(timeout=5)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
