#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, time
from pathlib import Path

PICTURES = Path.home() / "Pictures" / "CamPhoto"

def osa(script: str) -> str:
    try:
        p = subprocess.run(["osascript", "-"], input=script, text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError((e.stderr or e.stdout).strip() or "CamPhoto GUI automation failed")
    return p.stdout.strip()

def ensure_app() -> None:
    subprocess.run(["open", "-a", "CamPhoto"], check=True, stdout=subprocess.DEVNULL)
    time.sleep(1)

def click_button(index: int) -> None:
    osa(f'''tell application "System Events"
    tell process "CamPhoto"
        tell window "CamPhoto"
            tell group 1 to click button {index}
        end tell
    end tell
end tell''')

def recent_files(since: float) -> list[Path]:
    if not PICTURES.exists(): return []
    return sorted((p for p in PICTURES.iterdir() if p.is_file() and p.stat().st_mtime >= since), key=lambda p: p.stat().st_mtime)

def main() -> int:
    parser = argparse.ArgumentParser(prog="camphoto", description="Control CamPhoto on macOS")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("photo", help="capture a photo")
    rec = sub.add_parser("record", help="record a video for N seconds")
    rec.add_argument("--seconds", type=float, default=5)
    args = parser.parse_args()
    try:
        if args.command == "status":
            running = subprocess.run(["pgrep", "-x", "CamPhoto"], capture_output=True).returncode == 0
            print(json.dumps({"ok": True, "app": "CamPhoto", "running": running, "output_dir": str(PICTURES)}, indent=2))
            return 0
        ensure_app()
        started = time.time()
        if args.command == "photo":
            click_button(8); time.sleep(1)
            files = recent_files(started)
            print(json.dumps({"ok": True, "action": "photo", "files": [str(p) for p in files]}, indent=2))
            return 0
        if args.seconds <= 0 or args.seconds > 3600: raise ValueError("seconds must be between 0 and 3600")
        click_button(9); time.sleep(args.seconds); click_button(9); time.sleep(1)
        files = recent_files(started)
        print(json.dumps({"ok": True, "action": "record", "seconds": args.seconds, "files": [str(p) for p in files]}, indent=2))
        return 0
    except (RuntimeError, OSError, ValueError) as e:
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False), file=__import__('sys').stderr)
        return 1

if __name__ == "__main__": raise SystemExit(main())
