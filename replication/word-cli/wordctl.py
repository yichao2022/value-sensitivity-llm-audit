#!/usr/bin/env python3
"""Small macOS Microsoft Word CLI backed by AppleScript."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_applescript(script: str) -> str:
    try:
        result = subprocess.run(
            ["osascript", "-"], input=script, text=True,
            capture_output=True, check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("osascript is not available; this CLI requires macOS") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError((exc.stderr or exc.stdout).strip() or "Word automation failed") from exc
    return result.stdout.strip()


def apple_text(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def existing_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"document does not exist: {path}")
    return path


def execute(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "status":
        version = run_applescript('tell application "Microsoft Word" to return version')
        return {"ok": True, "word_version": version}

    document = existing_path(args.document)
    path = apple_text(str(document))

    if args.command == "read":
        script = f'''tell application "Microsoft Word"
    set doc to open POSIX file {path}
    set resultText to content of text object of doc
    close doc saving no
    return resultText
end tell'''
        return {"ok": True, "document": str(document), "text": run_applescript(script)}

    if args.command == "append":
        script = f'''tell application "Microsoft Word"
    set doc to open POSIX file {path}
    set bodyRange to text object of doc
    set content of bodyRange to (content of bodyRange & return & {apple_text(args.text)})
    save doc
    close doc saving no
end tell'''
        run_applescript(script)
        return {"ok": True, "document": str(document), "action": "append"}

    if args.command == "replace":
        script = f'''tell application "Microsoft Word"
    set doc to open POSIX file {path}
    set finder to find object of text object of doc
    set find text of finder to {apple_text(args.find)}
    set replace with of finder to {apple_text(args.replace)}
    set replace action of finder to replace all
    execute finder
    save doc
    close doc saving no
end tell'''
        run_applescript(script)
        return {"ok": True, "document": str(document), "action": "replace"}

    if args.command == "export-pdf":
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        script = f'''tell application "Microsoft Word"
    set doc to open POSIX file {path}
    save as doc file name {apple_text(str(output))} file format format PDF
    close doc saving no
end tell'''
        run_applescript(script)
        return {"ok": True, "document": str(document), "pdf": str(output)}

    raise ValueError(f"unknown command: {args.command}")


def make_parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(prog="wordctl", description="Control Microsoft Word on macOS")
    sub = cli.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="check Word and return its version")
    read = sub.add_parser("read", help="read document body text")
    read.add_argument("document")
    append = sub.add_parser("append", help="append text and save")
    append.add_argument("document")
    append.add_argument("text")
    replace = sub.add_parser("replace", help="find and replace all")
    replace.add_argument("document")
    replace.add_argument("find")
    replace.add_argument("replace")
    pdf = sub.add_parser("export-pdf", help="export a document as PDF")
    pdf.add_argument("document")
    pdf.add_argument("output")
    return cli


def main() -> int:
    try:
        result = execute(make_parser().parse_args())
    except (RuntimeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
