#!/usr/bin/env python3
"""Verify that every SHA256 in the .tex sources appears IN FULL in the compiled PDF.

Long hex strings are unbreakable, so a 64-character hash placed inside a
paragraph or a list item can overflow the line and be clipped in the PDF text
layer — a reviewer checking a hash then finds it does not match. Run this after
compiling:

    python3 check_hashes_in_pdf.py

Exit code 0 = every hash renders in full; 1 = at least one is missing/truncated.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HASH = re.compile(r"\b[0-9a-f]{64}\b")


def pdf_text(pdf: Path) -> str:
    out = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True)
    # drop line breaks, spaces and hyphenation so wrapped hashes can still match
    return re.sub(r"[\s\u00ad-]", "", out.stdout)


def main() -> int:
    failures = []
    checked = 0
    for tex in sorted(ROOT.glob("*.tex")):
        hashes = sorted(set(HASH.findall(tex.read_text(errors="ignore"))))
        if not hashes:
            continue
        pdf = tex.with_suffix(".pdf")
        if not pdf.exists():
            print(f"{tex.name}: hashes found but {pdf.name} missing — compile first")
            failures.append(f"{tex.name}: no PDF")
            continue
        text = pdf_text(pdf)
        for h in hashes:
            checked += 1
            status = "FULL" if h in text else "MISSING/TRUNCATED"
            print(f"{tex.name}  {h[:16]}…  {status}")
            if status != "FULL":
                failures.append(f"{tex.name}: {h}")
    print(f"\n{checked - len(failures)}/{checked} hashes render in full")
    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        print("\nFix: give each long hash its own line (e.g. break before it, or put it in its own quote block).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
