#!/usr/bin/env python3
"""Count the main-text word budget of main.tex.

Body text = Introduction .. end of Conclusion, excluding the title block, abstract,
figure/table environments (and their captions), and the bibliography. Per-section
counts are printed so the cuts can be targeted.
"""
import re
import sys
from pathlib import Path

TEX = Path('/Users/cary/Documents/value-sensitivity-llm-audit/main.tex')


def strip_env(text, names):
    for n in names:
        text = re.sub(rf"\\begin\{{{n}\}}.*?\\end\{{{n}\}}", " ", text, flags=re.S)
    return text


def words(text):
    text = re.sub(r"%.*", " ", text)                       # comments
    text = re.sub(r"\\caption\{.*?\}", " ", text, flags=re.S)
    text = re.sub(r"\\(cite[a-z]*|ref|label|input|includegraphics|hspace|vspace)\*?(\[[^\]]*\])?\{[^}]*\}", " ", text)
    text = re.sub(r"\\begin\{[^}]*\}|\\end\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", text)
    text = re.sub(r"[${}\\]", " ", text)
    return len(re.findall(r"[A-Za-z][A-Za-z'\-]*", text))


def main():
    t = TEX.read_text()
    start = t.index(r"\section{Introduction}")
    end = t.index(r"\subsection{Limitations and Future Research}")
    body = t[start:end]

    print("== main text (Introduction .. end of Results/Discussion, pre-Limitations) ==")
    secs = re.split(r"\n\\section\{([^}]*)\}", body)
    total = 0
    intro = strip_env(secs[0], ["table", "figure", "tabular", "tabular*"])
    n_intro = words(intro)
    total += n_intro
    print(f"  {'Introduction':<45} {n_intro:>6}")
    for i in range(1, len(secs), 2):
        name, txt = secs[i], secs[i + 1]
        txt = strip_env(txt, ["table", "figure", "tabular", "tabular*", "longtable", "landscape", "minipage"])
        n = words(txt)
        total += n
        print(f"  {name:<45} {n:>6}")
    # limitations + conclusion are after the split point
    tail = t[end:]
    tail_txt = strip_env(tail, ["table", "figure", "tabular", "tabular*"])
    n_tail = words(tail_txt)
    total += n_tail
    print(f"  {'Limitations + Conclusion':<45} {n_tail:>6}")
    print(f"  {'TOTAL (Introduction..Conclusion)':<45} {total:>6}")
    print(f"\nHSR limit 4500 -> need to cut {max(0, total-4500)} words")


if __name__ == "__main__":
    main()
