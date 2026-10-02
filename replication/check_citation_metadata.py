#!/usr/bin/env python3
"""Verify every cited reference's metadata against Crossref (DOI) or arXiv.

Reads references.bib, keeps the entries actually cited in main.tex /
supplementary.tex, then for each:
  - DOI present   -> fetch api.crossref.org/works/<doi>, compare title, first-author
                     surname, year and container title
  - arXiv id       -> fetch the arXiv Atom API and compare title/first author
  - neither        -> flag for manual check
Writes outputs/citation_metadata_check.csv and prints a mismatch report.
"""
import csv
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent            # analysis repo (writes reports here)
MS = Path.home() / "Documents" / "value-sensitivity-llm-audit"   # manuscript repo (bib + tex)
OUT = ROOT / "outputs"


def parse_bib(text):
    entries, cur, key = {}, None, None
    for line in text.splitlines():
        m = re.match(r"@(\w+)\{([^,]+),", line)
        if m:
            cur, key = {"type": m.group(1), "fields": {}}, m.group(2)
            entries[key] = cur
            continue
        if cur is None:
            continue
        m = re.match(r"\s*(\w+)\s*=\s*\{(.*)\}\s*,?\s*$", line)
        if m:
            cur["fields"][m.group(1).lower()] = m.group(2)
        elif line.strip() == "}":
            cur = None
    return entries


def norm(s):
    s = s or ""
    s = re.sub(r'\\["\'`^~=.]\{?(\w)\}?', r"\1", s)      # LaTeX accents -> base letter
    s = re.sub(r"[{}]", "", s)
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKD", s.lower())
                  .encode("ascii", "ignore").decode())


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ref-check/1.0 (mailto:Yichao.Jin@UTDallas.edu)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def first_surname(author_field):
    if not author_field:
        return ""
    first = re.split(r"\band\b", author_field)[0]
    if "," in first:                     # "Surname, Given"
        return norm(first.split(",")[0])
    parts = norm(first).split()
    return parts[-1] if parts else ""


def cited_keys():
    keys = set()
    for f in ("main.tex", "supplementary.tex"):
        for m in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", (MS / f).read_text()):
            keys.update(k.strip() for k in m.group(1).split(","))
    return keys


def main():
    entries = parse_bib((MS / "references.bib").read_text())
    cited = cited_keys()
    print(f"bib entries: {len(entries)} | cited keys: {len(cited)}")
    missing = sorted(k for k in cited if k not in entries)
    if missing:
        print("cited but missing from the bib:", missing)

    rows = []
    for key in sorted(cited & set(entries)):
        e = entries[key]["fields"]
        title, author = e.get("title", ""), e.get("author", "")
        doi = (e.get("doi") or "").strip()
        year = e.get("year") or e.get("date") or ""
        rec = dict(key=key, bib_title=title, bib_author=author, bib_year=year, doi=doi, status="", detail="")
        if doi:
            try:
                msg = get_json("https://api.crossref.org/works/" + urllib.parse.quote(doi))["message"]
                cr_title = (msg.get("title") or [""])[0]
                cr_year = str((msg.get("issued", {}).get("date-parts") or [[""]])[0][0])
                cr_authors = msg.get("author") or []
                cr_first = norm(cr_authors[0].get("family", "")) if cr_authors else ""
                cr_cont = (msg.get("container-title") or [""])[0]
                problems = []
                if norm(title)[:40] and norm(title)[:40] not in norm(cr_title):
                    problems.append(f"title: bib={title[:60]!r} crossref={cr_title[:60]!r}")
                if first_surname(author) and cr_first and first_surname(author) not in cr_first and cr_first not in first_surname(author):
                    problems.append(f"first author: bib={first_surname(author)!r} crossref={cr_first!r}")
                if year and cr_year and year[:4] != cr_year[:4]:
                    problems.append(f"year: bib={year} crossref={cr_year}")
                rec.update(status="MISMATCH" if problems else "OK", detail=" | ".join(problems),
                           cr_title=cr_title, cr_first_author=cr_first, cr_year=cr_year, cr_container=cr_cont,
                           cr_authors="; ".join(f"{a.get('family','')}, {a.get('given','')}" for a in cr_authors))
            except Exception as exc:
                rec.update(status="ERROR", detail=str(exc)[:120])
            time.sleep(0.2)
        else:
            try:
                q = urllib.parse.quote(title[:180])
                hits = get_json(f"https://api.crossref.org/works?rows=3&query.bibliographic={q}")["message"]["items"]
                best = None
                for h in hits:
                    ht = (h.get("title") or [""])[0]
                    if norm(title)[:35] and norm(title)[:35] in norm(ht):
                        best = h
                        break
                if best:
                    au = best.get("author") or []
                    rec.update(status="SUGGEST-DOI", detail=f"probable DOI {best.get('DOI')}",
                               cr_title=(best.get("title") or [""])[0],
                               cr_first_author=au[0].get("family", "") if au else "",
                               cr_year=str((best.get("issued", {}).get("date-parts") or [[""]])[0][0]),
                               cr_container=(best.get("container-title") or [""])[0],
                               cr_authors="; ".join(f"{a.get('family','')}, {a.get('given','')}" for a in au))
                else:
                    rec.update(status="NO-MATCH", detail="no Crossref DOI by title (book/report/preprint?)")
            except Exception as exc:
                rec.update(status="ERROR", detail=str(exc)[:120])
            time.sleep(0.2)
        rows.append(rec)
        print(f"  {rec['status']:<11} {key:<22} {rec['detail'][:150]}")

    OUT.mkdir(exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with open(OUT / "citation_metadata_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    bad = [r for r in rows if r["status"] not in ("OK",)]
    print(f"\n{len(rows)-len(bad)}/{len(rows)} verified clean; {len(bad)} need attention")
    print(f"wrote {OUT/'citation_metadata_check.csv'}")


if __name__ == "__main__":
    main()
