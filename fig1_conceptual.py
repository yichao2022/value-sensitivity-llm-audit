#!/usr/bin/env python3
"""Figure 1: three-level audit of normative-frame sensitivity (journal typeset).

Statistics are the canonical values (2026-09 rerun), guarded by the companion checks:
  H1 pooled bivariate   : beta1 = -2.14, 95% CI [-6.23, 1.95], R^2 = 0.09   (N = 15 endpoints)
                          ->  outputs/canonical/h1_sample_comparison.csv
  H2 pair FE (primary)  : access +0.13, collective +0.15, coercive +1.21, all n.s. (BH p = 0.86),
                          N = 2,019  ->  outputs/canonical/h2_pairfe_main.csv
  H3 pooled (4 models)  : autonomy +4.96, equity/access +4.84, collective -10.88, adj. R^2 = 0.77
                          ->  outputs/h3_clustered_results.csv (main model + profile FE specification)
Regenerate after any rerun; never hand-edit the PDF.

The figure is typeset at its final printed size (\\textwidth = 6.5 in) so nothing is scaled
down on inclusion. All three panels draw from one shared row grid (see `slot()`), so headers,
titles, equations, questions, design lines, estimates and result labels sit at identical
heights; the diagnostics are parallel, not a causal sequence.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(__file__).with_name("fig1_conceptual.pdf")
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})

# ---------------------------------------------------------------- content
PANELS = [
    dict(level="Model-level", edge="#2f5d8a", tint="#eef4fa",
         title=["H1: Composite", "orientation"],
         equation=r"$\mathrm{PVOC}_m \;\rightarrow\; \overline{\Delta}_m$",
         question="Does aggregate policy-value orientation predict cross-model burden effects?",
         design=["N = 15 model endpoints"],
         estimates=[r"$\beta_1$ = $-2.14$, 95% CI [$-6.23$, 1.95]", r"$R^2$ = 0.09"],
         result="No aggregate signal"),
    dict(level="Narrative-level", edge="#96504e", tint="#faf0ef",
         title=["H2: Narrative frame", "association"],
         equation=r"$\mathrm{Frame}_{mir} \;\rightarrow\; \Delta_{mir}$",
         question="Do naturally expressed rationale frames track simulated burden effects?",
         design=["Model $\\times$ profile pair", "fixed effects", "N = 2,019"],
         estimates=["Access barriers: $b$ = 0.13, n.s.",
                    "Collective responsibility: $b$ = 0.15, n.s.",
                    "Coercive backlash: $b$ = 1.21, n.s."],
         result="No robust within-pair signal"),
    dict(level="Within-model", edge="#4a7a5c", tint="#eef6f0",
         title=["H3: Frame", "manipulation"],
         equation=r"$\mathrm{Assigned\ Frame}_c \;\rightarrow\; \Delta_{mcir}$",
         question="Does an assigned prompt-frame clause shift the simulated burden effect within the same model?",
         design=["4 models $\\times$ 27 profiles", "$\\times$ 4 frames"],
         estimates=["Autonomy: +4.96", "Equity/access: +4.84", "Collective obligation: $-10.88$"],
         result="Frame-sensitive",
         footnote="Model-specific heterogeneity"),
]
TITLE = "Three-Level Audit of Normative-Frame Sensitivity in LLM-Assisted Policy Simulation"
SYNTHESIS = ["Aggregate orientation: no signal   ·   Natural rationale frames: no robust within-pair signal",
             "Assigned framing: shifts simulated effects"]

# ---------------------------------------------------------------- grid
FIG_W, FIG_H = 6.5, 3.7           # final printed size, landscape, no down-scaling
X0, X1, GAP, PAD = 1.2, 98.8, 2.0, 2.0
PANEL_W = (X1 - X0 - 2 * GAP) / 3
Y_TOP, HEAD_H, ROW = 89.0, 4.8, 4.4

_slots, _y = {}, Y_TOP - 2.3


def slot(name, gap_before=0.0):
    """Register a shared row and return its y (rows run top-down, identical in every panel)."""
    global _y
    _y -= gap_before
    _slots[name] = _y
    _y -= ROW
    return _slots[name]


for _name, _gap in [("title1", 0), ("title2", 0), ("eq", 1.0), ("q1", 1.0), ("q2", 0), ("q3", 0),
                    ("d1", 1.0), ("d2", 0), ("d3", 0), ("e1", 1.0), ("e2", 0), ("e3", 0),
                    ("rule", 1.0), ("result", 0), ("foot", 0.6)]:
    slot(_name, _gap)
Y_BOT = _y - 2.5                  # panel bottom below the last shared row
Y_FIG_TITLE, Y_SYNTH = 97.4, Y_BOT - 4.2

FS_TITLE, FS_HEAD, FS_PT, FS_EQ, FS_Q, FS_SMALL, FS_EST, FS_RES = 7.8, 8.0, 7.4, 7.4, 6.5, 6.6, 7.0, 8.2
INK, INK_SOFT = "#1a1a1a", "#3f3f3f"


def wrap(text, width=36):
    import textwrap
    return textwrap.fill(text, width=width).split("\n")


def panel(ax, x, p):
    edge, tint = p["edge"], p["tint"]
    texts = []
    ax.add_patch(FancyBboxPatch((x, Y_BOT), PANEL_W, Y_TOP + HEAD_H - Y_BOT,
                                boxstyle="round,pad=0,rounding_size=0.5",
                                linewidth=1.1, edgecolor=edge, facecolor=tint, zorder=1))
    ax.add_patch(Rectangle((x, Y_TOP), PANEL_W, HEAD_H, facecolor=edge, edgecolor="none", zorder=2))
    cx = x + PANEL_W / 2

    def put(name, s, size, color=INK, weight="normal", style="normal"):
        texts.append(ax.text(cx, _slots[name], s, ha="center", va="center", fontsize=size,
                             color=color, fontweight=weight, style=style, zorder=3))

    texts.append(ax.text(cx, Y_TOP + HEAD_H / 2, p["level"], ha="center", va="center",
                         fontsize=FS_HEAD, color="white", fontweight="bold", zorder=3))
    for name, line in zip(("title1", "title2"), p["title"]):
        put(name, line, FS_PT, edge, "bold")
    put("eq", p["equation"], FS_EQ)
    for name, line in zip(("q1", "q2", "q3"), wrap(p["question"])):
        put(name, line, FS_Q, INK_SOFT)
    for name, line in zip(("d1", "d2", "d3"), p["design"]):
        put(name, line, FS_SMALL, INK_SOFT)
    for name, line in zip(("e1", "e2", "e3"), p["estimates"]):
        put(name, line, FS_EST)
    ax.plot([x + PAD, x + PANEL_W - PAD], [_slots["rule"]] * 2, color=edge, linewidth=0.7, zorder=3)
    put("result", p["result"], FS_RES, edge, "bold")
    if p.get("footnote"):
        put("foot", p["footnote"], FS_SMALL, INK_SOFT, style="italic")
    return texts, (x, Y_BOT, PANEL_W, Y_TOP + HEAD_H - Y_BOT)


fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

ax.text(50, Y_FIG_TITLE, TITLE, ha="center", va="center", fontsize=FS_TITLE, color=INK,
        fontweight="bold")
ax.plot([X0, X1], [Y_FIG_TITLE - 2.6, Y_FIG_TITLE - 2.6], color="#bdbdbd", linewidth=0.6)
all_texts, rects = [], []
for i, p in enumerate(PANELS):
    t, r = panel(ax, X0 + i * (PANEL_W + GAP), p)
    all_texts.append(t); rects.append(r)
for k, line in enumerate(SYNTHESIS):
    ax.text(50, Y_SYNTH - k * 4.4, line, ha="center", va="center", fontsize=FS_Q, color=INK_SOFT)

fig.subplots_adjust(left=0.004, right=0.996, bottom=0.005, top=0.995)


def _check_layout():
    """Fail loudly if any text leaves its own panel, collides, or a shared row is missing."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(r)
    for ts, (x, y, w, h) in zip(all_texts, rects):
        px0, py0 = ax.transData.transform((x, y))
        px1, py1 = ax.transData.transform((x + w, y + h))
        for tx in ts:
            bb = tx.get_window_extent(r)
            assert bb.x0 >= px0 - 0.5 and bb.x1 <= px1 + 0.5, \
                f"text leaves its panel horizontally: {tx.get_text()[:44]!r}"
            assert bb.y0 >= py0 - 0.5 and bb.y1 <= py1 + 0.5, \
                f"text leaves its panel vertically: {tx.get_text()[:44]!r}"
    flat = [t for ts in all_texts for t in ts]
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            a, b = flat[i].get_window_extent(r), flat[j].get_window_extent(r)
            if a.overlaps(b):
                assert (min(a.x1, b.x1) - max(a.x0, b.x0) < 1
                        or min(a.y1, b.y1) - max(a.y0, b.y0) < 1), \
                    f"label collision: {flat[i].get_text()[:30]!r} x {flat[j].get_text()[:30]!r}"
    for tx in ax.texts:
        bb = tx.get_window_extent(r)
        assert bb.x0 >= ax_bb.x0 - 1 and bb.x1 <= ax_bb.x1 + 1, \
            f"text outside the figure: {tx.get_text()[:30]!r}"
    for tx in ax.texts:
        bb = tx.get_window_extent(r)
        assert bb.y0 >= ax_bb.y0 - 0.5 and bb.y1 <= ax_bb.y1 + 0.5, \
            f"text clipped vertically: {tx.get_text()[:34]!r}"
    for name in ("title1", "eq", "e1", "result"):
        ys = {round(t.get_position()[1], 6) for ts in all_texts for t in ts}
        assert round(_slots[name], 6) in ys, f"row {name} is not drawn in every panel"


if __name__ == "__main__":
    _check_layout()
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=600)
    print(f"wrote {OUT} (+600 dpi png); layout check passed; {FIG_W}x{FIG_H} in")
