#!/usr/bin/env python3
"""Figure 1: Diagnostic framework - three levels of value sensitivity testing.

Statistics are the canonical values (2026-09 rerun):
  H1 pooled bivariate  : beta1 = -2.14, 95% CI [-6.23, 1.95], R^2 = 0.090  (N = 15 endpoints)
  H2 frame-adjusted    : access +2.36 (p=.020), collective +0.69 (n.s.), coercive +7.26 (p<.001), N = 2,019
  H3 pooled (4 models) : autonomy +4.96 [3.57, 6.35], collective -10.88 [-12.54, -9.21],
                         equity +4.84 [3.45, 6.24], adj. R^2 = 0.77 (N = 2,159; 108 clusters)
Regenerate after any rerun; do not hand-edit the PDF.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).with_name("fig1_conceptual.pdf")
plt.rcParams.update({"font.family": "DejaVu Sans"})

H1_BODY = [r"$\mathrm{PVOC}_m \;\rightarrow\; \Delta_m$",
           "Aggregate PVOC index predicts", "cross-model burden effects", "",
           r"$N$ = 15 model endpoints",
           r"$\beta_1$ = $-2.14$, 95% CI [$-6.23$, 1.95]", r"$R^2$ = 0.09"]
H2_BODY = [r"$\mathrm{Frame}_{mi} \;\rightarrow\; \Delta_{mi}$",
           "Interpretive frames correlate with", "predicted burden effects", "",
           "N = 2,019 rationale\u2013effect observations",
           r"Access barriers $\uparrow$ ($b$ = 2.36)",
           "Co. responsibility n.s. ($b$ = 0.69)",
           r"Coercive backlash $\uparrow$ ($b$ = 7.26)"]
H3_BODY = [r"$\Delta_{mcir} \;\approx\; \mathrm{Frame}_c$",
           "Assigned governance frame shifts",
           "predicted effect within the same model", "",
           r"$N$ = 4 models $\times$ 27 profiles $\times$ 4 frames",
           r"Autonomy $\uparrow$ (+4.96), equity $\uparrow$ (+4.84)",
           r"Collective obligation $\downarrow$ ($-$10.88)",
           r"adj. $R^2$ = 0.77"]

COLS = [
    ("#1f4e79", "#dce9f5", "Model-level", "H1: Composite orientation", H1_BODY,
     "Null: no detectable association"),
    ("#8c1d1d", "#f7dede", "Narrative-level", "H2: Narrative frame association", H2_BODY,
     "Partial support: 2 of 3 frames"),
    ("#1d6b3f", "#d9eedd", "Within-model", "H3: Frame manipulation", H3_BODY,
     "Supported (pooled; model heterogeneity)"),
]

fig, ax = plt.subplots(figsize=(9.8, 4.3))
ax.set_xlim(0, 35); ax.set_ylim(0, 14); ax.axis("off")

box_w, gap = 10.6, 0.9
for i, (edge, fill, level, title, lines, verdict) in enumerate(COLS):
    x = 0.7 + i * (box_w + gap)
    ax.add_patch(FancyBboxPatch((x, 3.4), box_w, 9.6, boxstyle="round,pad=0.15,rounding_size=0.25",
                                linewidth=1.4, edgecolor=edge, facecolor=fill))
    ax.add_patch(Rectangle((x, 12.1), box_w, 0.9, facecolor=edge, edgecolor="none"))
    ax.text(x + box_w / 2, 12.55, level, ha="center", va="center", fontsize=13,
            color="white", fontweight="bold")
    ax.text(x + box_w / 2, 11.35, title, ha="center", va="center", fontsize=10.5,
            color=edge, fontweight="bold")
    ax.plot([x + 0.7, x + box_w - 0.7], [10.85, 10.85], color=edge, linewidth=0.8)
    ax.text(x + box_w / 2, 10.6, lines[0], ha="center", va="top", fontsize=11.5, color="#222222")
    for j, ln in enumerate(lines[1:], start=1):
        ax.text(x + box_w / 2, 9.95 - 0.665 * j, ln, ha="center", va="center", fontsize=10.2, color="#333333")
    ax.add_patch(FancyBboxPatch((x + 0.5, 3.75), box_w - 1.0, 1.05,
                                boxstyle="round,pad=0.1,rounding_size=0.15",
                                linewidth=1.0, edgecolor=edge, facecolor="white"))
    ax.text(x + box_w / 2, 4.27, verdict, ha="center", va="center", fontsize=10.2,
            color=edge, fontweight="bold")
    ax.add_patch(FancyArrowPatch((x + box_w / 2, 3.4), (x + box_w / 2, 3.02), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=1.1, color=edge, shrinkA=0, shrinkB=0))

ax.add_patch(FancyBboxPatch((5.5, 2.05), 24, 0.95, boxstyle="round,pad=0.15,rounding_size=0.2",
                            linewidth=1.2, edgecolor="#333333", facecolor="#f2f2f2"))
ax.text(17.5, 2.52, "Does normative-frame sensitivity affect LLM-assisted policy simulation?",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#222222")
ax.text(17.5, 1.45, "Three independent diagnostic tests \u2014 each informative regardless of the others.",
        ha="center", va="center", fontsize=10, color="#444444")
ax.text(17.5, 0.72, "PVOC null (H1)  \u2192  frames partially matter (H2)  \u2192  causal manipulation confirms (H3)",
        ha="center", va="center", fontsize=10, color="#444444")

fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.99)
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"wrote {OUT} (+ preview png)")

QUESTION_BOX_BOTTOM = 2.05          # keep the note lines clear of this border


QUESTION_BOX_BOTTOM = 2.05          # keep the note lines clear of this border


def _check_layout():
    """Fail loudly if labels collide, leave the axes, or crowd the question box."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    a = fig.axes[0]
    ab = a.get_window_extent(r)
    boxes = [tx.get_window_extent(r) for tx in a.texts]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                ox = min(boxes[i].x1, boxes[j].x1) - max(boxes[i].x0, boxes[j].x0)
                oy = min(boxes[i].y1, boxes[j].y1) - max(boxes[i].y0, boxes[j].y0)
                assert ox < 1 or oy < 1, (
                    f"label collision: {a.texts[i].get_text()[:30]!r} x {a.texts[j].get_text()[:30]!r}")
    for tx, bb in zip(a.texts, boxes):
        assert bb.x0 >= ab.x0 - 1 and bb.x1 <= ab.x1 + 1, f"label outside axes: {tx.get_text()[:30]!r}"
        if tx.get_text().startswith(("Three independent", "PVOC null")):
            assert tx.get_position()[1] + 0.3 < QUESTION_BOX_BOTTOM, (
                f"note line crowds the question box: y={tx.get_position()[1]}")
    for p_ in a.patches:
        bb = p_.get_window_extent(r)
        assert bb.x0 >= ab.x0 - 1 and bb.x1 <= ab.x1 + 1, (
            f"patch clipped horizontally: {bb.x0:.0f}..{bb.x1:.0f} vs {ab.x0:.0f}..{ab.x1:.0f}")


if __name__ == "__main__":
    _check_layout()
    print("layout check passed")
