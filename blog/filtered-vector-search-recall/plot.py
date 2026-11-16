"""Plot recall at 10 by filter selectivity from recall.csv (written by measure.py)."""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
# categorical slots 1 to 4 of the reference palette, assigned in fixed order
SERIES = ["post_filter", "vectorai_filter"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LABELS = {"post_filter": "Post-filter", "vectorai_filter": "VectorAI DB filter"}


def label(method: str) -> str:
    if method in LABELS:
        return LABELS[method]
    if method.startswith("vectorai_ef"):
        return f"VectorAI DB, hnsw_ef={method[len('vectorai_ef'):]}"
    return f"VectorAI DB, {method[len('vectorai_overfetch'):]}x overfetch"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="recall.csv")
    ap.add_argument("--out", default="images/recall-by-selectivity.png")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    shares = sorted({float(r["share"]) for r in rows}, reverse=True)
    methods = SERIES + sorted({r["method"] for r in rows} - set(SERIES))
    methods = methods[: len(COLORS)]                          # four series at most, fold the rest into the CSV
    data = defaultdict(dict)
    for r in rows:
        data[r["method"]][float(r["share"])] = float(r["recall_at_k"])

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    xs = list(range(len(shares)))
    ends = []
    for m, c in zip(methods, COLORS):
        ys = [data[m].get(s, float("nan")) for s in shares]
        ax.plot(xs, ys, color=c, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=label(m), zorder=3)
        ends.append([ys[-1], label(m), c])
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):                             # nudge end labels apart so they never overlap
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.07)
    for y, text, c in ends:
        ax.annotate(text, (xs[-1], y), xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    ax.set_xticks(xs, [f"{s:.0%}" for s in shares], color=MUTED)
    ax.set_xlim(-0.2, len(xs) - 1 + 1.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1.0"], color=MUTED)
    ax.set_xlabel("Share of the catalog the filter admits", color=MUTED)
    ax.set_ylabel("Recall at 10", color=MUTED)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()