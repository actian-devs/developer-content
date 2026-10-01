"""Turn results/benchmark.csv into the p50/p99 latency chart used in the article."""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless Pi, no display needed
import matplotlib.pyplot as plt

src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/benchmark.csv")
rows = list(csv.DictReader(src.open()))
sizes = [int(r["vectors"]) for r in rows]
labels = [f"{n // 1000}K" for n in sizes]
p50 = [float(r["p50_ms"]) for r in rows]
p99 = [float(r["p99_ms"]) for r in rows]

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
x = range(len(sizes))
w = 0.38
b1 = ax.bar([i - w / 2 for i in x], p50, w, label="p50", color="#2a6fdb")
b2 = ax.bar([i + w / 2 for i in x], p99, w, label="p99", color="#e8833a")
ax.bar_label(b1, fmt="%.1f ms", fontsize=9)
ax.bar_label(b2, fmt="%.1f ms", fontsize=9)
ax.set_xticks(list(x), labels)
ax.set_xlabel("Vectors in collection (768-dim, cosine, top-5)")
ax.set_ylabel("Search latency (ms)")
ax.set_title("VectorAI DB retrieval latency on Raspberry Pi 5", loc="left", fontweight="bold")
ax.text(0, 1.01, f"{rows[0]['board']} · {rows[0]['ram_gb']} GB · {rows[0]['storage']} · "
        f"{rows[0]['queries']} queries per size", transform=ax.transAxes, fontsize=8, color="#555")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
ax.set_ylim(0, max(p99) * 1.25)
fig.tight_layout()

out = src.with_suffix(".png")
fig.savefig(out)
print(f"Wrote {out}")
