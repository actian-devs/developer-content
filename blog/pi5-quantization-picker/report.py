"""Build the fit, quality and picker tables from results/<model>/<level>.json.

  uv run report.py            # prints Markdown tables and writes results/tables.md and results/summary.csv
"""
import csv
import json
from pathlib import Path

from qpick.models import LEVELS, MODELS

SWAP_LIMIT_GIB = 0.25  # the fit limit run.py uses
HEADROOM_GIB = 0.10    # a pick must stay this far under the limit, so one more process doesn't tip it into swap


def load():
    """Read every result file the test run wrote, one per model and level."""
    rows = {}
    for m in MODELS:
        for level in LEVELS:
            p = Path("results") / m.key / f"{level}.json"
            if p.exists():
                rows[(m.key, level)] = json.loads(p.read_text())
    return rows


def pct(x):
    """Show a score like 0.85 as 85%, or n/a for a level that did not finish."""
    return "n/a" if x is None else f"{x:.0%}"


def main():
    """Turn the result files into the four tables the article uses, and save them with a CSV of every number."""
    rows = load()
    lines = []

    # ── Table 1: how the board's RAM was shared before any LLM loaded (the memory-bar figure) ──
    idle_path = Path("results/cache/idle.json")
    if idle_path.exists():
        idle = json.loads(idle_path.read_text())
        vec, oll = idle.get("peak_vectorai_gib", 0), idle.get("peak_ollama_gib", 0)
        lines += ["## The stack before any LLM loads", "",
                  f"Board: {idle['board']['board']}, {idle['board']['ram_gib']} GiB RAM, "
                  f"swap {idle['board']['swap_total_gib']} GiB. Retrieval: right chunk in the top 3 for "
                  f"{idle.get('retrieval_top3_hits')} answerable questions.", "",
                  "| Part | GiB |", "| --- | --- |",
                  f"| OS and everything else | {idle['peak_used_gib'] - vec - oll:.2f} |",
                  f"| VectorAI DB (4,000-chunk catalog) | {vec:.2f} |",
                  f"| Ollama with nomic-embed-text | {oll:.2f} |",
                  f"| Total in use | {idle['peak_used_gib']:.2f} |",
                  f"| Left for the LLM | {idle['board']['ram_gib'] - idle['peak_used_gib']:.2f} |", ""]

    # ── Table 2: what fits, one row per model and level ──
    lines += ["## What fits", "", "| Model | Level | File size (GiB) | Peak RAM used, whole board (GiB) | Swap growth (GiB) | Result |",
             "| --- | --- | --- | --- | --- | --- |"]
    for (key, level), r in rows.items():
        mem = r["memory"]
        lines.append(f"| {r['model']} | {level} | {r['file_gib']:.2f} | {mem['peak_used_gib']:.2f} | "
                     f"{mem['swap_growth_gib']:.2f} | {r['fit']} |")

    # ── Table 3: what holds up, the scores for each kind of prompt ──
    lines += ["", "## What holds up", "", "| Model | Level | RAG (of 20) | JSON (of 10) | Instructions (of 10) | Overall |",
              "| --- | --- | --- | --- | --- | --- |"]
    for (key, level), r in rows.items():
        lines.append(f"| {r['model']} | {level} | {r['rag_pass']} | {r['json_pass']} | {r['instruction_pass']} | "
                     f"{pct(r['quality'])} |")

    # ── Table 4: the picker, one recommended level and one fallback per model ──
    lines += ["", "## Picker", "",
              f"Pick = the highest level that matches the model's best score and grows swap by no more than "
              f"{SWAP_LIMIT_GIB - HEADROOM_GIB:.2f} GiB ({HEADROOM_GIB:.2f} GiB under the fit limit). "
              "Fallback = the next level down that does the same, or failing that the next level down with that headroom.", "",
              "| Model | Pick | Fallback | Score at pick |", "| --- | --- | --- | --- |"]
    for m in MODELS:
        if not any(k == m.key for k, _ in rows):
            lines.append(f"| {m.name} | not run yet | | |")
            continue
        # levels that fit with headroom to spare, smallest file first
        roomy = [(lv, rows[(m.key, lv)]) for lv in LEVELS
                 if (m.key, lv) in rows and rows[(m.key, lv)]["fit"] == "fits"
                 and rows[(m.key, lv)]["quality"] is not None
                 and rows[(m.key, lv)]["memory"]["swap_growth_gib"] <= SWAP_LIMIT_GIB - HEADROOM_GIB]
        if not roomy:
            lines.append(f"| {m.name} | none fits with headroom | | |")
            continue
        best = max(r["quality"] for _, r in roomy)                   # the model's best score among those levels
        top = [lv for lv, r in roomy if r["quality"] == best]        # every level that reaches it
        pick_lv = top[-1]                                            # the highest of those is the pick
        below = [lv for lv in top[:-1]] or [lv for lv, _ in roomy if LEVELS.index(lv) < LEVELS.index(pick_lv)]
        lines.append(f"| {m.name} | {pick_lv} | {below[-1] if below else 'none'} | {pct(best)} |")

    # ── Save the tables as Markdown, and every number as a CSV spreadsheet ──
    text = "\n".join(lines)
    print(text)
    Path("results/tables.md").write_text(text + "\n")
    with open("results/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "level", "file_gib", "peak_used_gib", "peak_llm_rss_gib", "peak_vectorai_gib",
                    "peak_ollama_gib", "swap_growth_gib", "load_s", "fit", "rag_pass", "json_pass",
                    "instruction_pass", "quality"])
        for (key, level), r in rows.items():
            mem = r["memory"]
            w.writerow([r["model"], level, r["file_gib"], mem["peak_used_gib"], mem["peak_llm_rss_gib"],
                        mem.get("peak_vectorai_gib"), mem.get("peak_ollama_gib"), mem["swap_growth_gib"],
                        r["load_s"], r["fit"], r["rag_pass"], r["json_pass"], r["instruction_pass"], r["quality"]])


if __name__ == "__main__":
    main()
