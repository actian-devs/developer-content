"""Build the fit, quality and picker tables from results/<model>/<level>.json.

  uv run report.py            # prints Markdown tables and writes results/tables.md and results/summary.csv
"""
import csv
import json
from pathlib import Path

from qpick.models import LEVELS, MODELS

QUALITY_SLACK = 0.05   # a level within 5 points of the model's best score counts as holding up


def load():
    rows = {}
    for m in MODELS:
        for level in LEVELS:
            p = Path("results") / m.key / f"{level}.json"
            if p.exists():
                rows[(m.key, level)] = json.loads(p.read_text())
    return rows


def pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def main():
    rows = load()
    lines = []
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
    lines += ["## What fits", "", "| Model | Level | File size (GiB) | Peak RAM used, whole board (GiB) | Swap growth (GiB) | Result |",
             "| --- | --- | --- | --- | --- | --- |"]
    for (key, level), r in rows.items():
        mem = r["memory"]
        lines.append(f"| {r['model']} | {level} | {r['file_gib']:.2f} | {mem['peak_used_gib']:.2f} | "
                     f"{mem['swap_growth_gib']:.2f} | {r['fit']} |")
    lines += ["", "## What holds up", "", "| Model | Level | RAG (of 20) | JSON (of 10) | Instructions (of 10) | Overall |",
              "| --- | --- | --- | --- | --- | --- |"]
    for (key, level), r in rows.items():
        lines.append(f"| {r['model']} | {level} | {r['rag_pass']} | {r['json_pass']} | {r['instruction_pass']} | "
                     f"{pct(r['quality'])} |")

    lines += ["", "## Picker (draft, from the rule below)", "",
              f"Pick = the highest level that fits and scores within {QUALITY_SLACK:.0%} of the model's best "
              "fitting score. Fallback = the next fitting level below it.", "",
              "| Model | Pick | Fallback | Score at pick |", "| --- | --- | --- | --- |"]
    for m in MODELS:
        fit = [(lv, rows[(m.key, lv)]) for lv in LEVELS
               if (m.key, lv) in rows and rows[(m.key, lv)]["fit"] == "fits" and rows[(m.key, lv)]["quality"] is not None]
        if not any(k == m.key for k, _ in rows):
            lines.append(f"| {m.name} | not run yet | | |")
            continue
        if not fit:
            lines.append(f"| {m.name} | none fits | | |")
            continue
        best = max(r["quality"] for _, r in fit)
        ok = [(lv, r) for lv, r in fit if r["quality"] >= best - QUALITY_SLACK]
        pick_lv, pick = ok[-1]
        lower = [lv for lv, _ in fit if LEVELS.index(lv) < LEVELS.index(pick_lv)]
        lines.append(f"| {m.name} | {pick_lv} | {lower[-1] if lower else 'none'} | {pct(pick['quality'])} |")

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
