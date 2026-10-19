"""Build the seven-test results table from results/test*.json.

Tests 1 to 3 appear if their JSON files exist with the same "row" shape:
{"row": {"test", "stresses", "fp_before", "fp_after", "mitigation"}}.
"""

import json

from common import RESULTS


def fmt(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main():
    rows = []
    for n in range(1, 8):
        path = RESULTS / f"test{n}.json"
        if not path.exists():
            rows.append(f"| {n}. (not run) | | | |")
            continue
        r = json.loads(path.read_text())["row"]
        detail = f" ({r['fp_before_detail']})" if r.get("fp_before_detail") else ""
        fp = f"{fmt(r['fp_before'])}{detail} → {fmt(r['fp_after'])}"
        rows.append(f"| {r['test']} | {r['stresses']} | {fp} | {r['mitigation']} |")

    table = "\n".join([
        "| Test | What it stresses | False-positive rate (before → after mitigation) | Mitigation |",
        "|---|---|---|---|",
        *rows,
    ])
    out = RESULTS / "results_table.md"
    out.write_text(table + "\n")
    print(table)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
