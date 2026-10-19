"""Test 1: the threshold sweep.

What it stresses
  The one knob every semantic cache has. Raising the hit rate by loosening the
  similarity threshold raises the false-positive rate with it.

How it works
  1. The cache holds the Banking77 train split in VectorAI DB (one global
     collection, no filters).
  2. Every Banking77 test question is looked up once (top-1).
  3. For each threshold from 0.50 to 0.99: hit rate, false-positive rate, and
     wrong hits per 100 queries. A hit is wrong when the cached intent differs.
  4. The threshold that meets the correctness target is fitted on a
     calibration half and scored on the other half.
  5. Writes the sweep as CSV and draws the curve (both rates on one axis).
"""

import csv

import numpy as np

from common import (
    RESULTS, base_args, build_collection, connect, embed, load_banking77,
    lookup_all, pct, rates, save, split_halves, sweep, threshold_for_target,
)

COLLECTION = "fp_test1"

# Categorical slots 1 and 2 of the dataviz reference palette (validated as a pair).
# Swap for Aric's colours before publishing: Actian figures follow his identity.
HIT_COLOR = "#2a78d6"
FP_COLOR = "#eb6834"
MIN_HITS_PLOTTED = 20
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def plot(rows, op_thr, target_thr, fp_target, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [r["threshold"] for r in rows]
    hit = [100 * r["hit_rate"] for r in rows]
    # Hide the FP rate where too few hits remain for it to mean anything.
    fp = [100 * r["fp_rate"] if r["hits"] >= MIN_HITS_PLOTTED else float("nan") for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(t, hit, color=HIT_COLOR, lw=2, label="Hit rate")
    ax.plot(t, fp, color=FP_COLOR, lw=2, ls=(0, (5, 2)), label="False-positive rate")

    # Direct labels at the left end, so identity is never colour alone.
    ax.annotate("Hit rate", (t[0], hit[0]), xytext=(6, 6), textcoords="offset points",
                color=INK, fontsize=10)
    ax.annotate("False-positive rate", (t[0], fp[0]), xytext=(6, 6),
                textcoords="offset points", color=INK, fontsize=10)

        # Operating label sits left of its line, mid-height; target label sits right of
    # its line, near the top. The two lines are often close, so this keeps them apart.
    markers = [(op_thr, "operating", 50, -4, "right"),
               (target_thr, f"meets {fp_target:.0%} target", 92, 4, "left")]
    for thr, name, y, dx, ha in markers:
        if thr is None:
            continue
        ax.axvline(thr, color=INK_2, lw=1, ls=":")
        ax.annotate(f"{thr:.2f} ({name})", (thr, y), xytext=(dx, 0),
                    textcoords="offset points", color=INK_2, fontsize=9,
                    ha=ha, va="center")

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Similarity threshold", color=INK_2)
    ax.set_ylabel("Percent", color=INK_2)
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=INK_2)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.01), ncol=2,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    print(f"Saved {path}")


def main():
    args = base_args(__doc__.splitlines()[0]).parse_args()
    train, test = load_banking77(args)
    train_vecs = embed([r["text"] for r in train], "train")
    test_vecs = embed([r["text"] for r in test], "test")

    client = connect()
    build_collection(client, COLLECTION, train_vecs,
                     [{"text": r["text"], "intent": r["intent"]} for r in train])
    scores, hits = lookup_all(client, COLLECTION, test_vecs, label="global")
    correct = np.array([bool(h) and h["intent"] == r["intent"] for h, r in zip(hits, test)])

    rows = sweep(scores, correct)
    with open(RESULTS / "test1_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    at_op = rates(scores, correct, args.threshold)
    cal, ev = split_halves(len(test))
    target_thr = threshold_for_target(scores, correct, args.fp_target, mask=cal, min_hits=20)
    at_target = rates(scores, correct, target_thr, ev) if target_thr is not None else None

    # A few wrong hits at the operating threshold, for the article and the video hook.
    hit_mask = np.nan_to_num(scores, nan=-1) >= args.threshold
    wrong = [i for i in np.argsort(-scores) if hit_mask[i] and not correct[i]][:15]
    examples = [{"query": test[i]["text"], "query_intent": test[i]["intent"],
                 "cached": hits[i]["text"], "cached_intent": hits[i]["intent"],
                 "score": round(float(scores[i]), 4)} for i in wrong]

    print(f"\nAt operating threshold {args.threshold}: hit rate {pct(at_op['hit_rate'])}, "
          f"FP rate {pct(at_op['fp_rate'])}, "
          f"{at_op['wrong_hits_per_100_queries']:.1f} wrong answers per 100 queries")
    if at_target:
        print(f"Threshold for FP <= {pct(args.fp_target)}: {target_thr} -> hit rate "
              f"{pct(at_target['hit_rate'])}, FP rate {pct(at_target['fp_rate'])} (eval half)")
    else:
        print(f"No threshold in the grid meets FP <= {pct(args.fp_target)}")
    print("\nHighest-scoring wrong hits:")
    for e in examples[:5]:
        print(f"  {e['score']:.3f}  '{e['query']}'  <-  '{e['cached']}'")

    plot(rows, args.threshold, target_thr, args.fp_target, RESULTS / "test1_sweep.png")

    save("test1", {
        "test": 1,
        "threshold": args.threshold,
        "fp_target": args.fp_target,
        "at_operating_threshold": at_op,
        "threshold_for_target": target_thr,
        "eval_at_target_threshold": at_target,
        "wrong_hit_examples": examples,
        "sweep": rows,
        "row": {
            "test": "1. Threshold sweep",
            "stresses": "One global threshold: hit rate and wrong answers rise together",
            "fp_before": at_op["fp_rate"],
            "fp_before_detail": f"hit rate {pct(at_op['hit_rate'])} at {args.threshold}",
            "fp_after": at_target["fp_rate"] if at_target else None,
            "mitigation": (f"Threshold set against a {pct(args.fp_target)} FP target "
                           f"({target_thr}, hit rate {pct(at_target['hit_rate'])})"
                           if at_target else "No threshold meets the target alone"),
        },
    })


if __name__ == "__main__":
    main()
