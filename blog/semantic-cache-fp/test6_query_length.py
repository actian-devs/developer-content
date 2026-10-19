"""Test 6: short against long queries.

What it stresses
  One threshold applied to every query length. Short queries carry little
  meaning, so unrelated short questions can score high; long queries can carry
  the same boilerplate and differ in one clause.

How it works
  Splits the Banking77 test queries into three equal-size buckets by word count
  (short, medium, long). For each bucket it reports the hit rate and FP rate at
  the operating threshold, and the threshold each bucket needs to meet the
  correctness target (fitted on a calibration half, scored on the other half).
"""

import numpy as np

from common import (
    base_args, build_collection, connect, embed, load_banking77, lookup_all, pct,
    rates, save, split_halves, sweep, threshold_for_target,
)

COLLECTION = "fp_test6"


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

    words = np.array([len(r["text"].split()) for r in test])
    q1, q2 = np.percentile(words, [33.33, 66.67])
    buckets = {
        f"short (<= {int(q1)} words)": words <= q1,
        f"medium ({int(q1) + 1}-{int(q2)} words)": (words > q1) & (words <= q2),
        f"long (> {int(q2)} words)": words > q2,
    }

    cal, ev = split_halves(len(test))
    out = {}
    for name, m in buckets.items():
        at_op = rates(scores, correct, args.threshold, m)
        thr = threshold_for_target(scores, correct, args.fp_target, mask=m & cal, min_hits=20)
        out[name] = {
            "queries": int(m.sum()),
            "at_operating_threshold": at_op,
            "threshold_for_target": thr,
            "eval_at_bucket_threshold": rates(scores, correct, thr, m & ev) if thr else None,
            "sweep": sweep(scores, correct, m),
        }
        print(f"{name}: {m.sum()} queries | @ {args.threshold}: hit rate {pct(at_op['hit_rate'])}, "
              f"FP rate {pct(at_op['fp_rate'])} | threshold for FP<={pct(args.fp_target)}: {thr}")

    names = list(buckets)
    short = out[names[0]]["at_operating_threshold"]["fp_rate"]
    long_ = out[names[-1]]["at_operating_threshold"]["fp_rate"]
    after = [out[n]["eval_at_bucket_threshold"] for n in names]
    after_fp = None
    if all(after):
        w = sum(a["wrong_hits"] for a in after)
        h = sum(a["hits"] for a in after)
        after_fp = w / h if h else 0.0

    save("test6", {
        "test": 6,
        "threshold": args.threshold,
        "fp_target": args.fp_target,
        "buckets": out,
        "row": {
            "test": "6. Short against long queries",
            "stresses": "One threshold across query lengths",
            "fp_before": max(short, long_),
            "fp_before_detail": f"short {pct(short)}, long {pct(long_)}",
            "fp_after": after_fp,
            "mitigation": "Length-aware thresholds fitted per bucket to the correctness target",
        },
    })


if __name__ == "__main__":
    main()
