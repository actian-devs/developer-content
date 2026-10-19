"""Test 7: stale answers after the source changes.

What it stresses
  A cached answer can match the question perfectly and still be wrong, because
  the policy or data behind it changed after it was cached. Similarity cannot
  see this, so no threshold fixes it.

How it works
  1. Each Banking77 intent stands for one source document (a policy page).
     Every cached entry records its source_id and is_stale = false.
  2. A seeded fraction of sources changes (default 20%). Every answer cached
     from those sources is now stale.
  3. Lookups run without invalidation: a hit on a changed source counts as wrong
     even when the intent matches.
  4. Invalidation: one set_payload call with a filter marks every entry of the
     changed sources is_stale = true, and lookups add MUST NOT is_stale.
     Lookups run again.

The stale FP rate scales with the share of traffic that hits changed sources,
so the article should state the change fraction beside the number.
"""

import random

import numpy as np

from common import (
    Field, FilterBuilder, base_args, build_collection, connect, embed,
    load_banking77, lookup_all, pct, rates, save,
)

COLLECTION = "fp_test7"


def main():
    p = base_args(__doc__.splitlines()[0])
    p.add_argument("--change-frac", type=float, default=0.20,
                   help="Fraction of sources (intents) whose answer changes.")
    args = p.parse_args()

    train, test = load_banking77(args)
    train_vecs = embed([r["text"] for r in train], "train")
    test_vecs = embed([r["text"] for r in test], "test")

    intents = sorted({r["intent"] for r in train})
    changed = set(random.Random(7).sample(intents, round(args.change_frac * len(intents))))
    print(f"{len(changed)} of {len(intents)} sources change after caching")

    client = connect()
    build_collection(client, COLLECTION, train_vecs,
                     [{"text": r["text"], "source_id": r["intent"], "is_stale": False}
                      for r in train])

    def correctness(hits):
        return np.array([bool(h) and h["source_id"] == r["intent"]
                         and h["source_id"] not in changed
                         for h, r in zip(hits, test)])

    # Before invalidation: the cache does not know the sources changed.
    scores_a, hits_a = lookup_all(client, COLLECTION, test_vecs, label="no invalidation")
    correct_a = correctness(hits_a)
    before = rates(scores_a, correct_a, args.threshold)
    hit_a = np.nan_to_num(scores_a, nan=-1) >= args.threshold
    stale_hits = int(sum(1 for i in np.where(hit_a)[0]
                         if hits_a[i]["source_id"] in changed
                         and hits_a[i]["source_id"] == test[i]["intent"]))
    before["stale_hits_right_intent"] = stale_hits

    # Invalidate every entry from a changed source with one filtered update.
    stale_filter = FilterBuilder().must(Field("source_id").any_of(sorted(changed))).build()
    client.points.set_payload(COLLECTION, payload={"is_stale": True}, filter=stale_filter)
    client.vde.flush(COLLECTION)
    marked = client.points.count(
        COLLECTION, filter=FilterBuilder().must(Field("is_stale").eq(True)).build())
    marked = int(getattr(marked, "count", marked))
    print(f"Marked {marked} cached entries stale")

    fresh_only = FilterBuilder().must_not(Field("is_stale").eq(True)).build()
    scores_b, hits_b = lookup_all(client, COLLECTION, test_vecs, lambda i: fresh_only,
                                  label="with invalidation")
    after = rates(scores_b, correctness(hits_b), args.threshold)

    print(f"\nNo invalidation @ {args.threshold}: hit rate {pct(before['hit_rate'])}, "
          f"FP rate {pct(before['fp_rate'])} ({stale_hits} hits had the right intent "
          f"but a stale answer)")
    print(f"With invalidation: hit rate {pct(after['hit_rate'])}, FP rate {pct(after['fp_rate'])}")

    save("test7", {
        "test": 7,
        "threshold": args.threshold,
        "change_fraction": args.change_frac,
        "changed_sources": sorted(changed),
        "entries_marked_stale": marked,
        "before": before,
        "after": after,
        "row": {
            "test": "7. Stale answers after the source changes",
            "stresses": f"Hits on answers whose source changed ({pct(args.change_frac)} of sources)",
            "fp_before": before["fp_rate"],
            "fp_after": after["fp_rate"],
            "mitigation": "Filtered set_payload marks a changed source's entries stale; "
                          "lookups add MUST NOT is_stale",
        },
    })


if __name__ == "__main__":
    main()
