"""Test 4: partitions and per-intent thresholds.

What it stresses
  A single global cache with a single global threshold lets a hit cross a tenant
  or intent boundary. This test measures how often that happens, then measures
  the same lookups with VectorAI DB filters doing the partitioning:

    A  global          no filter
    B  tenant          MUST tenant == query tenant
    C  tenant+intent   MUST tenant, MUST intent == predicted intent
    D  tenant+intent2  MUST tenant, SHOULD intent == top-1 OR top-2 predicted intent

  Then it fits per-intent thresholds on a calibration half and compares them,
  on the evaluation half, with one global threshold tuned to the same target.

Tenants are assigned at random (seeded) to cache entries and queries, standing in
for separate customers. Serving another tenant's cached answer counts as wrong.
The intent classifier is a logistic regression on the cache's own embeddings,
which is the cheap classifier a real cache would run before lookup.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from common import (
    Field, FilterBuilder, assign_tenants, base_args, build_collection, connect,
    embed, load_banking77, lookup_all, pct, rates, save, split_halves, summarize,
    sweep, threshold_for_target,
)

COLLECTION = "fp_test4"


def per_intent_thresholds(scores, correct, hit_intent, cal, target, fallback, min_hits=3):
    """For each cached intent, the lowest threshold meeting the target on the calibration half."""
    thresholds = {}
    for intent in sorted(set(x for x in hit_intent if x)):
        m = cal & (hit_intent == intent)
        chosen = None
        for r in sweep(scores, correct, m):
            if r["hits"] < min_hits:
                break  # not enough evidence at this threshold or above
            if r["fp_rate"] <= target:
                chosen = r["threshold"]
                break
        thresholds[intent] = chosen if chosen is not None else fallback
    return thresholds


def apply_per_intent(scores, hit_intent, thresholds, mask):
    s = np.nan_to_num(scores, nan=-1.0)
    thr = np.array([thresholds.get(x, 2.0) for x in hit_intent])
    return mask & (s >= thr)


def main():
    args = base_args(__doc__.splitlines()[0]).parse_args()
    train, test = load_banking77(args)
    train_vecs = embed([r["text"] for r in train], "train")
    test_vecs = embed([r["text"] for r in test], "test")

    train_tenant = assign_tenants(len(train), seed=1)
    test_tenant = assign_tenants(len(test), seed=2)
    q_intent = np.array([r["intent"] for r in test])
    q_tenant = np.array(test_tenant)

    # Intent classifier used for partitioning at lookup time.
    clf = LogisticRegression(max_iter=3000)
    clf.fit(train_vecs, [r["intent"] for r in train])
    proba = clf.predict_proba(test_vecs)
    order = np.argsort(-proba, axis=1)
    pred1 = clf.classes_[order[:, 0]]
    pred2 = clf.classes_[order[:, 1]]
    clf_acc = float((pred1 == q_intent).mean())
    clf_top2 = float(((pred1 == q_intent) | (pred2 == q_intent)).mean())
    print(f"Intent classifier accuracy: top-1 {pct(clf_acc)}, top-2 {pct(clf_top2)}")

    client = connect()
    payloads = [{"text": r["text"], "intent": r["intent"], "tenant": t}
                for r, t in zip(train, train_tenant)]
    build_collection(client, COLLECTION, train_vecs, payloads)

    filters = {
        "A_global": None,
        "B_tenant": lambda i: FilterBuilder().must(Field("tenant").eq(test_tenant[i])).build(),
        "C_tenant_intent": lambda i: (FilterBuilder()
                                      .must(Field("tenant").eq(test_tenant[i]))
                                      .must(Field("intent").eq(str(pred1[i])))
                                      .build()),
        "D_tenant_intent_top2": lambda i: (FilterBuilder()
                                           .must(Field("tenant").eq(test_tenant[i]))
                                           .should(Field("intent").eq(str(pred1[i])))
                                           .should(Field("intent").eq(str(pred2[i])))
                                           .build()),
    }

    cal, ev = split_halves(len(test))
    configs = {}
    for name, fn in filters.items():
        scores, hits = lookup_all(client, COLLECTION, test_vecs, fn, label=name)
        hit_intent = np.array([p["intent"] if p else "" for p in hits])
        hit_tenant = np.array([p["tenant"] if p else "" for p in hits])
        intent_ok = hit_intent == q_intent
        tenant_ok = hit_tenant == q_tenant
        correct = intent_ok & tenant_ok

        at_op = rates(scores, correct, args.threshold)
        hit_mask = np.nan_to_num(scores, nan=-1) >= args.threshold
        cross_tenant = int((hit_mask & ~tenant_ok).sum())
        cross_intent = int((hit_mask & ~intent_ok).sum())
        at_op["cross_tenant_hits"] = cross_tenant
        at_op["cross_intent_hits"] = cross_intent
        at_op["intent_fp_rate"] = cross_intent / at_op["hits"] if at_op["hits"] else 0.0
        at_op["cross_tenant_share"] = cross_tenant / at_op["hits"] if at_op["hits"] else 0.0

        # Same correctness target two ways, fitted on cal half, scored on eval half.
        g_thr = threshold_for_target(scores, correct, args.fp_target, mask=cal, min_hits=20)
        g_eval = rates(scores, correct, g_thr, ev) if g_thr is not None else None
        pi = per_intent_thresholds(scores, correct, hit_intent, cal, args.fp_target,
                                   fallback=g_thr if g_thr is not None else 0.99)
        pi_eval = summarize(apply_per_intent(scores, hit_intent, pi, ev), correct, ev)

        configs[name] = {
            "at_operating_threshold": at_op,
            "sweep": sweep(scores, correct),
            "global_threshold_for_target": g_thr,
            "global_threshold_eval_half": g_eval,
            "per_intent_eval_half": pi_eval,
            "per_intent_thresholds": pi,
        }
        print(f"\n{name} @ {args.threshold}: hit rate {pct(at_op['hit_rate'])}, "
              f"FP rate {pct(at_op['fp_rate'])} "
              f"(cross-tenant {cross_tenant}, cross-intent {cross_intent} of {at_op['hits']} hits)")
        if g_eval:
            print(f"  target FP<={pct(args.fp_target)}: global thr {g_thr} -> hit rate "
                  f"{pct(g_eval['hit_rate'])}, FP {pct(g_eval['fp_rate'])} | per-intent -> "
                  f"hit rate {pct(pi_eval['hit_rate'])}, FP {pct(pi_eval['fp_rate'])}")

    a = configs["A_global"]["at_operating_threshold"]
    c = configs["C_tenant_intent"]["per_intent_eval_half"]
    save("test4", {
        "test": 4,
        "threshold": args.threshold,
        "fp_target": args.fp_target,
        "classifier_top1_accuracy": clf_acc,
        "classifier_top2_accuracy": clf_top2,
        "configs": configs,
        "row": {
            "test": "4. Partitions and per-intent thresholds",
            "stresses": "Hits crossing tenant and intent boundaries under one global threshold",
            # Intent-only FP: the cross-tenant share is structural with random tenants
            # (about 1 - 1/num_tenants), so it is reported separately, not in the row.
            "fp_before": a["intent_fp_rate"],
            "fp_before_detail": f"plus {pct(a['cross_tenant_share'])} of hits cross-tenant",
            "fp_after": c["fp_rate"],
            "mitigation": "MUST filters on tenant and predicted intent in VectorAI DB, "
                          "per-intent thresholds fitted to the correctness target",
        },
    })


if __name__ == "__main__":
    main()
