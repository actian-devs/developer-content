"""Test 5: a local decider-4b verification pass on cache hits.

What it stresses
  Whether a small local decision model can catch the wrong hits a threshold lets
  through, and what it costs in latency and in good hits thrown away.

How it works
  1. Global cache, no filter, operating threshold (same as test 1).
  2. Every hit goes to decider-4b with both questions and one yes/no question:
     can one answer serve both? The hit is served only if p(yes) >= tau.
  3. Reports, per tau: wrong hits caught, good hits rejected, FP rate and hit
     rate after verification. Reports per-call latency p50 / p95 and the
     hardware it was measured on.

Runs on a CUDA GPU with bf16 (L4, A10G, A100, RTX 30xx/40xx). Apple MPS is not
verified by the model authors; it may work but measure latency there separately.
"""

import json
import random
import time
from pathlib import Path

import numpy as np

from common import (
    base_args, build_collection, connect, embed, hardware, load_banking77,
    lookup_all, pct, save, summarize,
)

COLLECTION = "fp_test5"
QUESTION = ("Are these two customer questions asking for the same thing, "
            "so that one answer would correctly serve both?")
TAUS = np.round(np.arange(0.10, 0.95, 0.05), 2)


def p_yes(result):
    probs = result.get("probs") or {}
    if "yes" in probs:
        return float(probs["yes"])
    conf = float(result.get("confidence", 0.5))
    return conf if result.get("choice") == "yes" else 1.0 - conf


def main():
    p = base_args(__doc__.splitlines()[0])
    p.add_argument("--model", default="Mapika/decider-4b")
    p.add_argument("--tau", type=float, default=0.5, help="Reported acceptance cut-off on p(yes).")
    p.add_argument("--max-hits", type=int, default=300,
                   help="Random sample of hits to verify (0 = all).")
    p.add_argument("--export-hits", metavar="PATH",
                   help="Step 1 (laptop with VectorAI DB): run lookups, save them to PATH, stop.")
    p.add_argument("--hits-file", metavar="PATH",
                   help="Step 2 (GPU machine, no database needed): verify the saved lookups.")
    args = p.parse_args()

    if args.hits_file:
        items = json.loads(Path(args.hits_file).read_text())
    else:
        train, test = load_banking77(args)
        train_vecs = embed([r["text"] for r in train], "train")
        test_vecs = embed([r["text"] for r in test], "test")
        client = connect()
        build_collection(client, COLLECTION, train_vecs,
                         [{"text": r["text"], "intent": r["intent"]} for r in train])
        sc, hp = lookup_all(client, COLLECTION, test_vecs, label="global")
        items = [{"query": r["text"], "query_intent": r["intent"],
                  "cached": h["text"] if h else "", "cached_intent": h["intent"] if h else "",
                  "score": float(s)} for r, h, s in zip(test, hp, sc)]
        if args.export_hits:
            Path(args.export_hits).write_text(json.dumps(items))
            print(f"Saved {len(items)} lookups to {args.export_hits}. "
                  f"Copy it to the GPU machine and run with --hits-file.")
            return

    scores = [it["score"] for it in items]
    hit_idx = [i for i, s in enumerate(scores) if s >= args.threshold]
    if args.max_hits and len(hit_idx) > args.max_hits:
        hit_idx = random.Random(42).sample(hit_idx, args.max_hits)
    print(f"{len(hit_idx)} hits at threshold {args.threshold} go to the verifier")

    from decider.infer import Decider
    decider = Decider(args.model)
    q = [{"question": QUESTION, "options": ["no", "yes"]}]

    def state(i):
        return (f"Cached question: {items[i]['cached']}\n"
                f"New question: {items[i]['query']}")

    for i in hit_idx[:5]:  # warm-up, not timed
        decider.decide(state(i), q)

    pys, lat, correct = [], [], []
    for n, i in enumerate(hit_idx, 1):
        t0 = time.perf_counter()
        r = decider.decide(state(i), q)[0]
        lat.append((time.perf_counter() - t0) * 1000)
        pys.append(p_yes(r))
        correct.append(items[i]["cached_intent"] == items[i]["query_intent"])
        if n % 200 == 0:
            print(f"  verified {n}/{len(hit_idx)}")
    pys, lat, correct = np.array(pys), np.array(lat), np.array(correct, bool)

    n_wrong = int((~correct).sum())
    n_good = int(correct.sum())
    all_q = np.ones(len(hit_idx), bool)
    before = summarize(all_q, correct, all_q, args.threshold)

    by_tau = []
    for tau in TAUS:
        keep = pys >= tau
        by_tau.append({
            "tau": float(tau),
            "wrong_hits_caught": int((~keep & ~correct).sum()),
            "wrong_hits_caught_rate": float((~keep & ~correct).sum() / n_wrong) if n_wrong else 0.0,
            "good_hits_rejected": int((~keep & correct).sum()),
            "good_hits_rejected_rate": float((~keep & correct).sum() / n_good) if n_good else 0.0,
            "fp_rate_after": float((keep & ~correct).sum() / keep.sum()) if keep.sum() else 0.0,
            "hits_kept": int(keep.sum()),
        })

    chosen = min(by_tau, key=lambda r: abs(r["tau"] - args.tau))
    latency = {"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
               "mean_ms": float(lat.mean()), "hardware": hardware(), "batch_size": 1}

    print(f"\nBefore verification: FP rate {pct(before['fp_rate'])} "
          f"({n_wrong} wrong of {len(hit_idx)} hits)")
    print(f"tau={chosen['tau']}: caught {chosen['wrong_hits_caught']}/{n_wrong} wrong hits "
          f"({pct(chosen['wrong_hits_caught_rate'])}), rejected "
          f"{chosen['good_hits_rejected']}/{n_good} good hits "
          f"({pct(chosen['good_hits_rejected_rate'])}), FP rate after {pct(chosen['fp_rate_after'])}")
    print(f"Latency per verification: p50 {latency['p50_ms']:.1f} ms, "
          f"p95 {latency['p95_ms']:.1f} ms on {latency['hardware']}")

    save("test5", {
        "test": 5,
        "threshold": args.threshold,
        "model": args.model,
        "question": QUESTION,
        "hits_verified": len(hit_idx),
        "before": before,
        "by_tau": by_tau,
        "chosen_tau": chosen,
        "latency": latency,
        "row": {
            "test": "5. Local verification with decider-4b",
            "stresses": "Wrong hits that pass the threshold; cost of checking them",
            "fp_before": before["fp_rate"],
            "fp_after": chosen["fp_rate_after"],
            "mitigation": f"decider-4b yes/no check on each hit (tau {chosen['tau']}), "
                          f"p50 {latency['p50_ms']:.0f} ms on {latency['hardware']}",
        },
    })


if __name__ == "__main__":
    main()
