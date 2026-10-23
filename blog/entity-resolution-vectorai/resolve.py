"""Entity resolution harness: VectorAI DB blocking, a similarity threshold, and Gemini 4 Argon on the ambiguous band.

    uv run resolve.py --dataset abt-buy                 # full run, threshold-only and with Argon
    uv run resolve.py --dataset abt-buy --no-argon      # blocking + threshold only, no API calls
    uv run resolve.py --dataset abt-buy --band-scale 2  # band-width experiment: double the band

Writes results/<dataset>/summary.json, candidates.csv, argon_verdicts.jsonl and score_distribution.png.
"""
import argparse
import csv
import json
import time
from pathlib import Path

from actian_vectorai import VectorAIClient

from er import bands, data
from er.block import blocking_quality, load_records, neighbours, returned_per_record, top_k
from er.embed import DEFAULT_MODEL, embed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(data.DATASETS), default="abt-buy")
    ap.add_argument("--k", type=int, default=10, help="neighbours per record kept as candidates")
    ap.add_argument("--sweep-k", type=int, nargs="+", default=[1, 3, 5, 10, 20],
                    help="K values reported in the blocking table")
    ap.add_argument("--hnsw-ef", type=int, default=256,
                    help="HNSW search breadth; keep well above K so the source filter does not starve results")
    ap.add_argument("--dev-share", type=float, default=0.3, help="share of left records used to tune edges")
    ap.add_argument("--target-precision", type=float, default=0.95, help="precision required above the upper edge")
    ap.add_argument("--max-miss", type=float, default=0.05, help="share of matches allowed below the lower edge")
    ap.add_argument("--near-tie-rank", type=int, default=3,
                    help="lowest-ranked candidate that can enter the band as a near-tie")
    ap.add_argument("--near-tie-gap", type=float, default=0.03,
                    help="a rank 2..R candidate enters the band if it scores within this of its record's best")
    ap.add_argument("--band-scale", type=float, default=1.0, help="widen (>1) or narrow (<1) the band")
    ap.add_argument("--embed-model", default=DEFAULT_MODEL)
    ap.add_argument("--embed-backend", choices=["ollama", "sentence-transformers"], default="ollama")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--argon-model", default="gemini-4-argon")
    ap.add_argument("--argon-backend", choices=["ai-studio", "agent-platform", "openrouter"], default="ai-studio",
                    help="ai-studio: GEMINI_API_KEY; agent-platform: Google Cloud (GOOGLE_CLOUD_API_KEY, or ADC "
                         "with GOOGLE_CLOUD_PROJECT); openrouter: prepaid credit, OPENROUTER_API_KEY")
    ap.add_argument("--thinking-level", choices=["LOW", "MEDIUM", "HIGH", "default"], default="LOW",
                    help="Argon's thinking level; thinking tokens are billed, LOW suits a match/no-match call")
    ap.add_argument("--no-argon", action="store_true", help="skip the Argon pass (no API calls)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent Argon requests")
    ap.add_argument("--budget-usd", type=float, default=10.0,
                    help="hard cap on total Argon spend across all runs (counted from the verdict logs)")
    ap.add_argument("--max-argon-pairs", type=int, default=3000,
                    help="stop before calling Argon if the band holds more pairs than this")
    ap.add_argument("--price-in", type=float, default=2.00, help="USD per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=10.00, help="USD per 1M output tokens")
    ap.add_argument("--host", default="localhost:6574")
    args = ap.parse_args()

    out = Path("results") / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    timings = {}

    # 1. Load the benchmark
    ds = data.load(args.dataset)
    print(f"{args.dataset}: {len(ds.left)} {ds.spec.left_name} + {len(ds.right)} {ds.spec.right_name} records, "
          f"{len(ds.gold)} true matches")

    # 2. Embed every record locally
    t0 = time.perf_counter()
    vectors = embed(list(ds.left["text"]) + list(ds.right["text"]), args.embed_model, out / "cache",
                    backend=args.embed_backend, ollama_url=args.ollama_url)
    left_vectors = vectors[:len(ds.left)]
    timings["embed_s"] = time.perf_counter() - t0

    # 3. Block with VectorAI DB: K nearest neighbours from the other source
    t0 = time.perf_counter()
    collection = f"er_{args.dataset.replace('-', '_')}"
    with VectorAIClient(args.host) as db:
        load_records(db, collection, ds, vectors)
        ranked = neighbours(db, collection, ds, left_vectors, max(args.sweep_k + [args.k]), args.hnsw_ef)
    timings["block_s"] = time.perf_counter() - t0

    sweep = [{"k": k, **blocking_quality(top_k(ranked, k), ds), "returned": returned_per_record(ranked, k)}
             for k in sorted(set(args.sweep_k + [args.k]))]
    print(f"\nBlocking (all records, hnsw_ef={args.hnsw_ef}):")
    for row in sweep:
        ret = row["returned"]
        print(f"  K={row['k']:<3} candidates={row['candidate_pairs']:>7,}  "
              f"reduction ratio={row['reduction_ratio']:.4f}  pair completeness={row['pair_completeness']:.4f}  "
              f"returned/record: mean {ret['mean']:.1f}, min {ret['min']}, short {ret['records_short']}")
    candidates = top_k(ranked, args.k)

    # 4. Split, tune on dev, score on test
    dev_ids, test_ids = bands.split(ds.left["id"], args.dev_share)
    dev_cand, dev_gold = bands.subset(candidates, ds.gold, dev_ids)
    test_cand, test_gold = bands.subset(candidates, ds.gold, test_ids)

    info = bands.rank_and_gap(candidates)
    dev_best, test_best = bands.best_match(dev_cand, info), bands.best_match(test_cand, info)

    # Naive reference: one global cutoff over every candidate (all K per record compete)
    t_global = bands.best_threshold(dev_cand, dev_gold)
    global_only = bands.prf({p for p, s in test_cand.items() if s >= t_global}, test_gold)

    # Baseline: best-match threshold, each record's nearest candidate if it clears the cutoff
    t_best = bands.best_threshold(dev_best, dev_gold)
    threshold_pred = {p for p, s in test_best.items() if s >= t_best}
    threshold_only = bands.prf(threshold_pred, test_gold)

    # Bands: edges tuned on dev best matches; the band adds near-ties ranked 2..R
    lower, upper = bands.band_edges(dev_best, dev_gold, args.target_precision, args.max_miss)
    lower, upper = bands.widen(lower, upper, args.band_scale)
    lower = min(lower, t_best)   # never auto-reject a pair the baseline would accept
    auto_match = {p for p, s in test_best.items() if s >= upper}
    band = bands.near_tie_band(test_cand, info, lower, upper, args.near_tie_rank, args.near_tie_gap)
    errors = (threshold_pred ^ test_gold) & test_cand.keys()     # baseline mistakes among candidates
    band_errors = errors & set(band)
    recall_ceiling = len((auto_match | set(band)) & test_gold) / len(test_gold)   # if Argon were always right

    print(f"\nTest split: {len(test_cand):,} candidates, {len(test_gold)} true matches")
    print(f"  Global cutoff, all candidates (reference, {t_global:.4f}): P={global_only['precision']:.4f} "
          f"R={global_only['recall']:.4f} F1={global_only['f1']:.4f}")
    print(f"  Best-match threshold (baseline, {t_best:.4f}): P={threshold_only['precision']:.4f} "
          f"R={threshold_only['recall']:.4f} F1={threshold_only['f1']:.4f}")
    print(f"  Band [{lower:.4f}, {upper:.4f}) + near-ties (rank <= {args.near_tie_rank}, gap <= {args.near_tie_gap}): "
          f"{len(band):,} pairs ({len(band) / len(test_cand):.1%} of candidates), "
          f"holding {len(band_errors)}/{len(errors)} of the baseline's errors; recall ceiling {recall_ceiling:.4f}")

    summary = {
        "dataset": args.dataset, "records": len(ds.left) + len(ds.right),
        "test_records": len(test_ids) + len(ds.right), "gold_matches": len(ds.gold),
        "embed_model": args.embed_model, "k": args.k, "hnsw_ef": args.hnsw_ef, "blocking_sweep": sweep,
        "dev_share": args.dev_share,
        "global_cutoff": t_global, "global_cutoff_all_candidates": global_only,
        "best_threshold": t_best, "threshold_only": threshold_only,
        "band": {"lower": lower, "upper": upper, "scale": args.band_scale,
                 "near_tie_rank": args.near_tie_rank, "near_tie_gap": args.near_tie_gap,
                 "pairs": len(band), "share_of_candidates": len(band) / len(test_cand),
                 "auto_match": len(auto_match), "recall_ceiling": recall_ceiling,
                 "threshold_errors": len(errors), "threshold_errors_in_band": len(band_errors)},
    }

    # 5. Argon on the band only
    verdicts = {}
    if not args.no_argon and len(band) > args.max_argon_pairs:
        raise SystemExit(f"\nThe band holds {len(band):,} pairs, more than --max-argon-pairs {args.max_argon_pairs:,}. "
                         "Narrow it (--band-scale, --target-precision, --max-miss) or raise the limit.")
    if not args.no_argon and band:
        from er.adjudicate import Adjudicator, record_cost, spent_so_far

        already = spent_so_far(Path("results"), args.price_in, args.price_out)
        budget_left = args.budget_usd - already
        print(f"\nArgon budget: ${args.budget_usd:.2f} total, ${already:.4f} already spent, ${budget_left:.4f} left")
        if budget_left <= 0:
            raise SystemExit("Budget used up. Raise --budget-usd to continue; cached verdicts are reused for free.")

        text_left = dict(zip(ds.left["id"], ds.left["text"]))
        text_right = dict(zip(ds.right["id"], ds.right["text"]))
        t0 = time.perf_counter()
        verdicts = Adjudicator(args.argon_model, out / "argon_verdicts.jsonl", args.workers,
                               backend=args.argon_backend,
                               thinking_level=None if args.thinking_level == "default" else args.thinking_level).run(
            band, text_left, text_right, budget_left, args.price_in, args.price_out)
        timings["argon_s"] = time.perf_counter() - t0

        if len(verdicts) < len(band):
            summary["argon_incomplete"] = {"decided": len(verdicts), "band": len(band)}
            summary["timings_s"] = {k: round(v, 2) for k, v in timings.items()}
            (out / "summary.json").write_text(json.dumps(summary, indent=2))
            raise SystemExit(f"\nArgon decided {len(verdicts)} of {len(band)} band pairs before the budget ran out, "
                             "so no with-Argon accuracy is reported. Rerun with a higher --budget-usd to finish.")

        argon_pred = auto_match | {p for p, v in verdicts.items() if v["match"]}
        with_argon = bands.prf(argon_pred, test_gold)
        tokens_in = sum(v["tokens_in"] for v in verdicts.values())
        tokens_out = sum(v["tokens_out"] for v in verdicts.values())
        cost = sum(record_cost(v, args.price_in, args.price_out) for v in verdicts.values())
        band_correct = sum(v["match"] == (p in test_gold) for p, v in verdicts.items())
        summary["with_argon"] = with_argon
        summary["argon"] = {
            "model": args.argon_model, "backend": args.argon_backend, "thinking_level": args.thinking_level, "pairs": len(verdicts), "correct": band_correct,
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "price_per_1m": {"input": args.price_in, "output": args.price_out},
            "cost_usd": cost, "cost_per_1k_test_records_usd": cost / (summary["test_records"] / 1000),
        }
        print(f"  With Argon: P={with_argon['precision']:.4f} R={with_argon['recall']:.4f} "
              f"F1={with_argon['f1']:.4f}  ({band_correct}/{len(verdicts)} band verdicts correct, ${cost:.4f})")

    summary["timings_s"] = {k: round(v, 2) for k, v in timings.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    with (out / "candidates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "left_id", "right_id", "score", "rank", "gap", "gold", "zone",
                    "argon_match", "argon_reason"])
        band_set = set(band)
        for (l, r), s in sorted(candidates.items(), key=lambda kv: -kv[1]):
            rank, gap = info[(l, r)]
            if l in dev_ids:
                zone = ""
            else:
                zone = "match" if (l, r) in auto_match else "band" if (l, r) in band_set else "reject"
            v = verdicts.get((l, r), {})
            w.writerow(["dev" if l in dev_ids else "test", l, r, f"{s:.6f}", rank, f"{gap:.6f}",
                        int((l, r) in ds.gold), zone, v.get("match", ""), v.get("reason", "")])

    plot_scores(test_cand, test_gold, lower, upper, t_best, out / "score_distribution.png", args.dataset)
    print(f"\nWrote {out}/summary.json, candidates.csv, score_distribution.png")


def plot_scores(cand, gold, lower, upper, cutoff, path, title):
    """Score distribution of true matches against non-matches, with the band edges and the best cutoff."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    match = [s for p, s in cand.items() if p in gold]
    other = [s for p, s in cand.items() if p not in gold]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    bins = 60
    ax.hist(other, bins=bins, alpha=0.75, color="#2a78d6", label="not a match")
    ax.hist(match, bins=bins, alpha=0.85, color="#eb6834", label="true match")
    ax.axvspan(lower, upper, color="#8a8984", alpha=0.15, label="band sent to Argon")
    ax.axvline(cutoff, color="#0b0b0b", linewidth=1, linestyle="--", label="best single cutoff")
    ax.set_yscale("log")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Candidate pairs (log scale)")
    ax.set_title(f"Candidate scores, {title} test split", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    main()
