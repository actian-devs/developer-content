# Is Jev Any Good for Reranking?

A quick benchmark of Jev (TypeSafe) against a cross-encoder and decider-4b (local, served by Ollaya) on BEIR nfcorpus and SciFact, with Actian VectorAI DB as the first-stage retriever.

By default it uses the first 50 test queries per dataset, indexes 2,400 documents per dataset, and has each reranker score the top 30 hybrid candidates, so a full run fits on a laptop.

VectorAI DB Community Edition stores at most 5,000 vectors in total, so the default corpus subset keeps both datasets under that limit (4,800 vectors). The subset keeps every judged document for the chosen queries, then adds each query's top BM25 matches from the full corpus as hard distractors, the same idea as the NanoBEIR subsets. The script deletes its own older collections before indexing.

## Setup (once)

1. Install [uv](https://docs.astral.sh/uv/) and Docker.
2. Install dependencies and add your key:
   ```bash
   uv sync
   cp .env.example .env    # paste your TYPESAFE_API_KEY
   ```
3. Start VectorAI DB:
   ```bash
   docker compose up -d
   ```
4. Start Ollaya with decider-4b, in its own terminal:
   ```bash
   curl -fsSL https://ollaya.dev/install.sh | OLLAYA_INSTALL_DIR=$HOME/.local sh
   export PATH="$HOME/.local/bin:$PATH"
   ollaya pull decider:4b
   ollaya serve
   ```

## Run

Smoke test, a few minutes:

```bash
uv run benchmark.py --queries 3
```

Article run:

```bash
uv run benchmark.py
```

Results land in `results/`:

| File | Contents |
|---|---|
| `summary.md` | Lift, calibration, latency, cost, and adaptive top-k tables |
| `summary.json` | Every number plus hardware, package versions, and settings |
| `reliability_<dataset>.png` | Reliability curves |
| `scores_<dataset>_<reranker>.jsonl` | Raw per-pair scores, latency, and tokens |

If a run stops, run the same command again. Finished queries are reused, so Jev is never billed twice.

## Options

```bash
uv run benchmark.py --queries 100 --depth 50     # bigger run
uv run benchmark.py --datasets scifact           # one dataset
uv run benchmark.py --rerankers jev cross-encoder  # skip decider-4b
uv run benchmark.py --docs 0                     # full corpus (needs a VectorAI DB licence)
```

Every other setting (models, RRF constant, adaptive top-k thresholds, Jev price) sits in the `SETTINGS` block at the top of `benchmark.py`.

## Method

- First stage: dense search in VectorAI DB (`BAAI/bge-base-en-v1.5`) and BM25, each top 100, fused with the VectorAI DB SDK's reciprocal rank fusion (k = 60).
- Every reranker scores each query-document pair on its own and returns a probability between 0 and 1.
- nDCG uses trec_eval's linear gain, matching published BEIR results.
- Calibration counts a pair as relevant at label grade 1 or above; unjudged pairs count as not relevant.
- Adaptive top-k reranks blocks of 5 in first-stage order and stops once 5 candidates clear the threshold.
