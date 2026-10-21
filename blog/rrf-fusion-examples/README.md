# Actian RRF fusion examples

Reproducible Python examples for **“4 Things You'll Wish You Knew About Reciprocal Rank
Fusion.”** They show why rank fusion works across incompatible score scales, how `k` changes the
ranking, why parent-document deduplication belongs before fusion, and how RRF compares with
Actian's Distribution-Based Score Fusion (DBSF).

All retrieval lists and scores are deliberately **illustrative**. No live database retrieval is
claimed or required.

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install -e ".[dev]"
# macOS/Linux
.venv/bin/python -m pip install -e ".[dev]"
```

The runtime dependency is the official `actian-vectorai-client` package. Pytest and Ruff are the
only development dependencies.

## Run the examples

From the repository root, with the virtual environment activated:

```bash
python scripts/01_score_scale_example.py
python scripts/02_k_sensitivity.py
python scripts/03_parent_deduplication.py
python scripts/04_rrf_vs_dbsf.py
python -m pytest -q
python -m ruff check .
```

The first script reproduces the article's five-decimal RRF scores. The second shows the general
guide leading at `k=60` and `k=10`, tying the exact guide at `k=2`, and losing to it at `k=1`.
The third displays three intentionally distinct operations: fusing chunks, mapping and
deduplicating each source list before parent-level fusion, and merely limiting repeated parent
passages after chunk-level fusion. The last operation does not recompute parent scores.

## Official SDK comparison

`scripts/04_rrf_vs_dbsf.py` runs the installed Actian client's public functions over the same
three candidates for the illustrative query `ConnectionTimeoutError: pool exhausted`:

- `actian_vectorai.hybrid.fusion.reciprocal_rank_fusion` uses ranks and ignores raw score scale.
- `actian_vectorai.hybrid.fusion.distribution_based_score_fusion` normalizes each source's raw
  scores independently using a clipped `mean ± 3 * population standard deviation` range, then
  averages across all sources.

Both functions require lists of `actian_vectorai.models.ScoredPoint`, expect larger scores to be
better, identify duplicates by point ID, and return `ScoredPoint` values whose `score` is the
fused score. RRF also supports source weights; the article's compact implementation does not.

With these inputs both methods produce the same document order, but their scores have different
meanings. That agreement is specific to the illustrative distributions and is not a promise that
DBSF will favor the exact-match guide—or that RRF and DBSF generally agree.
