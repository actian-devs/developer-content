# Actian RRF fusion examples

Reproducible Python examples for **“4 Things You'll Wish You Knew About Reciprocal Rank
Fusion.”** The code matches the article's snippets, and each script prints the output the
article shows.

All retrieval lists and scores are deliberately **illustrative**. No live database retrieval is
claimed or required: the fusion functions run locally.

## Setup

Python 3.11 or newer is required.

```bash
git clone https://github.com/actian-devs/developer-content.git
cd developer-content/blog/rrf-fusion-examples
python -m venv .venv
```

Activate the environment on Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the project:

```bash
python -m pip install -e .
```

With [uv](https://docs.astral.sh/uv/), skip the steps above and prefix each command with
`uv run` (for example `uv run scripts/01_score_scale_example.py`). It creates the environment
and installs the project on the first run.

The runtime dependency is the official `actian-vectorai-client` package. Pytest and Ruff are the
only development dependencies (`pip install -e ".[dev]"`).

## Run the examples

One script per article section, in article order:

| Script | Article section |
| --- | --- |
| `scripts/01_score_scale_example.py` | The Scores Don’t Share a Scale, So You Fuse Ranks |
| `scripts/02_k_sensitivity.py` | The Default k=60 Can Bury a Strong Match / What Changing k Actually Does |
| `scripts/03_parent_deduplication.py` | Duplicate Chunks and Inconsistent IDs Affect RRF Results |
| `scripts/05_fusion_ranking_difference.py` | Treat RRF as a Floor, Then Move Past It |

```bash
python scripts/01_score_scale_example.py
python scripts/02_k_sensitivity.py
python scripts/03_parent_deduplication.py
python scripts/05_fusion_ranking_difference.py
python -m pytest -q
python -m ruff check .
```

`src/rrf_examples/rrf.py` holds the article's `reciprocal_rank_fusion()` function, which the
first three scripts import. `tests/test_scripts.py` runs every script and checks that its output
matches the article exactly.

## RRF vs DBSF with the Actian client

`scripts/05_fusion_ranking_difference.py` runs the installed Actian client's public functions over
the same six-document keyword and vector lists, defined in `src/rrf_examples/ranking_difference.py`:

- `actian_vectorai.hybrid.fusion.reciprocal_rank_fusion` uses ranks and ignores raw score scale.
- `actian_vectorai.hybrid.fusion.distribution_based_score_fusion` normalizes each source's raw
  scores independently using a clipped `mean ± 3 * population standard deviation` range, then
  averages across all sources.

Both functions take lists of `actian_vectorai.models.ScoredPoint`, expect larger scores to be
better, identify duplicates by point ID, and return `ScoredPoint` values whose `score` is the
fused score.

The exact-error guide is a large keyword-score outlier at rank 1 (42.0 vs 14.0) and sits at vector
rank 6. RRF ranks the pooling guide, timeout FAQ, and exact-error guide first through third
because it uses only their positions. DBSF ranks the pooling guide, exact-error guide, and timeout
FAQ first through third because it retains how unusual the exact guide's keyword score is. This
demonstrates different behavior, not that either method is universally better.
