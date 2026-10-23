# Verification results

Verified locally on 2026-10-21. These are illustrative, deterministic client-side examples; no
live VectorAI DB connection was attempted.

## Environment

- Python 3.12.14
- actian-vectorai-client 1.0.3
- pytest 9.1.1
- Ruff 0.17.1
- actian-rrf-fusion-examples 0.1.0 (editable install)

Confirmed public signatures:

```text
reciprocal_rank_fusion(responses: 'Sequence[Sequence[ScoredPoint]]', limit: 'int' = 10, *, ranking_constant_k: 'int' = 60, weights: 'Sequence[float] | None' = None) -> 'list[ScoredPoint]'
distribution_based_score_fusion(responses: 'Sequence[Sequence[ScoredPoint]]', limit: 'int' = 10) -> 'list[ScoredPoint]'
```

## Commands and actual output

```text
> python -m pytest -q
.............                                                            [100%]
13 passed in 0.77s

> python -m ruff check .
All checks passed!

> python scripts/01_score_scale_example.py
pooling-guide: 0.03252
error-guide: 0.03227
timeout-faq: 0.03200

> python scripts/02_k_sensitivity.py
k=60: exact=0.01639, general=0.03125; leader=general-pooling-guide
k=10: exact=0.09091, general=0.14286; leader=general-pooling-guide
k= 2: exact=0.33333, general=0.33333; leader=tie
k= 1: exact=0.50000, general=0.40000; leader=exact-error-guide

> python scripts/03_parent_deduplication.py
Chunk-level fusion: pool-1, pool-2, error-1, faq-1
Group before fusion: pool-guide, error-guide, timeout-faq
Limit after fusion: pool-1, error-1, faq-1

> python scripts/04_rrf_vs_dbsf.py
Illustrative query: ConnectionTimeoutError: pool exhausted
Inputs are illustrative; no live VectorAI DB query was run.

RRF
pooling-guide: 0.03252
error-guide: 0.03227
timeout-faq: 0.03200

DBSF
pooling-guide: 0.58854
error-guide: 0.50609
timeout-faq: 0.40537
```

Both SDK fusion methods produce the same order for this small input. RRF derives scores only from
rank positions. DBSF independently normalizes the keyword and vector score distributions, then
averages normalized values across both sources; its fused scores therefore are not comparable to
RRF scores. Different source distributions can produce different rankings.

The initial combined dependency-install command ended after downloading several wheels and before
installing the Actian client and pytest. Re-running installation completed successfully. There
were no implementation, test, lint, or script failures in the recorded verification pass.

## Controlled ranking-difference experiment

Added and verified on 2026-10-22. The query and all scores remain illustrative. A small controlled
sensitivity check kept five candidates fixed and varied only the exact guide's keyword strength
and low-ranked vector score. Cases with vector scores from `0.70` through `0.78` retained the same
top-three order. At a plausible `0.79` vector score, the exact guide remains sixth in that source,
but DBSF moves it above the timeout FAQ while RRF does not. This is the recorded demonstration,
not a random search or a claim that DBSF is better.

Keyword input:

```text
1. exact-error-guide: 42.0
2. pooling-guide: 14.0
3. timeout-faq: 12.0
4. retry-policy: 10.5
5. connection-limits: 9.5
6. network-diagnostics: 8.5
```

Vector input:

```text
1. pooling-guide: 0.91
2. timeout-faq: 0.89
3. retry-policy: 0.87
4. connection-limits: 0.85
5. network-diagnostics: 0.83
6. exact-error-guide: 0.79
```

Actual SDK output:

```text
RRF (ranking_constant_k=60)
1. pooling-guide: 0.03252247
2. timeout-faq: 0.03200205
3. exact-error-guide: 0.03154496
4. retry-policy: 0.03149802
5. connection-limits: 0.03100962
6. network-diagnostics: 0.03053613

DBSF
1. pooling-guide: 0.59787901
2. exact-error-guide: 0.54335525
3. timeout-faq: 0.54140540
4. retry-policy: 0.48848577
5. connection-limits: 0.43912011
6. network-diagnostics: 0.38975445
```

RRF ignores the raw `42.0` versus `14.0` keyword gap: it sees only keyword ranks 1 and 2, then
combines them with vector ranks 6 and 1. DBSF normalizes the two score distributions separately,
so it retains the exact guide's unusually large keyword advantage as well as its weaker vector
position. That retained magnitude is enough to swap the exact guide and timeout FAQ. No independent
relevance assessment was performed, so the experiment supports no claim that either order is
better.

Verification after adding the experiment:

```text
python -m pytest -q
15 passed

python -m ruff check .
All checks passed!
```

All five scripts completed successfully. The fifth script reproduced the inputs and fused scores
shown above.

## Update: scripts aligned with the article

Updated on 2026-10-23. Scripts 01, 02, 03 and 05 now use the article's code snippets, and
`src/rrf_examples/rrf.py` uses the article's `reciprocal_rank_fusion()`. Rankings and scores are
unchanged. Output formatting changed in three places:

- `02_k_sensitivity.py` prints the article's k table, one row per k value.
- `03_parent_deduplication.py` prints the article's chunk and parent rankings with scores. The
  limit-after-fusion case is still covered by `limit_chunks_per_parent()` and its test.
- `05_fusion_ranking_difference.py` prints the fused rankings to five decimals, as in the article.

```text
python -m pytest -q
19 passed

python -m ruff check .
All checks passed!
```
