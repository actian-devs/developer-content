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
