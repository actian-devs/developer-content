# Benchmark results

2026-10-14T06:47:24+00:00 on Apple arm64 (MPS). 50 test queries per dataset; every reranker scores the top 30 hybrid candidates.

## nfcorpus

2400 documents indexed. Dataset sha256 `efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b`. First-stage recall, hybrid: R@30 0.1805, R@100 0.2682. Dense only: R@100 0.2746.

| First stage and reranker | nDCG@5 | nDCG@10 | Recall@10 |
|---|---|---|---|
| dense only | 0.3520 | 0.3185 | 0.1437 |
| hybrid (RRF) | 0.3654 | 0.3267 | 0.1443 |
| hybrid + jev | 0.4131 | 0.3662 | 0.1477 |
| hybrid + cross-encoder | 0.3295 | 0.3028 | 0.1378 |

| Reranker | ECE | Pairs | p50 ms | p95 ms | Cost per 1K queries (USD) |
|---|---|---|---|---|---|
| jev | 0.0604 | 1500 | 1273 | 1685 | 0.9138 |
| cross-encoder | 0.1008 | 1500 | 4741 | 8034 | n/a (own hardware) |

| Reranker | Strategy | nDCG@10 | Mean candidates reranked | Cost per 1K queries (USD) |
|---|---|---|---|---|
| jev | fixed top-30 | 0.3662 | 30.0 | 0.9138 |
| jev | adaptive, threshold 0.5 | 0.3655 | 24.8 | 0.7554 |
| jev | adaptive, threshold 0.7 | 0.3654 | 27.9 | 0.8498 |
| jev | adaptive, threshold 0.9 | 0.3662 | 30.0 | 0.9138 |
| cross-encoder | fixed top-30 | 0.3028 | 30.0 | n/a |
| cross-encoder | adaptive, threshold 0.5 | 0.3128 | 24.5 | n/a |
| cross-encoder | adaptive, threshold 0.7 | 0.3089 | 25.6 | n/a |
| cross-encoder | adaptive, threshold 0.9 | 0.3042 | 27.7 | n/a |

## scifact

2400 documents indexed. Dataset sha256 `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`. First-stage recall, hybrid: R@30 0.9000, R@100 0.9400. Dense only: R@100 0.9800.

| First stage and reranker | nDCG@5 | nDCG@10 | Recall@10 |
|---|---|---|---|
| dense only | 0.7116 | 0.7351 | 0.8600 |
| hybrid (RRF) | 0.6457 | 0.6818 | 0.8300 |
| hybrid + jev | 0.8064 | 0.8146 | 0.9000 |
| hybrid + cross-encoder | 0.7039 | 0.7141 | 0.8300 |

| Reranker | ECE | Pairs | p50 ms | p95 ms | Cost per 1K queries (USD) |
|---|---|---|---|---|---|
| jev | 0.0436 | 1500 | 1282 | 1667 | 0.9140 |
| cross-encoder | 0.0560 | 1500 | 5242 | 24301 | n/a (own hardware) |

| Reranker | Strategy | nDCG@10 | Mean candidates reranked | Cost per 1K queries (USD) |
|---|---|---|---|---|
| jev | fixed top-30 | 0.8146 | 30.0 | 0.9140 |
| jev | adaptive, threshold 0.5 | 0.8146 | 29.0 | 0.8835 |
| jev | adaptive, threshold 0.7 | 0.8146 | 29.5 | 0.8987 |
| jev | adaptive, threshold 0.9 | 0.8146 | 30.0 | 0.9140 |
| cross-encoder | fixed top-30 | 0.7141 | 30.0 | n/a |
| cross-encoder | adaptive, threshold 0.5 | 0.7144 | 28.2 | n/a |
| cross-encoder | adaptive, threshold 0.7 | 0.7141 | 29.4 | n/a |
| cross-encoder | adaptive, threshold 0.9 | 0.7141 | 30.0 | n/a |
