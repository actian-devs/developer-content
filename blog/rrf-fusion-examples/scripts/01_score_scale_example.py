"""Example 1: RRF ignores incompatible raw score scales."""

from rrf_examples.examples import KEYWORD_RESULTS, VECTOR_RESULTS, ids
from rrf_examples.rrf import reciprocal_rank_fusion

results = reciprocal_rank_fusion([ids(KEYWORD_RESULTS), ids(VECTOR_RESULTS)], k=60)
for result in results:
    print(f"{result.id}: {result.score:.5f}")

