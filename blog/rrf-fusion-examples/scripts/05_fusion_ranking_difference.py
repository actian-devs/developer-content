"""Section 4: RRF and DBSF rank the same results differently."""

from rrf_examples.ranking_difference import (
    compare_ranking_difference,
)

for comparison in compare_ranking_difference():
    print(comparison.method)
    for point in comparison.ranking:
        print(f"{point.id}: {point.score:.5f}")
