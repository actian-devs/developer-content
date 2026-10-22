"""Example 5: a controlled input for which the SDK fusion rankings differ."""

from rrf_examples.comparison import QUERY
from rrf_examples.ranking_difference import (
    DIFFERENCE_KEYWORD_RESULTS,
    DIFFERENCE_VECTOR_RESULTS,
    compare_ranking_difference,
)

print(f"Illustrative query: {QUERY}")
print("Inputs are illustrative; no live VectorAI DB query was run.")

for source, results in (
    ("Keyword", DIFFERENCE_KEYWORD_RESULTS),
    ("Vector", DIFFERENCE_VECTOR_RESULTS),
):
    print(f"\n{source} input")
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['id']}: {result['score']}")

for comparison in compare_ranking_difference():
    print(f"\n{comparison.method}")
    for point in comparison.ranking:
        print(f"{point.id}: {point.score:.8f}")

