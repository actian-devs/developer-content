"""Example 4: run the official Actian SDK's RRF and DBSF utilities."""

from rrf_examples.comparison import QUERY, compare_fusions

print(f"Illustrative query: {QUERY}")
print("Inputs are illustrative; no live VectorAI DB query was run.")
for comparison in compare_fusions():
    print(f"\n{comparison.method}")
    for point in comparison.ranking:
        print(f"{point.id}: {point.score:.5f}")

