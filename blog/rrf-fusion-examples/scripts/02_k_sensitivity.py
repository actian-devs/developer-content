"""Section 2: the default k=60 can bury a strong match."""

from rrf_examples.rrf import reciprocal_rank_fusion

# Exact guide: keyword rank 1, not returned by vector search.
# General guide: rank 4 in both lists.
keyword_results = [
    {"id": "exact-error-guide"},
    {"id": "keyword-only-a"},
    {"id": "keyword-only-b"},
    {"id": "general-pooling-guide"},
]

vector_results = [
    {"id": "vector-only-a"},
    {"id": "vector-only-b"},
    {"id": "vector-only-c"},
    {"id": "general-pooling-guide"},
]

for k in [60, 10, 2, 1]:
    scores = dict(reciprocal_rank_fusion([keyword_results, vector_results], k=k))
    exact = scores["exact-error-guide"]
    general = scores["general-pooling-guide"]

    if round(exact, 5) == round(general, 5):
        leader = "Tie"
    elif exact > general:
        leader = "Exact error guide"
    else:
        leader = "General guide"

    print(f"k={k:<2}  exact: {exact:.5f}  general: {general:.5f}  higher: {leader}")
