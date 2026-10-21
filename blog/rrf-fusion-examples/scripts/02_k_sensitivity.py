"""Example 2: changing k changes how strongly top ranks matter."""

from rrf_examples.examples import K_KEYWORD_RANKING, K_VECTOR_RANKING
from rrf_examples.rrf import reciprocal_rank_fusion

targets = ("exact-error-guide", "general-pooling-guide")
for k in (60, 10, 2, 1):
    results = reciprocal_rank_fusion([K_KEYWORD_RANKING, K_VECTOR_RANKING], k=k)
    scores = {result.id: result.score for result in results}
    relation = "tie" if scores[targets[0]] == scores[targets[1]] else " > ".join(
        target for target in targets if scores[target] == max(scores[name] for name in targets)
    )
    print(
        f"k={k:>2}: exact={scores[targets[0]]:.5f}, "
        f"general={scores[targets[1]]:.5f}; leader={relation}"
    )

