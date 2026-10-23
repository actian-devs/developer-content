"""Section 1: the scores don't share a scale, so you fuse ranks."""

from rrf_examples.rrf import reciprocal_rank_fusion

keyword_results = [
    {"id": "error-guide", "score": 18.7},
    {"id": "pooling-guide", "score": 12.4},
    {"id": "timeout-faq", "score": 8.2},
]

vector_results = [
    {"id": "pooling-guide", "score": 0.89},
    {"id": "timeout-faq", "score": 0.85},
    {"id": "error-guide", "score": 0.81},
]

ranked_results = reciprocal_rank_fusion(
    [keyword_results, vector_results]
)

for doc_id, score in ranked_results:
    print(f"{doc_id}: {score:.5f}")
