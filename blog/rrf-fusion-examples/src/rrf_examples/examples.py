"""Fixed illustrative inputs used by the first three scripts."""

from typing import Final

KEYWORD_RESULTS: Final = [
    {"id": "error-guide", "score": 18.7},
    {"id": "pooling-guide", "score": 12.4},
    {"id": "timeout-faq", "score": 8.2},
]

VECTOR_RESULTS: Final = [
    {"id": "pooling-guide", "score": 0.89},
    {"id": "timeout-faq", "score": 0.85},
    {"id": "error-guide", "score": 0.81},
]

K_KEYWORD_RANKING: Final = [
    "exact-error-guide",
    "keyword-only-a",
    "keyword-only-b",
    "general-pooling-guide",
]
K_VECTOR_RANKING: Final = [
    "vector-only-a",
    "vector-only-b",
    "vector-only-c",
    "general-pooling-guide",
]

CHUNK_KEYWORD_RANKING: Final = ["pool-1", "pool-2", "error-1", "faq-1"]
CHUNK_VECTOR_RANKING: Final = ["pool-2", "pool-1", "faq-1", "error-1"]
PARENT_IDS: Final = {
    "pool-1": "pool-guide",
    "pool-2": "pool-guide",
    "error-1": "error-guide",
    "faq-1": "timeout-faq",
}


def ids(results: list[dict[str, str | float]]) -> list[str]:
    """Extract IDs from score-bearing illustrative retrieval results."""
    return [str(result["id"]) for result in results]

