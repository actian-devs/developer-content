"""Controlled SDK experiment in which RRF and DBSF rank candidates differently."""

from typing import Final

from actian_vectorai.hybrid.fusion import (
    distribution_based_score_fusion,
    reciprocal_rank_fusion,
)
from actian_vectorai.models import ScoredPoint

from rrf_examples.comparison import ComparisonResult, to_scored_points

DIFFERENCE_KEYWORD_RESULTS: Final = [
    {"id": "exact-error-guide", "score": 42.0},
    {"id": "pooling-guide", "score": 14.0},
    {"id": "timeout-faq", "score": 12.0},
    {"id": "retry-policy", "score": 10.5},
    {"id": "connection-limits", "score": 9.5},
    {"id": "network-diagnostics", "score": 8.5},
]

DIFFERENCE_VECTOR_RESULTS: Final = [
    {"id": "pooling-guide", "score": 0.91},
    {"id": "timeout-faq", "score": 0.89},
    {"id": "retry-policy", "score": 0.87},
    {"id": "connection-limits", "score": 0.85},
    {"id": "network-diagnostics", "score": 0.83},
    {"id": "exact-error-guide", "score": 0.79},
]


def ranking_difference_inputs() -> tuple[list[ScoredPoint], list[ScoredPoint]]:
    """Return fresh SDK inputs for the controlled ranking-difference experiment."""
    return (
        to_scored_points(DIFFERENCE_KEYWORD_RESULTS),
        to_scored_points(DIFFERENCE_VECTOR_RESULTS),
    )


def compare_ranking_difference() -> tuple[ComparisonResult, ComparisonResult]:
    """Run both official SDK fusion methods over the same source lists."""
    keyword, vector = ranking_difference_inputs()
    responses = [keyword, vector]
    limit = len({point.id for response in responses for point in response})
    rrf = reciprocal_rank_fusion(responses, limit=limit, ranking_constant_k=60)
    dbsf = distribution_based_score_fusion(responses, limit=limit)
    return ComparisonResult("RRF", tuple(rrf)), ComparisonResult("DBSF", tuple(dbsf))

