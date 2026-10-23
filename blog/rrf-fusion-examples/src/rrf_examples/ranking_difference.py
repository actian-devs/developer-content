"""Run the Actian client's RRF and DBSF over the same six-document result lists."""

from dataclasses import dataclass

from actian_vectorai.hybrid.fusion import (
    distribution_based_score_fusion,
    reciprocal_rank_fusion,
)
from actian_vectorai.models import ScoredPoint

# Exact-error guide: keyword score 42.0 (vs 14.0), vector rank 6.
keyword_results = [
    {"id": "exact-error-guide", "score": 42.0},
    {"id": "pooling-guide", "score": 14.0},
    {"id": "timeout-faq", "score": 12.0},
    {"id": "retry-policy", "score": 10.5},
    {"id": "connection-limits", "score": 9.5},
    {"id": "network-diagnostics", "score": 8.5},
]

vector_results = [
    {"id": "pooling-guide", "score": 0.91},
    {"id": "timeout-faq", "score": 0.89},
    {"id": "retry-policy", "score": 0.87},
    {"id": "connection-limits", "score": 0.85},
    {"id": "network-diagnostics", "score": 0.83},
    {"id": "exact-error-guide", "score": 0.79},
]


@dataclass(frozen=True)
class Comparison:
    method: str
    ranking: list[ScoredPoint]


def to_scored_points(results):
    return [ScoredPoint(id=result["id"], score=result["score"]) for result in results]


def compare_ranking_difference():
    responses = [to_scored_points(keyword_results), to_scored_points(vector_results)]
    limit = len(keyword_results)

    return [
        Comparison("RRF", reciprocal_rank_fusion(responses, limit=limit, ranking_constant_k=60)),
        Comparison("DBSF", distribution_based_score_fusion(responses, limit=limit)),
    ]
