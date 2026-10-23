"""Use the official Actian client to compare RRF with DBSF."""

from collections.abc import Sequence
from dataclasses import dataclass

from actian_vectorai.hybrid.fusion import (
    distribution_based_score_fusion,
    reciprocal_rank_fusion,
)
from actian_vectorai.models import ScoredPoint

from rrf_examples.examples import KEYWORD_RESULTS, VECTOR_RESULTS

QUERY = "ConnectionTimeoutError: pool exhausted"


@dataclass(frozen=True)
class ComparisonResult:
    method: str
    ranking: tuple[ScoredPoint, ...]


def to_scored_points(results: Sequence[dict[str, str | float]]) -> list[ScoredPoint]:
    """Convert the fixed illustrative inputs to the SDK's required model."""
    return [
        ScoredPoint(
            id=str(result["id"]),
            score=float(result["score"]),
            payload={"title": str(result["id"])},
        )
        for result in results
    ]


def comparison_inputs() -> tuple[list[ScoredPoint], list[ScoredPoint]]:
    """Return fresh keyword and vector lists containing identical candidates."""
    return to_scored_points(KEYWORD_RESULTS), to_scored_points(VECTOR_RESULTS)


def compare_fusions() -> tuple[ComparisonResult, ComparisonResult]:
    """Run both official SDK fusion functions over the same two source lists."""
    keyword, vector = comparison_inputs()
    responses = [keyword, vector]
    limit = len({point.id for response in responses for point in response})
    rrf = reciprocal_rank_fusion(responses, limit=limit, ranking_constant_k=60)
    dbsf = distribution_based_score_fusion(responses, limit=limit)
    return ComparisonResult("RRF", tuple(rrf)), ComparisonResult("DBSF", tuple(dbsf))

