import inspect

import pytest
from actian_vectorai.hybrid.fusion import (
    distribution_based_score_fusion,
    reciprocal_rank_fusion,
)
from actian_vectorai.models import ScoredPoint

from rrf_examples.comparison import compare_fusions, comparison_inputs


def test_comparison_uses_same_candidates_and_original_scores() -> None:
    keyword, vector = comparison_inputs()

    assert all(isinstance(point, ScoredPoint) for point in keyword + vector)
    assert {point.id for point in keyword} == {point.id for point in vector}
    assert [point.score for point in keyword] == [18.7, 12.4, 8.2]
    assert [point.score for point in vector] == [0.89, 0.85, 0.81]


def test_official_sdk_comparison_results() -> None:
    rrf, dbsf = compare_fusions()

    assert [point.id for point in rrf.ranking] == [
        "pooling-guide",
        "error-guide",
        "timeout-faq",
    ]
    assert [point.id for point in dbsf.ranking] == [
        "pooling-guide",
        "error-guide",
        "timeout-faq",
    ]
    assert [point.score for point in rrf.ranking] == pytest.approx(
        [1 / 62 + 1 / 61, 1 / 61 + 1 / 63, 1 / 63 + 1 / 62]
    )


def test_installed_sdk_signatures() -> None:
    assert str(inspect.signature(reciprocal_rank_fusion)) == (
        "(responses: 'Sequence[Sequence[ScoredPoint]]', limit: 'int' = 10, *, "
        "ranking_constant_k: 'int' = 60, weights: 'Sequence[float] | None' = None) "
        "-> 'list[ScoredPoint]'"
    )
    assert str(inspect.signature(distribution_based_score_fusion)) == (
        "(responses: 'Sequence[Sequence[ScoredPoint]]', limit: 'int' = 10) "
        "-> 'list[ScoredPoint]'"
    )

