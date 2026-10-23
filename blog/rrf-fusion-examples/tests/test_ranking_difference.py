import inspect

import pytest
from actian_vectorai.hybrid.fusion import (
    distribution_based_score_fusion,
    reciprocal_rank_fusion,
)

from rrf_examples.ranking_difference import compare_ranking_difference


def test_rrf_and_dbsf_rank_differently() -> None:
    rrf, dbsf = compare_ranking_difference()

    assert [point.id for point in rrf.ranking][:3] == [
        "pooling-guide",
        "timeout-faq",
        "exact-error-guide",
    ]
    assert [point.id for point in dbsf.ranking][:3] == [
        "pooling-guide",
        "exact-error-guide",
        "timeout-faq",
    ]
    assert [point.score for point in rrf.ranking][:3] == pytest.approx(
        [1 / 62 + 1 / 61, 1 / 63 + 1 / 62, 1 / 61 + 1 / 66]
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
