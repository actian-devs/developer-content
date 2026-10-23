import pytest
from actian_vectorai.models import ScoredPoint

from rrf_examples.ranking_difference import (
    DIFFERENCE_KEYWORD_RESULTS,
    DIFFERENCE_VECTOR_RESULTS,
    compare_ranking_difference,
    ranking_difference_inputs,
)


def test_ranking_difference_inputs_preserve_candidates_order_and_scores() -> None:
    keyword, vector = ranking_difference_inputs()

    assert all(isinstance(point, ScoredPoint) for point in keyword + vector)
    assert [point.id for point in keyword] == [
        result["id"] for result in DIFFERENCE_KEYWORD_RESULTS
    ]
    assert [point.id for point in vector] == [result["id"] for result in DIFFERENCE_VECTOR_RESULTS]
    assert {point.id for point in keyword} == {point.id for point in vector}
    assert [point.score for point in keyword] == [42.0, 14.0, 12.0, 10.5, 9.5, 8.5]
    assert [point.score for point in vector] == [0.91, 0.89, 0.87, 0.85, 0.83, 0.79]


def test_official_sdk_methods_produce_expected_different_rankings() -> None:
    rrf, dbsf = compare_ranking_difference()

    assert [point.id for point in rrf.ranking] == [
        "pooling-guide",
        "timeout-faq",
        "exact-error-guide",
        "retry-policy",
        "connection-limits",
        "network-diagnostics",
    ]
    assert [point.id for point in dbsf.ranking] == [
        "pooling-guide",
        "exact-error-guide",
        "timeout-faq",
        "retry-policy",
        "connection-limits",
        "network-diagnostics",
    ]
    assert [point.score for point in rrf.ranking] == pytest.approx(
        [
            1 / 62 + 1 / 61,
            1 / 63 + 1 / 62,
            1 / 61 + 1 / 66,
            1 / 64 + 1 / 63,
            1 / 65 + 1 / 64,
            1 / 66 + 1 / 65,
        ]
    )
    assert [point.score for point in dbsf.ranking] == pytest.approx(
        [
            0.5978790108147307,
            0.5433552535197013,
            0.5414054038551958,
            0.4884857704514382,
            0.43912011060345785,
            0.38975445075547743,
        ]
    )

