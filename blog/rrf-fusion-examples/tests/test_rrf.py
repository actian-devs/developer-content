import pytest

from rrf_examples.examples import KEYWORD_RESULTS, VECTOR_RESULTS, ids
from rrf_examples.rrf import reciprocal_rank_fusion


def test_article_score_scale_example() -> None:
    results = reciprocal_rank_fusion([ids(KEYWORD_RESULTS), ids(VECTOR_RESULTS)], k=60)

    assert [item.id for item in results] == ["pooling-guide", "error-guide", "timeout-faq"]
    assert [item.score for item in results] == pytest.approx(
        [1 / 62 + 1 / 61, 1 / 61 + 1 / 63, 1 / 63 + 1 / 62]
    )
    assert [round(item.score, 5) for item in results] == [0.03252, 0.03227, 0.03200]


def test_missing_document_contributes_zero() -> None:
    scores = {item.id: item.score for item in reciprocal_rank_fusion([["only-here"], []], k=60)}
    assert scores["only-here"] == pytest.approx(1 / 61)


def test_ties_use_stable_first_seen_order() -> None:
    results = reciprocal_rank_fusion([["first", "second"], ["second", "first"]])
    assert [item.id for item in results] == ["first", "second"]


def test_invalid_k() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        reciprocal_rank_fusion([["doc"]], k=0)

