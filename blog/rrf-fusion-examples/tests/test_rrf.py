import pytest

from rrf_examples.rrf import reciprocal_rank_fusion


def test_article_score_scale_example() -> None:
    keyword = [{"id": "error-guide"}, {"id": "pooling-guide"}, {"id": "timeout-faq"}]
    vector = [{"id": "pooling-guide"}, {"id": "timeout-faq"}, {"id": "error-guide"}]

    results = reciprocal_rank_fusion([keyword, vector])

    assert [doc_id for doc_id, _ in results] == ["pooling-guide", "error-guide", "timeout-faq"]
    assert [score for _, score in results] == pytest.approx(
        [1 / 62 + 1 / 61, 1 / 61 + 1 / 63, 1 / 63 + 1 / 62]
    )


def test_raw_scores_are_ignored() -> None:
    low = [{"id": "a", "score": 0.1}, {"id": "b", "score": 0.0}]
    high = [{"id": "a", "score": 999.0}, {"id": "b", "score": 1.0}]

    assert reciprocal_rank_fusion([low]) == reciprocal_rank_fusion([high])


def test_missing_document_contributes_zero() -> None:
    scores = dict(reciprocal_rank_fusion([[{"id": "only-here"}], []]))
    assert scores["only-here"] == pytest.approx(1 / 61)


def test_ties_use_stable_first_seen_order() -> None:
    results = reciprocal_rank_fusion(
        [[{"id": "first"}, {"id": "second"}], [{"id": "second"}, {"id": "first"}]]
    )
    assert [doc_id for doc_id, _ in results] == ["first", "second"]
