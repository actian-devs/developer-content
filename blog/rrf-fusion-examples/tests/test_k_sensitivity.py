import pytest

from rrf_examples.examples import K_KEYWORD_RANKING, K_VECTOR_RANKING
from rrf_examples.rrf import reciprocal_rank_fusion


@pytest.mark.parametrize(
    ("k", "exact", "general", "relative_order"),
    [
        (60, 0.01639, 0.03125, ["general-pooling-guide", "exact-error-guide"]),
        (10, 0.09091, 0.14286, ["general-pooling-guide", "exact-error-guide"]),
        (2, 0.33333, 0.33333, ["exact-error-guide", "general-pooling-guide"]),
        (1, 0.50000, 0.40000, ["exact-error-guide", "general-pooling-guide"]),
    ],
)
def test_k_values(k: int, exact: float, general: float, relative_order: list[str]) -> None:
    results = reciprocal_rank_fusion(
        [
            [{"id": doc_id} for doc_id in ranking]
            for ranking in (K_KEYWORD_RANKING, K_VECTOR_RANKING)
        ],
        k=k,
    )
    scores = dict(results)
    target_order = [doc_id for doc_id, _ in results if doc_id in relative_order]

    assert scores["exact-error-guide"] == pytest.approx(exact, abs=0.000005)
    assert scores["general-pooling-guide"] == pytest.approx(general, abs=0.000005)
    assert target_order == relative_order
    if k == 2:
        assert scores["exact-error-guide"] == scores["general-pooling-guide"]

