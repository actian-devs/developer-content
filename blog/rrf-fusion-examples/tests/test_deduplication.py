from rrf_examples.examples import CHUNK_KEYWORD_RANKING, CHUNK_VECTOR_RANKING, PARENT_IDS
from rrf_examples.rrf import fuse_by_parent, limit_chunks_per_parent, reciprocal_rank_fusion


def as_results(ranking: list[str]) -> list[dict[str, str]]:
    return [{"id": doc_id} for doc_id in ranking]


def test_chunk_level_and_parent_level_orders() -> None:
    rankings = [CHUNK_KEYWORD_RANKING, CHUNK_VECTOR_RANKING]
    chunks = reciprocal_rank_fusion([as_results(ranking) for ranking in rankings])
    parents = fuse_by_parent(rankings, PARENT_IDS)

    assert [doc_id for doc_id, _ in chunks] == ["pool-1", "pool-2", "error-1", "faq-1"]
    assert [doc_id for doc_id, _ in parents] == ["pool-guide", "error-guide", "timeout-faq"]


def test_limiting_after_fusion_does_not_relabel_or_recompute() -> None:
    chunks = reciprocal_rank_fusion(
        [as_results(CHUNK_KEYWORD_RANKING), as_results(CHUNK_VECTOR_RANKING)]
    )
    limited = limit_chunks_per_parent(chunks, PARENT_IDS)

    assert [doc_id for doc_id, _ in limited] == ["pool-1", "error-1", "faq-1"]
    assert limited[0][1] == chunks[0][1]
