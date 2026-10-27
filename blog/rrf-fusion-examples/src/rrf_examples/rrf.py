"""Small, transparent RRF implementation used by the article examples."""

from collections.abc import Hashable, Mapping, Sequence


def reciprocal_rank_fusion(result_lists, k=60):
    fused_scores = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            doc_id = result["id"]
            contribution = 1 / (k + rank)

            fused_scores[doc_id] = (
                fused_scores.get(doc_id, 0.0) + contribution
            )

    return sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )


def group_ranking_by_parent(
    ranking: Sequence[Hashable], parent_ids: Mapping[Hashable, Hashable]
) -> list[Hashable]:
    """Map chunks to parents and retain only the first occurrence per ranking."""
    grouped: list[Hashable] = []
    seen: set[Hashable] = set()
    for chunk_id in ranking:
        parent_id = parent_ids[chunk_id]
        if parent_id not in seen:
            seen.add(parent_id)
            grouped.append(parent_id)
    return grouped


def fuse_by_parent(
    rankings: Sequence[Sequence[Hashable]],
    parent_ids: Mapping[Hashable, Hashable],
    k: int = 60,
) -> list[tuple[Hashable, float]]:
    """Group chunks within each source ranking before applying RRF."""
    return reciprocal_rank_fusion(
        [
            [{"id": parent_id} for parent_id in group_ranking_by_parent(ranking, parent_ids)]
            for ranking in rankings
        ],
        k=k,
    )


def limit_chunks_per_parent(
    results: Sequence[tuple[Hashable, float]],
    parent_ids: Mapping[Hashable, Hashable],
    limit: int = 1,
) -> list[tuple[Hashable, float]]:
    """Limit duplicate passages after chunk-level fusion without recomputing scores."""
    if limit < 1:
        raise ValueError("limit must be at least 1")

    counts: dict[Hashable, int] = {}
    limited: list[tuple[Hashable, float]] = []
    for doc_id, score in results:
        parent_id = parent_ids[doc_id]
        count = counts.get(parent_id, 0)
        if count < limit:
            limited.append((doc_id, score))
            counts[parent_id] = count + 1
    return limited
