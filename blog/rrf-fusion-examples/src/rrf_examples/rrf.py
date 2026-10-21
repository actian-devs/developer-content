"""Small, transparent RRF implementation used by the article examples."""

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RankedResult:
    """A document ID and its fused score."""

    id: Hashable
    score: float


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]], k: int = 60
) -> list[RankedResult]:
    """Fuse ranked document-ID lists using ``sum(1 / (k + rank))``.

    Ranks begin at one. A document absent from a ranking contributes nothing.
    Ties retain the order in which IDs are first encountered because Python's
    sort is stable and dictionaries preserve insertion order.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (k + rank)

    return [
        RankedResult(document_id, score)
        for document_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


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
) -> list[RankedResult]:
    """Group chunks within each source ranking before applying RRF."""
    return reciprocal_rank_fusion(
        [group_ranking_by_parent(ranking, parent_ids) for ranking in rankings], k=k
    )


def limit_chunks_per_parent(
    results: Sequence[RankedResult],
    parent_ids: Mapping[Hashable, Hashable],
    limit: int = 1,
) -> list[RankedResult]:
    """Limit duplicate passages after chunk-level fusion without recomputing scores."""
    if limit < 1:
        raise ValueError("limit must be at least 1")

    counts: dict[Hashable, int] = {}
    limited: list[RankedResult] = []
    for result in results:
        parent_id = parent_ids[result.id]
        count = counts.get(parent_id, 0)
        if count < limit:
            limited.append(result)
            counts[parent_id] = count + 1
    return limited

