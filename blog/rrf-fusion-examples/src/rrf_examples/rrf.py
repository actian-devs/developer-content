"""The Reciprocal Rank Fusion function from the article, unchanged."""


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
