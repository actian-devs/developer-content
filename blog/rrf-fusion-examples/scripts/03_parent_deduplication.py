"""Section 3: duplicate chunks and inconsistent IDs affect RRF results."""

from rrf_examples.rrf import reciprocal_rank_fusion

keyword_results = [
    "pool-1", "pool-2", "error-1", "faq-1"
]

vector_results = [
    "pool-2", "pool-1", "faq-1", "error-1"
]

parent_ids = {
    "pool-1": "pool-guide",
    "pool-2": "pool-guide",
    "error-1": "error-guide",
    "faq-1": "timeout-faq",
}

def fuse(result_lists):
    return reciprocal_rank_fusion(
        [
            [{"id": doc_id} for doc_id in results]
            for results in result_lists
        ],
        k=60,
    )

def unique_parents(results):
    return list(dict.fromkeys(
        parent_ids[doc_id] for doc_id in results
    ))

chunk_ranking = fuse([keyword_results, vector_results])
parent_ranking = fuse([
    unique_parents(keyword_results),
    unique_parents(vector_results),
])

for label, ranking in [
    ("Chunks", chunk_ranking),
    ("Parents", parent_ranking),
]:
    print(label)
    for doc_id, score in ranking:
        print(f"  {doc_id}: {score:.5f}")
