"""Example 3: compare chunk fusion, parent grouping, and result limiting."""

from rrf_examples.examples import CHUNK_KEYWORD_RANKING, CHUNK_VECTOR_RANKING, PARENT_IDS
from rrf_examples.rrf import fuse_by_parent, limit_chunks_per_parent, reciprocal_rank_fusion

rankings = [CHUNK_KEYWORD_RANKING, CHUNK_VECTOR_RANKING]
chunks = reciprocal_rank_fusion(rankings)
parents = fuse_by_parent(rankings, PARENT_IDS)
limited_chunks = limit_chunks_per_parent(chunks, PARENT_IDS)

print("Chunk-level fusion:", ", ".join(str(item.id) for item in chunks))
print("Group before fusion:", ", ".join(str(item.id) for item in parents))
print("Limit after fusion:", ", ".join(str(item.id) for item in limited_chunks))

