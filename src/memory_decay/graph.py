"""MemoryGraph: graph-based memory store.

Manages a NetworkX DiGraph where nodes represent memory items (facts or episodes)
with activation scores, embeddings, and impact values, and edges represent
associations between related memories with configurable weights.

Uses sentence-transformers (all-MiniLM-L6-v2, 384-dim) for embedding generation
and similarity-based recall queries.

Key methods (to implement):
- add_memory(): insert a memory node with its attributes and edges
- query_by_similarity(): find memories matching a query via embedding cosine similarity
- get_associated(): retrieve memories connected by association edges
- re_activate(): boost activation of a memory and cascade to associated nodes
- prune_below_threshold(): remove memories whose activation has fallen below a cutoff
"""
