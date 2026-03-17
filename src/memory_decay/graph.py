"""Graph-based memory store with embedding similarity retrieval."""

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Optional


class MemoryGraph:
    """NetworkX DiGraph-based memory store.

    Nodes represent memory items with activation scores.
    Edges represent associations between memories.
    Embeddings enable semantic similarity queries.
    """

    def __init__(self, embedder=None):
        self._graph = nx.DiGraph()
        self._embedder = embedder  # callable(text) -> np.ndarray
        self._model = None

    def _get_embedder(self):
        """Lazy-load sentence-transformers on first use if no custom embedder."""
        if self._embedder is not None:
            return self._embedder
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model.encode

    def add_memory(
        self,
        memory_id: str,
        mtype: str,
        content: str,
        impact: float,
        created_tick: int,
        associations: list[tuple[str, float]] | None = None,
    ) -> None:
        """Insert a memory node with associations.

        Args:
            memory_id: Unique identifier.
            mtype: "fact" or "episode".
            content: Memory text content.
            impact: Emotional significance (0.1-1.0).
            created_tick: Time step when memory was created.
            associations: List of (target_id, weight) tuples.
        """
        embedder = self._get_embedder()
        embedding = np.array(embedder(content), dtype=np.float32)

        self._graph.add_node(
            memory_id,
            type=mtype,
            content=content,
            embedding=embedding,
            activation_score=1.0,
            impact=impact,
            created_tick=created_tick,
            last_activated_tick=created_tick,
        )

        if associations:
            for target_id, weight in associations:
                # Ensure target node exists (may be added later — still create edge)
                if not self._graph.has_node(target_id):
                    self._graph.add_node(
                        target_id,
                        type="unknown",
                        content="",
                        embedding=np.zeros(384, dtype=np.float32),
                        activation_score=0.0,
                        impact=0.0,
                        created_tick=0,
                        last_activated_tick=0,
                    )
                self._graph.add_edge(
                    memory_id, target_id, weight=weight, created_tick=created_tick
                )
                # Bidirectional association
                if not self._graph.has_edge(target_id, memory_id):
                    self._graph.add_edge(
                        target_id, memory_id, weight=weight, created_tick=created_tick
                    )

    def query_by_similarity(
        self, query_text: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Find memories matching query via embedding cosine similarity.

        Returns list of (node_id, similarity_score) sorted descending.
        Only returns nodes with actual content (type != "unknown").
        """
        embedder = self._get_embedder()
        query_vec = np.array(embedder(query_text), dtype=np.float32)

        results = []
        for nid, attrs in self._graph.nodes(data=True):
            if attrs.get("type") == "unknown":
                continue
            emb = attrs["embedding"]
            norm_q = np.linalg.norm(query_vec)
            norm_e = np.linalg.norm(emb)
            if norm_q == 0 or norm_e == 0:
                continue
            sim = float(np.dot(query_vec, emb) / (norm_q * norm_e))
            results.append((nid, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_associated(self, node_id: str) -> list[tuple[str, float]]:
        """Get neighbors connected by association edges.

        Returns list of (neighbor_id, edge_weight).
        """
        results = []
        for neighbor in self._graph.predecessors(node_id):
            w = self._graph.edges[neighbor, node_id].get("weight", 0.0)
            results.append((neighbor, w))
        for neighbor in self._graph.successors(node_id):
            w = self._graph.edges[node_id, neighbor].get("weight", 0.0)
            results.append((neighbor, w))
        return results

    def re_activate(self, node_id: str, boost_amount: float) -> None:
        """Boost activation of a node and cascade to associated nodes (one hop).

        Cascade boost = boost_amount * edge_weight * 0.5
        """
        if not self._graph.has_node(node_id):
            return

        current = self._graph.nodes[node_id]["activation_score"]
        self._graph.nodes[node_id]["activation_score"] = min(current + boost_amount, 2.0)

        # One-hop cascade
        for neighbor, weight in self.get_associated(node_id):
            cascade_boost = boost_amount * weight * 0.5
            n_score = self._graph.nodes[neighbor]["activation_score"]
            self._graph.nodes[neighbor]["activation_score"] = min(
                n_score + cascade_boost, 2.0
            )

    def get_all_activations(self) -> dict[str, float]:
        """Return dict of node_id -> activation_score for all nodes."""
        return {
            nid: attrs["activation_score"]
            for nid, attrs in self._graph.nodes(data=True)
            if attrs.get("type") != "unknown"
        }

    def set_activation(self, node_id: str, score: float) -> None:
        """Directly set a node's activation score."""
        if self._graph.has_node(node_id):
            self._graph.nodes[node_id]["activation_score"] = score

    def get_node(self, node_id: str) -> dict | None:
        """Return all attributes for a node."""
        if self._graph.has_node(node_id):
            return dict(self._graph.nodes[node_id])
        return None

    @property
    def num_memories(self) -> int:
        return sum(
            1 for _, attrs in self._graph.nodes(data=True)
            if attrs.get("type") != "unknown"
        )
