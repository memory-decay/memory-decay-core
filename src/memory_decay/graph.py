"""Graph-based memory store with embedding similarity retrieval."""

from __future__ import annotations

import os
from typing import Optional, Callable

import networkx as nx
import numpy as np


class MemoryGraph:
    """NetworkX DiGraph-based memory store.

    Nodes represent memory items with activation scores.
    Edges represent associations between memories.
    Embeddings enable semantic similarity queries.
    """

    def __init__(self, embedder: Optional[Callable] = None, embedding_backend: str = "auto"):
        self._graph = nx.DiGraph()
        self._custom_embedder = embedder
        self._model = None
        self._embedding_backend = embedding_backend
        self._gemini_client = None
        self._embedding_dim = 768  # default for ko-sroberta
        self._embedding_cache: dict[tuple[str, str], np.ndarray] = {}

    def _get_embedder(self) -> Callable:
        """Get the embedding function based on backend."""
        if self._custom_embedder is not None:
            return self._custom_embedder

        backend = self._embedding_backend
        if backend == "auto":
            backend = "gemini" if os.environ.get("GEMINI_API_KEY") else "local"

        if backend == "gemini":
            return self._gemini_embed
        return self._local_embed

    def _embed_text(self, text: str) -> np.ndarray:
        """Embed text with a simple in-memory cache to reduce API churn."""
        backend = self._embedding_backend
        cache_key = (backend, text)
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        embedder = self._get_embedder()
        embedding = np.array(embedder(text), dtype=np.float32)
        self._embedding_dim = embedding.shape[0]
        self._embedding_cache[cache_key] = embedding
        return embedding.copy()

    def _local_embed(self, text: str) -> np.ndarray:
        """Local sentence-transformers embedding (Korean-optimized)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("jhgan/ko-sroberta-multitask")
            self._embedding_dim = 768
        return np.array(self._model.encode(text), dtype=np.float32)

    def _gemini_embed(self, text: str) -> np.ndarray:
        """Gemini API embedding via google-genai SDK."""
        if self._gemini_client is None:
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is required for Gemini embeddings.")
            self._gemini_client = genai.Client(api_key=api_key)
            self._embedding_dim = 768

        result = self._gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
        )
        return np.array(result.embeddings[0].values, dtype=np.float32)

    def add_memory(
        self,
        memory_id: str,
        mtype: str,
        content: str,
        impact: float,
        created_tick: int,
        associations: list[tuple[str, float]] | None = None,
    ) -> None:
        """Insert a memory node with associations."""
        embedding = self._embed_text(content)

        self._graph.add_node(
            memory_id,
            type=mtype,
            content=content,
            embedding=embedding,
            activation_score=1.0,
            stability_score=0.0,
            impact=impact,
            created_tick=created_tick,
            last_activated_tick=created_tick,
            retrieval_count=0,
            last_reinforced_tick=created_tick,
        )

        if associations:
            for target_id, weight in associations:
                if not self._graph.has_node(target_id):
                    self._graph.add_node(
                        target_id,
                        type="unknown",
                        content="",
                        embedding=np.zeros(self._embedding_dim, dtype=np.float32),
                        activation_score=0.0,
                        stability_score=0.0,
                        impact=0.0,
                        created_tick=0,
                        last_activated_tick=0,
                        retrieval_count=0,
                        last_reinforced_tick=0,
                    )
                self._graph.add_edge(
                    memory_id, target_id, weight=weight, created_tick=created_tick
                )
                if not self._graph.has_edge(target_id, memory_id):
                    self._graph.add_edge(
                        target_id, memory_id, weight=weight, created_tick=created_tick
                    )

    def query_by_similarity(
        self, query_text: str, top_k: int = 5, current_tick: int | None = None
    ) -> list[tuple[str, float]]:
        """Find memories matching query via embedding cosine similarity."""
        query_vec = self._embed_text(query_text)

        results = []
        for nid, attrs in self._graph.nodes(data=True):
            if attrs.get("type") == "unknown":
                continue
            if current_tick is not None and attrs.get("created_tick", 0) > current_tick:
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
        """Get neighbors connected by association edges."""
        results = []
        for neighbor in self._graph.predecessors(node_id):
            w = self._graph.edges[neighbor, node_id].get("weight", 0.0)
            results.append((neighbor, w))
        for neighbor in self._graph.successors(node_id):
            w = self._graph.edges[node_id, neighbor].get("weight", 0.0)
            results.append((neighbor, w))
        return results

    def _effective_reinforce_tick(
        self, node_id: str, current_tick: Optional[int]
    ) -> int:
        attrs = self._graph.nodes[node_id]
        if current_tick is not None:
            return current_tick
        base_tick = attrs.get("last_reinforced_tick", attrs.get("created_tick", 0))
        return int(base_tick) + 1

    def _apply_reactivation(
        self,
        node_id: str,
        boost_amount: float,
        *,
        source: str,
        reinforce: bool,
        current_tick: Optional[int],
        direct_gain: float,
        assoc_gain: float,
        stability_cap: float,
    ) -> None:
        attrs = self._graph.nodes[node_id]
        attrs["activation_score"] = min(
            max(attrs["activation_score"] + boost_amount, 0.0), 1.0
        )
        attrs["last_activated_tick"] = self._effective_reinforce_tick(node_id, current_tick)

        if not reinforce or attrs.get("type") == "unknown":
            return

        gain = direct_gain if source == "direct" else assoc_gain
        current_stability = float(attrs.get("stability_score", 0.0))
        saturation = max(1.0 - current_stability / max(stability_cap, 1e-9), 0.0)
        attrs["stability_score"] = min(
            max(current_stability + gain * saturation, 0.0), stability_cap
        )
        attrs["last_reinforced_tick"] = self._effective_reinforce_tick(
            node_id, current_tick
        )
        if source == "direct":
            attrs["retrieval_count"] = int(attrs.get("retrieval_count", 0)) + 1

    def re_activate(
        self,
        node_id: str,
        boost_amount: float,
        *,
        source: str = "direct",
        reinforce: bool = True,
        current_tick: Optional[int] = None,
        reinforcement_gain_direct: float = 0.2,
        reinforcement_gain_assoc: float = 0.05,
        stability_cap: float = 1.0,
        cascade_decay: float = 0.5,
    ) -> None:
        """Boost activation and optionally reinforce memory stability.

        Direct reactivation strengthens the target memory more than its neighbors.
        Cascaded neighbors receive a smaller activation/stability update.
        """
        if not self._graph.has_node(node_id):
            return

        self._apply_reactivation(
            node_id,
            boost_amount,
            source=source,
            reinforce=reinforce,
            current_tick=current_tick,
            direct_gain=reinforcement_gain_direct,
            assoc_gain=reinforcement_gain_assoc,
            stability_cap=stability_cap,
        )

        if source != "direct":
            return

        for neighbor, weight in self.get_associated(node_id):
            cascade_boost = boost_amount * weight * cascade_decay
            cascade_gain = reinforcement_gain_assoc * weight
            self._apply_reactivation(
                neighbor,
                cascade_boost,
                source="cascade",
                reinforce=reinforce,
                current_tick=current_tick,
                direct_gain=reinforcement_gain_direct,
                assoc_gain=cascade_gain,
                stability_cap=stability_cap,
            )

    def get_all_activations(self) -> dict[str, float]:
        return {
            nid: attrs["activation_score"]
            for nid, attrs in self._graph.nodes(data=True)
            if attrs.get("type") != "unknown"
        }

    def set_activation(self, node_id: str, score: float) -> None:
        if self._graph.has_node(node_id):
            self._graph.nodes[node_id]["activation_score"] = score

    def get_node(self, node_id: str) -> dict | None:
        if self._graph.has_node(node_id):
            return dict(self._graph.nodes[node_id])
        return None

    @property
    def num_memories(self) -> int:
        return sum(
            1 for _, attrs in self._graph.nodes(data=True)
            if attrs.get("type") != "unknown"
        )
