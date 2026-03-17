"""Graph-based memory store with embedding similarity retrieval."""

from __future__ import annotations

import numpy as np
import networkx as nx
import os
from typing import Optional, Callable


class MemoryGraph:
    """NetworkX DiGraph-based memory store.

    Nodes represent memory items with activation scores.
    Edges represent associations between memories.
    Embeddings enable semantic similarity queries.
    """

    def __init__(self, embedder: Optional[Callable] = None, embedding_backend: str = "local"):
        self._graph = nx.DiGraph()
        self._custom_embedder = embedder
        self._model = None
        self._embedding_backend = embedding_backend
        self._gemini_client = None
        self._embedding_dim = 768  # default for ko-sroberta

    def _get_embedder(self) -> Callable:
        """Get the embedding function based on backend."""
        if self._custom_embedder is not None:
            return self._custom_embedder

        if self._embedding_backend == "gemini":
            return self._gemini_embed
        else:
            return self._local_embed

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
                # Try loading from .env
                import pathlib
                env_path = pathlib.Path(__file__).parent.parent.parent / ".env"
                if env_path.exists():
                    for line in env_path.read_text().splitlines():
                        if line.startswith("GEMINI_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
                            break
            self._gemini_client = genai.Client(api_key=api_key)
            self._embedding_dim = 768

        result = self._gemini_client.models.embed_content(
            model="gemini-embedding-exp-03-07",
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
        embedder = self._get_embedder()
        embedding = np.array(embedder(content), dtype=np.float32)
        self._embedding_dim = embedding.shape[0]

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
                if not self._graph.has_node(target_id):
                    self._graph.add_node(
                        target_id,
                        type="unknown",
                        content="",
                        embedding=np.zeros(self._embedding_dim, dtype=np.float32),
                        activation_score=0.0,
                        impact=0.0,
                        created_tick=0,
                        last_activated_tick=0,
                    )
                self._graph.add_edge(
                    memory_id, target_id, weight=weight, created_tick=created_tick
                )
                if not self._graph.has_edge(target_id, memory_id):
                    self._graph.add_edge(
                        target_id, memory_id, weight=weight, created_tick=created_tick
                    )

    def query_by_similarity(
        self, query_text: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Find memories matching query via embedding cosine similarity."""
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
        """Get neighbors connected by association edges."""
        results = []
        for neighbor in self._graph.predecessors(node_id):
            w = self._graph.edges[neighbor, node_id].get("weight", 0.0)
            results.append((neighbor, w))
        for neighbor in self._graph.successors(node_id):
            w = self._graph.edges[node_id, neighbor].get("weight", 0.0)
            results.append((neighbor, w))
        return results

    def re_activate(self, node_id: str, boost_amount: float) -> None:
        """Boost activation and cascade to associated nodes (one hop)."""
        if not self._graph.has_node(node_id):
            return

        current = self._graph.nodes[node_id]["activation_score"]
        self._graph.nodes[node_id]["activation_score"] = min(current + boost_amount, 1.0)

        for neighbor, weight in self.get_associated(node_id):
            cascade_boost = boost_amount * weight * 0.5
            n_score = self._graph.nodes[neighbor]["activation_score"]
            self._graph.nodes[neighbor]["activation_score"] = min(
                n_score + cascade_boost, 1.0
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
