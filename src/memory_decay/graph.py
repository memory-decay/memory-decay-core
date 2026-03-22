"""Graph-based memory store with embedding similarity retrieval."""

from __future__ import annotations

import math
import os
import re
import time
from collections import Counter
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
        self._query_similarity_total_time: float = 0.0
        self._query_similarity_call_count: int = 0
        # Precomputed embedding matrix for vectorized similarity search
        self._emb_matrix: np.ndarray | None = None
        self._emb_nids: list[str] = []
        self._emb_nid_to_idx: dict[str, int] = {}
        self._emb_created_ticks: np.ndarray | None = None
        self._emb_retrieval_scores: np.ndarray | None = None
        self._emb_node_count: int = 0  # tracks when matrix needs rebuild
        # BM25 global IDF index (built alongside embedding matrix)
        self._bm25_idf: dict[str, float] | None = None
        self._bm25_doc_tokens: dict[str, list[str]] | None = None
        self._bm25_avgdl: float = 0.0

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

        model_name = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        result = self._gemini_client.models.embed_content(
            model=model_name,
            contents=text,
        )
        return np.array(result.embeddings[0].values, dtype=np.float32)

    def _gemini_embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Batch Gemini API embedding via google-genai SDK."""
        if self._gemini_client is None:
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is required for Gemini embeddings.")
            self._gemini_client = genai.Client(api_key=api_key)
            self._embedding_dim = 768

        model_name = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        result = self._gemini_client.models.embed_content(
            model=model_name,
            contents=texts,  # type: ignore[reportArgumentType]
        )
        embeddings = [np.array(emb.values, dtype=np.float32) for emb in result.embeddings]  # type: ignore[reportOptionalIterable]
        if embeddings:
            self._embedding_dim = embeddings[0].shape[0]
        return embeddings

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
            storage_score=1.0,
            retrieval_score=1.0,
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
                        storage_score=0.0,
                        retrieval_score=0.0,
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

    @staticmethod
    def _bm25_tokenize(text: str) -> list[str]:
        return re.findall(r"[0-9A-Za-z가-힣]+", text.lower())

    def _bm25_score_candidates(
        self,
        query_text: str,
        candidate_nids: list[str],
        k1: float = 1.2,
        b: float = 0.75,
    ) -> dict[str, float]:
        """Score candidates against query using BM25 with pre-computed global IDF."""
        if self._bm25_idf is None or self._bm25_doc_tokens is None:
            return {}

        query_terms = list(dict.fromkeys(self._bm25_tokenize(query_text)))
        if not query_terms:
            return {}

        avgdl = max(self._bm25_avgdl, 1.0)
        scores: dict[str, float] = {}

        for nid in candidate_nids:
            tokens = self._bm25_doc_tokens.get(nid, [])
            tf = Counter(tokens)
            dl = max(len(tokens), 1)
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                idf = self._bm25_idf.get(term, 0.0)
                denom = freq + k1 * (1.0 - b + b * dl / avgdl)
                score += idf * (freq * (k1 + 1.0)) / max(denom, 1e-9)
            scores[nid] = score

        return scores

    def _ensure_embedding_matrix(self) -> None:
        """Build or rebuild the precomputed embedding matrix if needed."""
        current_count = self._graph.number_of_nodes()
        if self._emb_matrix is not None and self._emb_node_count == current_count:
            return

        nids = []
        embeddings = []
        created_ticks = []
        for nid, attrs in self._graph.nodes(data=True):
            if attrs.get("type") == "unknown":
                continue
            emb = attrs.get("embedding")
            if emb is None:
                continue
            nids.append(nid)
            embeddings.append(emb)
            created_ticks.append(attrs.get("created_tick", 0))

        if embeddings:
            matrix = np.array(embeddings, dtype=np.float64)
            # Pre-normalize rows for cosine similarity via dot product
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # avoid division by zero
            self._emb_matrix = matrix / norms
            self._emb_nids = nids
            self._emb_nid_to_idx = {nid: i for i, nid in enumerate(nids)}
            self._emb_created_ticks = np.array(created_ticks, dtype=np.int64)
            # Pre-populate retrieval scores array
            self._emb_retrieval_scores = np.array([
                max(float(self._graph.nodes[nid].get(
                    "retrieval_score",
                    self._graph.nodes[nid].get("activation_score", 0.0),
                )), 0.0)
                for nid in nids
            ], dtype=np.float64)
            # Build BM25 global IDF index
            doc_freq: Counter[str] = Counter()
            total_tokens = 0
            bm25_doc_tokens: dict[str, list[str]] = {}
            for nid in nids:
                content = self._graph.nodes[nid].get("content", "")
                tokens = self._bm25_tokenize(content)
                bm25_doc_tokens[nid] = tokens
                doc_freq.update(set(tokens))
                total_tokens += len(tokens)

            n_docs = len(nids)
            self._bm25_avgdl = total_tokens / max(n_docs, 1)
            self._bm25_idf = {}
            for term, df in doc_freq.items():
                self._bm25_idf[term] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            self._bm25_doc_tokens = bm25_doc_tokens
        else:
            self._emb_matrix = None
            self._emb_nids = []
            self._emb_nid_to_idx = {}
            self._emb_created_ticks = None
            self._emb_retrieval_scores = None
            self._bm25_idf = None
            self._bm25_doc_tokens = None
            self._bm25_avgdl = 0.0

        self._emb_node_count = current_count

    def query_by_similarity(
        self, query_text: str, top_k: int = 5, current_tick: int | None = None,
        activation_weight: float = 0.0,
        assoc_boost: float = 0.0,
        bm25_weight: float = 0.0,
        bm25_candidates: int = 20,
    ) -> list[tuple[str, float]]:
        """Find memories matching query via embedding cosine similarity.

        When *activation_weight* > 0 the ranking score becomes:

            score = cosine_sim * activation ^ activation_weight

        When *assoc_boost* > 0, each candidate's score is further boosted
        by the mean activation of its associated neighbors, weighted by
        edge weight.  This implements spreading-activation retrieval:
        memories whose neighbors are also active rank higher.

            score *= (1 + assoc_boost * mean_neighbor_activation)

        When *bm25_weight* > 0, a two-stage retrieval is performed:
        1. Fetch *bm25_candidates* results by cosine similarity
        2. Re-rank using BM25 lexical matching with global IDF
        3. Combined score = (1-bm25_weight)*cosine + bm25_weight*bm25
        4. Return top *top_k* by combined score
        """
        t0 = time.perf_counter()
        self._ensure_embedding_matrix()

        if self._emb_matrix is None or len(self._emb_nids) == 0:
            self._query_similarity_total_time += time.perf_counter() - t0
            self._query_similarity_call_count += 1
            return []

        query_vec = self._embed_text(query_text)
        norm_q = np.linalg.norm(query_vec)
        if norm_q == 0:
            self._query_similarity_total_time += time.perf_counter() - t0
            self._query_similarity_call_count += 1
            return []
        query_normalized = query_vec / norm_q

        # Vectorized cosine similarity: dot product with pre-normalized matrix
        similarities = self._emb_matrix @ query_normalized

        # Apply current_tick filter
        if current_tick is not None:
            mask = self._emb_created_ticks <= current_tick
            similarities = np.where(mask, similarities, -np.inf)

        # Apply activation weighting via cached score array
        if activation_weight > 0 and self._emb_retrieval_scores is not None:
            similarities = similarities * (self._emb_retrieval_scores ** activation_weight)

        # Get top candidates (fetch at least bm25_candidates when BM25 is active)
        fetch_k = max(top_k, bm25_candidates) if bm25_weight > 0 else top_k
        if len(self._emb_nids) <= fetch_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -fetch_k)[-fetch_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        candidates = [
            (self._emb_nids[i], float(similarities[i]))
            for i in top_indices
            if similarities[i] > -np.inf
        ]

        # Second pass: associative boost
        if assoc_boost > 0:
            boosted = []
            for nid, score in candidates:
                neighbors = self.get_associated(nid)
                if neighbors:
                    weighted_act_sum = 0.0
                    weight_sum = 0.0
                    for neighbor_id, edge_weight in neighbors:
                        n_node = self._graph.nodes.get(neighbor_id)
                        if n_node and n_node.get("type") != "unknown":
                            n_act = max(
                                float(
                                    n_node.get(
                                        "retrieval_score",
                                        n_node.get("activation_score", 0.0),
                                    )
                                ),
                                0.0,
                            )
                            weighted_act_sum += edge_weight * n_act
                            weight_sum += edge_weight
                    if weight_sum > 0:
                        mean_neighbor_act = weighted_act_sum / weight_sum
                        score *= (1.0 + assoc_boost * mean_neighbor_act)
                boosted.append((nid, score))
            candidates = boosted

        # BM25 re-ranking pass
        if bm25_weight > 0 and len(candidates) > 0:
            cand_nids = [nid for nid, _ in candidates]
            bm25_scores = self._bm25_score_candidates(query_text, cand_nids)

            if bm25_scores:
                cos_scores = {nid: score for nid, score in candidates}
                cos_vals = list(cos_scores.values())
                cos_min, cos_max = min(cos_vals), max(cos_vals)
                cos_range = cos_max - cos_min

                bm25_max = max(bm25_scores.values())

                combined = []
                for nid in cand_nids:
                    norm_cos = (cos_scores[nid] - cos_min) / max(cos_range, 1e-8)
                    norm_bm25 = bm25_scores.get(nid, 0.0) / max(bm25_max, 1e-8)
                    combined_score = (1.0 - bm25_weight) * norm_cos + bm25_weight * norm_bm25
                    combined.append((nid, combined_score))
                candidates = combined

        candidates.sort(key=lambda x: x[1], reverse=True)
        self._query_similarity_total_time += time.perf_counter() - t0
        self._query_similarity_call_count += 1
        return candidates[:top_k]

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
        score_mode: str,
    ) -> None:
        attrs = self._graph.nodes[node_id]
        effective_tick = self._effective_reinforce_tick(node_id, current_tick)

        if score_mode in ("both", "retrieval_only"):
            retrieval_score = min(
                max(
                    float(attrs.get("retrieval_score", attrs.get("activation_score", 0.0)))
                    + boost_amount,
                    0.0,
                ),
                1.0,
            )
            attrs["retrieval_score"] = retrieval_score
            attrs["activation_score"] = retrieval_score
            attrs["last_activated_tick"] = effective_tick
            idx = self._emb_nid_to_idx.get(node_id)
            if idx is not None and self._emb_retrieval_scores is not None:
                self._emb_retrieval_scores[idx] = max(retrieval_score, 0.0)

        if score_mode in ("both", "storage_only"):
            storage_score = min(
                max(float(attrs.get("storage_score", attrs.get("activation_score", 0.0))) + boost_amount, 0.0),
                1.0,
            )
            attrs["storage_score"] = storage_score
            attrs["last_activated_tick"] = effective_tick

        if not reinforce or attrs.get("type") == "unknown":
            return

        gain = direct_gain if source == "direct" else assoc_gain
        current_stability = float(attrs.get("stability_score", 0.0))
        saturation = max(1.0 - current_stability / max(stability_cap, 1e-9), 0.0)
        attrs["stability_score"] = min(
            max(current_stability + gain * saturation, 0.0), stability_cap
        )
        attrs["last_reinforced_tick"] = effective_tick
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
        score_mode: str = "both",
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
            score_mode=score_mode,
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
                score_mode=score_mode,
            )

    def reinforce_memory(
        self,
        node_id: str,
        *,
        reinforcement_gain: float,
        stability_cap: float,
        current_tick: Optional[int] = None,
        count_as_retrieval: bool = True,
    ) -> None:
        """Increase stability without changing activation or cascading."""
        if not self._graph.has_node(node_id):
            return

        attrs = self._graph.nodes[node_id]
        if attrs.get("type") in ("unknown", None):
            return

        current_stability = float(attrs.get("stability_score", 0.0))
        saturation = max(1.0 - current_stability / max(stability_cap, 1e-9), 0.0)
        attrs["stability_score"] = min(
            max(current_stability + reinforcement_gain * saturation, 0.0),
            stability_cap,
        )
        effective_tick = self._effective_reinforce_tick(node_id, current_tick)
        attrs["last_reinforced_tick"] = effective_tick
        attrs["last_activated_tick"] = effective_tick
        if count_as_retrieval:
            attrs["retrieval_count"] = int(attrs.get("retrieval_count", 0)) + 1

    def get_all_activations(self) -> dict[str, float]:
        return {
            nid: attrs.get("retrieval_score", attrs["activation_score"])
            for nid, attrs in self._graph.nodes(data=True)
            if attrs.get("type") != "unknown"
        }

    def set_activation(self, node_id: str, score: float) -> None:
        if self._graph.has_node(node_id):
            self._graph.nodes[node_id]["retrieval_score"] = score
            self._graph.nodes[node_id]["activation_score"] = score
            idx = self._emb_nid_to_idx.get(node_id)
            if idx is not None and self._emb_retrieval_scores is not None:
                self._emb_retrieval_scores[idx] = max(score, 0.0)

    def set_storage_score(self, node_id: str, score: float) -> None:
        if self._graph.has_node(node_id):
            self._graph.nodes[node_id]["storage_score"] = score

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
