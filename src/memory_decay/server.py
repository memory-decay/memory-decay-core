"""FastAPI server exposing MemoryGraph + DecayEngine over HTTP.

Designed to be called by the openclaw-memory-decay TypeScript plugin.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .decay import DecayEngine
from .embedding_provider import EmbeddingProvider, create_embedding_provider
from .graph import MemoryGraph
from .persistence import MemoryPersistence

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class StoreRequest(BaseModel):
    text: str
    importance: float = Field(default=0.7, ge=0.0, le=1.0)
    category: str = "other"
    mtype: str = "fact"
    associations: list[str] | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)


class TickRequest(BaseModel):
    count: int = Field(default=1, ge=1, le=1000)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class ServerState:
    """Holds the in-memory graph, engine, and persistence."""

    def __init__(
        self,
        graph: MemoryGraph,
        engine: DecayEngine,
        persistence: MemoryPersistence | None,
        tick_interval_seconds: float = 3600.0,
    ):
        self.graph = graph
        self.engine = engine
        self.persistence = persistence
        self.current_tick = 0
        self.last_tick_time = time.time()
        self.tick_interval_seconds = tick_interval_seconds
        self._memory_counter = 0
        self._auto_save_interval = 300  # 5 minutes
        self._last_save_time = time.time()

    def next_memory_id(self) -> str:
        self._memory_counter += 1
        return f"mem_{uuid.uuid4().hex[:12]}"

    def maybe_auto_save(self) -> None:
        if self.persistence and (time.time() - self._last_save_time > self._auto_save_interval):
            self.persistence.save(self.graph, self.current_tick)
            self._last_save_time = time.time()


_state: ServerState | None = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    embedding_provider: EmbeddingProvider | None = None,
    persistence_dir: str | None = None,
    decay_type: str = "exponential",
    decay_params: dict | None = None,
    tick_interval_seconds: float = 3600.0,
    _test_embedder=None,
) -> FastAPI:
    """Create the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _state

        if _test_embedder:
            graph = MemoryGraph(embedder=_test_embedder)
        elif embedding_provider:
            graph = MemoryGraph(embedder=embedding_provider.embed)
        else:
            graph = MemoryGraph(embedding_backend="auto")

        engine = DecayEngine(graph, decay_type=decay_type, params=decay_params)

        persistence = None
        state_tick = 0
        if persistence_dir:
            persistence = MemoryPersistence(persistence_dir)
            meta = persistence.load(graph)
            if meta:
                state_tick = meta.get("current_tick", 0)

        _state = ServerState(graph, engine, persistence, tick_interval_seconds)
        _state.current_tick = state_tick

        yield

        if _state and _state.persistence:
            _state.persistence.save(_state.graph, _state.current_tick)
        _state = None

    app = FastAPI(title="memory-decay", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "current_tick": _state.current_tick if _state else 0}

    @app.get("/stats")
    def stats():
        if not _state:
            raise HTTPException(503, "Server not initialized")
        return {
            "num_memories": _state.graph.num_memories,
            "current_tick": _state.current_tick,
            "last_tick_time": _state.last_tick_time,
        }

    @app.post("/store")
    def store(req: StoreRequest):
        if not _state:
            raise HTTPException(503, "Server not initialized")

        memory_id = _state.next_memory_id()
        associations = None
        if req.associations:
            associations = [(a, 0.5) for a in req.associations]

        _state.graph.add_memory(
            memory_id=memory_id,
            mtype=req.mtype,
            content=req.text,
            impact=req.importance,
            created_tick=_state.current_tick,
            associations=associations,
        )
        _state.maybe_auto_save()

        return {"id": memory_id, "text": req.text, "tick": _state.current_tick}

    @app.post("/search")
    def search(req: SearchRequest):
        if not _state:
            raise HTTPException(503, "Server not initialized")

        params = _state.engine.get_params()
        results = _state.graph.query_by_similarity(
            query_text=req.query,
            top_k=req.top_k,
            current_tick=_state.current_tick,
            activation_weight=params.get("activation_weight", 0.5),
            assoc_boost=params.get("assoc_boost", 0.0),
        )

        enriched = []
        for node_id, score in results:
            node = _state.graph.get_node(node_id)
            if node:
                enriched.append({
                    "id": node_id,
                    "text": node.get("content", ""),
                    "score": round(score, 4),
                    "storage_score": round(node.get("storage_score", 0.0), 4),
                    "retrieval_score": round(node.get("retrieval_score", 0.0), 4),
                    "category": node.get("type", "unknown"),
                    "created_tick": node.get("created_tick", 0),
                })
        return {"results": enriched}

    @app.post("/tick")
    def tick(req: TickRequest):
        if not _state:
            raise HTTPException(503, "Server not initialized")

        for _ in range(req.count):
            _state.engine.tick()
            _state.current_tick += 1
        _state.last_tick_time = time.time()
        _state.maybe_auto_save()

        return {"current_tick": _state.current_tick}

    @app.post("/auto-tick")
    def auto_tick():
        """Apply ticks based on elapsed real time since last tick."""
        if not _state:
            raise HTTPException(503, "Server not initialized")

        elapsed = time.time() - _state.last_tick_time
        ticks_due = int(elapsed / _state.tick_interval_seconds)
        if ticks_due > 0:
            ticks_due = min(ticks_due, 100)
            for _ in range(ticks_due):
                _state.engine.tick()
                _state.current_tick += 1
            _state.last_tick_time = time.time()
            _state.maybe_auto_save()

        return {
            "ticks_applied": ticks_due,
            "current_tick": _state.current_tick,
            "elapsed_seconds": round(elapsed, 1),
        }

    @app.delete("/forget/{memory_id}")
    def forget(memory_id: str):
        if not _state:
            raise HTTPException(503, "Server not initialized")

        node = _state.graph.get_node(memory_id)
        if node is None:
            raise HTTPException(404, f"Memory {memory_id} not found")

        _state.graph._graph.remove_node(memory_id)
        _state.graph._emb_node_count = 0  # force matrix rebuild
        _state.maybe_auto_save()

        return {"deleted": memory_id}

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """Run the server from command line."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="memory-decay HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--persistence-dir", default=None)
    parser.add_argument("--decay-type", default="exponential")
    parser.add_argument("--tick-interval", type=float, default=3600.0,
                        help="Real seconds per tick")
    parser.add_argument("--embedding-provider", default="gemini",
                        choices=["gemini", "openai"])
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-api-key", default=None)
    args = parser.parse_args()

    provider = None
    if args.embedding_api_key:
        provider = create_embedding_provider(
            provider=args.embedding_provider,
            api_key=args.embedding_api_key,
            model=args.embedding_model,
        )

    app = create_app(
        embedding_provider=provider,
        persistence_dir=args.persistence_dir,
        decay_type=args.decay_type,
        tick_interval_seconds=args.tick_interval,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
