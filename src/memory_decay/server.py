"""FastAPI server exposing MemoryStore + DecayEngine over HTTP.

Designed to be called by the openclaw-memory-decay TypeScript plugin.
All endpoints are async to avoid blocking the event loop during
embedding API calls or SQLite I/O.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .decay import DecayEngine
from .embedding_provider import EmbeddingProvider, create_embedding_provider
from .memory_store import MemoryStore

# ---------------------------------------------------------------------------
# Best experiment loader
# ---------------------------------------------------------------------------

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
_DEFAULT_DB_DIR = _PACKAGE_ROOT / "data"


def _load_best_experiment(experiment_dir: Path | None = None) -> tuple[dict, object | None]:
    """Load best experiment params and custom decay function.

    Returns (params_dict, decay_fn_or_None).
    """
    if experiment_dir is None:
        experiment_dir = _PACKAGE_ROOT / "experiments" / "best"

    params = {}
    params_path = experiment_dir / "params.json"
    if params_path.exists():
        params = json.loads(params_path.read_text())

    decay_fn = None
    decay_fn_path = experiment_dir / "decay_fn.py"
    if decay_fn_path.exists():
        spec = importlib.util.spec_from_file_location("best_decay_fn", decay_fn_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "compute_decay"):
            decay_fn = mod.compute_decay

    return params, decay_fn



# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class StoreRequest(BaseModel):
    text: str
    importance: float = Field(default=0.7, ge=0.0, le=1.0)
    mtype: str = "fact"
    category: str = ""
    associations: Optional[List[str]] = None
    created_tick: Optional[int] = None
    speaker: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)


class TickRequest(BaseModel):
    count: int = Field(default=1, ge=1, le=1000)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class ServerState:
    """Holds the SQLite-backed store, engine, and embedder."""

    def __init__(
        self,
        store: MemoryStore,
        engine: DecayEngine,
        embedder: Callable | None = None,
        provider: EmbeddingProvider | None = None,
        tick_interval_seconds: float = 3600.0,
        embedding_model: str = "",
    ):
        self.store = store
        self.engine = engine
        self._embedder = embedder
        self._provider = provider
        self.current_tick = 0
        self.last_tick_time = time.time()
        self.tick_interval_seconds = tick_interval_seconds
        self._embedding_model = embedding_model

    def next_memory_id(self) -> str:
        return f"mem_{uuid.uuid4().hex[:12]}"

    async def embed(self, text: str) -> np.ndarray:
        """Get embedding for text, using store cache first."""
        cached = await asyncio.to_thread(
            self.store.get_cached_embedding, text, model=self._embedding_model
        )
        if cached is not None:
            return cached

        if self._provider is not None:
            embedding = await self._provider.aembed(text)
        elif self._embedder is not None:
            embedding = await asyncio.to_thread(self._embedder, text)
            embedding = np.array(embedding, dtype=np.float32)
        else:
            raise RuntimeError("No embedding provider configured")

        await asyncio.to_thread(
            self.store.cache_embedding, text, embedding, model=self._embedding_model
        )
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Batch embed with cache: only compute uncached texts."""
        results: list[np.ndarray | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache for all texts in a single thread call
        def _check_cache():
            for i, text in enumerate(texts):
                cached = self.store.get_cached_embedding(text, model=self._embedding_model)
                if cached is not None:
                    results[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)

        await asyncio.to_thread(_check_cache)

        # Batch embed uncached texts
        if uncached_texts:
            if self._provider is not None:
                new_embeddings = await self._provider.aembed_batch(uncached_texts)
            elif self._embedder is not None:
                new_embeddings = await asyncio.to_thread(
                    lambda ts: [np.array(self._embedder(t), dtype=np.float32) for t in ts],
                    uncached_texts,
                )
            else:
                raise RuntimeError("No embedding provider configured")

            for i, (idx, embedding) in enumerate(zip(uncached_indices, new_embeddings)):
                results[idx] = embedding

            # Cache all new embeddings in one transaction
            def _cache_all():
                for i, emb in enumerate(new_embeddings):
                    self.store.cache_embedding(
                        uncached_texts[i], emb, model=self._embedding_model, auto_commit=False
                    )
                self.store.commit()

            await asyncio.to_thread(_cache_all)

        return results  # type: ignore[return-value]


_state: ServerState | None = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_default_app() -> FastAPI:
    """Factory for uvicorn multi-worker mode. Reads config from env vars."""
    import os
    provider_name = os.environ.get("MD_EMBEDDING_PROVIDER", "local")
    provider = create_embedding_provider(
        provider=provider_name,
        api_key=os.environ.get("MD_EMBEDDING_API_KEY"),
        model=os.environ.get("MD_EMBEDDING_MODEL"),
    )
    dim = os.environ.get("MD_EMBEDDING_DIM")
    return create_app(
        embedding_provider=provider,
        tick_interval_seconds=float(os.environ.get("MD_TICK_INTERVAL", "3600")),
        experiment_dir=os.environ.get("MD_EXPERIMENT_DIR"),
        db_path=os.environ.get("MD_DB_PATH"),
        embedding_dim=int(dim) if dim else None,
    )


def create_app(
    embedding_provider: EmbeddingProvider | None = None,
    tick_interval_seconds: float = 3600.0,
    experiment_dir: Path | str | None = None,
    db_path: str | None = None,
    embedding_dim: int | None = None,
    _test_embedder=None,
) -> FastAPI:
    """Create the FastAPI application.

    Automatically loads the best experiment's decay function and parameters.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _state

        # --- Resolve embedder function and dimension ---
        base_embedder = None
        resolved_provider = embedding_provider

        if _test_embedder:
            base_embedder = _test_embedder
            resolved_provider = None
            resolved_embedding_dim = embedding_dim or len(_test_embedder("_dim_probe_"))
        elif embedding_provider:
            resolved_embedding_dim = embedding_dim or embedding_provider.dimension
        else:
            from .graph import MemoryGraph
            _graph = MemoryGraph(embedding_backend="auto")
            base_embedder = _graph._embed_text
            resolved_embedding_dim = embedding_dim or 768

        # --- Resolve DB path ---
        resolved_db_path = db_path or ":memory:"
        if resolved_db_path != ":memory:":
            resolved_db_path = str(Path(resolved_db_path).expanduser())
            db_dir = Path(resolved_db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

        # --- Create MemoryStore ---
        store = MemoryStore(resolved_db_path, embedding_dim=resolved_embedding_dim)

        # --- Load best experiment (decay function + tuned parameters) ---
        exp_dir = Path(experiment_dir) if experiment_dir else None
        best_params, best_decay_fn = _load_best_experiment(exp_dir)
        engine = DecayEngine(
            store=store,
            custom_decay_fn=best_decay_fn,
            params=best_params,
        )

        # --- Restore current_tick from store metadata ---
        state_tick = int(store.get_metadata("current_tick", "0"))
        engine.current_tick = state_tick

        # Resolve model name for cache keying
        if _test_embedder:
            model_name = "test"
        elif embedding_provider:
            model_name = getattr(embedding_provider, '_model', '') or getattr(embedding_provider, '_model_name', '')
        else:
            model_name = "ko-sroberta"

        _state = ServerState(
            store, engine,
            embedder=base_embedder,
            provider=resolved_provider,
            tick_interval_seconds=tick_interval_seconds,
            embedding_model=model_name,
        )
        _state.current_tick = state_tick

        yield

        # --- Shutdown: save tick to metadata and close ---
        if _state:
            _state.store.set_metadata("current_tick", str(_state.current_tick))
            _state.store.close()
        _state = None

    app = FastAPI(title="memory-decay", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "current_tick": _state.current_tick if _state else 0}

    @app.get("/stats")
    async def stats():
        if not _state:
            raise HTTPException(503, "Server not initialized")
        num = await asyncio.to_thread(lambda: _state.store.num_memories)
        return {
            "num_memories": num,
            "current_tick": _state.current_tick,
            "last_tick_time": _state.last_tick_time,
        }

    @app.post("/store")
    async def store(req: StoreRequest):
        if not _state:
            raise HTTPException(503, "Server not initialized")

        memory_id = _state.next_memory_id()
        associations = None
        if req.associations:
            associations = [(a, 0.5) for a in req.associations]

        embedding = await _state.embed(req.text)

        await asyncio.to_thread(
            lambda: _state.store.add_memory(
                memory_id=memory_id,
                content=req.text,
                embedding=embedding,
                mtype=req.mtype,
                category=req.category,
                importance=req.importance,
                created_tick=req.created_tick if req.created_tick is not None else _state.current_tick,
                associations=associations,
                speaker=req.speaker or "",
            )
        )

        return {"id": memory_id, "text": req.text, "tick": _state.current_tick}

    @app.post("/store-batch")
    async def store_batch(items: list[StoreRequest]):
        if not _state:
            raise HTTPException(503, "Server not initialized")

        # Batch embed all texts at once
        texts = [req.text for req in items]
        embeddings = await _state.embed_batch(texts)

        # Build batch and insert in single transaction
        memories = []
        ids = []
        for req, embedding in zip(items, embeddings):
            memory_id = _state.next_memory_id()
            associations = None
            if req.associations:
                associations = [(a, 0.5) for a in req.associations]
            memories.append({
                "memory_id": memory_id,
                "content": req.text,
                "embedding": embedding,
                "mtype": req.mtype,
                "category": req.category,
                "importance": req.importance,
                "created_tick": req.created_tick if req.created_tick is not None else _state.current_tick,
                "associations": associations,
                "speaker": req.speaker or "",
            })
            ids.append(memory_id)

        await asyncio.to_thread(_state.store.add_memories_batch, memories)

        return {"ids": ids, "count": len(ids), "tick": _state.current_tick}

    @app.post("/search")
    async def search(req: SearchRequest):
        if not _state:
            raise HTTPException(503, "Server not initialized")

        query_embedding = await _state.embed(req.query)
        params = _state.engine.get_params()

        results = await asyncio.to_thread(
            lambda: _state.store.search(
                query_embedding=query_embedding,
                top_k=req.top_k,
                current_tick=_state.current_tick,
                activation_weight=params.get("activation_weight", 0.5),
                bm25_weight=params.get("bm25_weight", 0.0),
                query_text=req.query,
            )
        )

        # Retrieval consolidation: boost top result on successful recall
        if results and results[0]["score"] > 0.3:
            await asyncio.to_thread(
                lambda: _state.store.reinforce(
                    results[0]["id"],
                    retrieval_boost=params.get("retrieval_boost", 0.10),
                    stability_gain=params.get("reinforcement_gain_direct", 0.2),
                    stability_cap=params.get("stability_cap", 1.0),
                )
            )

        return {"results": results}

    @app.post("/tick")
    async def tick(req: TickRequest):
        if not _state:
            raise HTTPException(503, "Server not initialized")

        def _do_ticks():
            for _ in range(req.count):
                _state.engine.tick()
            _state.current_tick = _state.engine.current_tick
            _state.last_tick_time = time.time()

        await asyncio.to_thread(_do_ticks)

        return {"current_tick": _state.current_tick}

    @app.post("/auto-tick")
    async def auto_tick():
        """Apply ticks based on elapsed real time since last tick."""
        if not _state:
            raise HTTPException(503, "Server not initialized")

        elapsed = time.time() - _state.last_tick_time
        ticks_due = int(elapsed / _state.tick_interval_seconds)

        if ticks_due > 0:
            ticks_due = min(ticks_due, 100)

            def _do_ticks():
                for _ in range(ticks_due):
                    _state.engine.tick()
                _state.current_tick = _state.engine.current_tick
                _state.last_tick_time = time.time()

            await asyncio.to_thread(_do_ticks)

        return {
            "ticks_applied": ticks_due,
            "current_tick": _state.current_tick,
            "elapsed_seconds": round(elapsed, 1),
        }

    @app.delete("/forget/{memory_id}")
    async def forget(memory_id: str):
        if not _state:
            raise HTTPException(503, "Server not initialized")

        node = await asyncio.to_thread(_state.store.get_node, memory_id)
        if node is None:
            raise HTTPException(404, f"Memory {memory_id} not found")

        await asyncio.to_thread(_state.store.delete_memory, memory_id)

        return {"deleted": memory_id}

    @app.post("/reset")
    async def reset():
        if not _state:
            raise HTTPException(503, "Server not initialized")

        def _do_reset():
            cleared = _state.store.clear()
            _state.engine.reset()
            _state.current_tick = 0
            _state.last_tick_time = time.time()
            return cleared

        cleared = await asyncio.to_thread(_do_reset)

        return {"status": "ok", "cleared": cleared}

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
    parser.add_argument("--db-path", default=None,
                        help="Path to SQLite DB file (default: data/memories.db)")
    parser.add_argument("--experiment-dir", default=None,
                        help="Path to experiment dir (default: experiments/best)")
    parser.add_argument("--tick-interval", type=float, default=3600.0,
                        help="Real seconds per tick")
    parser.add_argument("--embedding-provider", default="local",
                        choices=["local", "gemini", "openai"])
    parser.add_argument("--embedding-model", default=None,
                        help="Model name (default: auto per provider)")
    parser.add_argument("--embedding-api-key", default=None)
    parser.add_argument("--embedding-dim", type=int, default=None,
                        help="Embedding dimension (default: auto-detect from provider)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of uvicorn workers (default: 1)")
    args = parser.parse_args()

    provider = create_embedding_provider(
        provider=args.embedding_provider,
        api_key=args.embedding_api_key,
        model=args.embedding_model,
    )

    # Auto-detect dimension from provider if not specified
    resolved_dim = args.embedding_dim or provider.dimension

    resolved_db_path = args.db_path
    if resolved_db_path is None:
        resolved_db_path = str(_DEFAULT_DB_DIR / "memories.db")

    if args.workers > 1:
        # Multi-worker: pass config via env vars so each worker can rebuild
        import os
        os.environ["MD_EMBEDDING_PROVIDER"] = args.embedding_provider
        if args.embedding_api_key:
            os.environ["MD_EMBEDDING_API_KEY"] = args.embedding_api_key
        if args.embedding_model:
            os.environ["MD_EMBEDDING_MODEL"] = args.embedding_model
        if resolved_dim:
            os.environ["MD_EMBEDDING_DIM"] = str(resolved_dim)
        os.environ["MD_TICK_INTERVAL"] = str(args.tick_interval)
        os.environ["MD_DB_PATH"] = resolved_db_path
        if args.experiment_dir:
            os.environ["MD_EXPERIMENT_DIR"] = args.experiment_dir
        uvicorn.run(
            "memory_decay.server:create_default_app",
            factory=True,
            host=args.host,
            port=args.port,
            workers=args.workers,
        )
    else:
        app = create_app(
            embedding_provider=provider,
            tick_interval_seconds=args.tick_interval,
            experiment_dir=args.experiment_dir,
            db_path=resolved_db_path,
            embedding_dim=resolved_dim,
        )
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
