"""FastAPI server exposing MemoryStore + DecayEngine over HTTP.

Designed to be called by the openclaw-memory-decay TypeScript plugin.
"""

from __future__ import annotations

import importlib.util
import json
import pickle  # noqa: S403 — self-generated local cache only
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .cache_builder import load_cached_embedder
from .decay import DecayEngine
from .embedding_provider import EmbeddingProvider, create_embedding_provider
from .memory_store import MemoryStore

# ---------------------------------------------------------------------------
# Best experiment loader
# ---------------------------------------------------------------------------

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
_DEFAULT_CACHE_DIR = _PACKAGE_ROOT / "cache"
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


def _resolve_cache_dir(cache_dir: str | None) -> Path | None:
    """Return the embedding cache dir when one is available."""
    if cache_dir:
        resolved = Path(cache_dir)
        if not (resolved / "embeddings.pkl").exists():
            raise FileNotFoundError(f"Embedding cache not found at {resolved / 'embeddings.pkl'}")
        return resolved

    if (_DEFAULT_CACHE_DIR / "embeddings.pkl").exists():
        return _DEFAULT_CACHE_DIR

    return None

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class StoreRequest(BaseModel):
    text: str
    importance: float = Field(default=0.7, ge=0.0, le=1.0)
    category: str = "other"
    mtype: str = "fact"
    associations: Optional[List[str]] = None
    created_tick: Optional[int] = None
    speaker: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)


from enum import Enum

class FeedbackSignal(str, Enum):
    positive = "positive"
    negative = "negative"


class FeedbackItem(BaseModel):
    memory_id: str
    signal: FeedbackSignal
    strength: float = Field(default=0.5, ge=0.1, le=1.0)


class FeedbackRequest(BaseModel):
    items: List[FeedbackItem] = Field(default_factory=list)


class FeedbackSingleRequest(BaseModel):
    memory_id: str
    signal: FeedbackSignal
    strength: float = Field(default=0.5, ge=0.1, le=1.0)


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
        embedder: Callable,
        tick_interval_seconds: float = 3600.0,
    ):
        self.store = store
        self.engine = engine
        self.embedder = embedder
        self.current_tick = 0
        self.last_tick_time = time.time()
        self.tick_interval_seconds = tick_interval_seconds

    def next_memory_id(self) -> str:
        return f"mem_{uuid.uuid4().hex[:12]}"

    def embed(self, text: str) -> np.ndarray:
        """Get embedding for text, using store cache first."""
        cached = self.store.get_cached_embedding(text)
        if cached is not None:
            return cached
        embedding = np.array(self.embedder(text), dtype=np.float32)
        self.store.cache_embedding(text, embedding)
        return embedding


_state: ServerState | None = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    embedding_provider: EmbeddingProvider | None = None,
    cache_dir: str | None = None,
    persistence_dir: str | None = None,
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

        # Only auto-detect default cache dir when no test embedder is provided
        # (test embedder is authoritative in tests — default cache is a production artifact)
        resolved_cache_dir = _resolve_cache_dir(cache_dir) if (cache_dir or not _test_embedder) else None
        cached_embedding_dim: int | None = None
        if resolved_cache_dir is not None:
            pkl_path = Path(resolved_cache_dir) / "embeddings.pkl"
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    cached_embeddings: dict[str, np.ndarray] = pickle.load(f)  # noqa: S301
                if cached_embeddings:
                    cached_embedding_dim = len(next(iter(cached_embeddings.values())))

        # --- Resolve embedder function and dimension ---
        if _test_embedder:
            base_embedder = _test_embedder
        elif embedding_provider:
            base_embedder = embedding_provider.embed
            provider_embedding_dim = embedding_provider.dimension
        else:
            from .graph import MemoryGraph
            _graph = MemoryGraph(embedding_backend="auto")
            base_embedder = _graph._embed_text
            provider_embedding_dim = 768

        resolved_embedding_dim = embedding_dim or cached_embedding_dim
        if resolved_embedding_dim is None:
            if _test_embedder:
                resolved_embedding_dim = len(_test_embedder("_dim_probe_"))
            else:
                resolved_embedding_dim = provider_embedding_dim

        # Wrap with PKL cache if available
        if resolved_cache_dir is not None:
            embedder_fn = load_cached_embedder(
                str(resolved_cache_dir),
                fallback_embedder=base_embedder,
            )
        else:
            embedder_fn = base_embedder

        # --- Resolve DB path ---
        resolved_db_path = db_path or ":memory:"
        if resolved_db_path != ":memory:":
            db_dir = Path(resolved_db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

        # --- Create MemoryStore ---
        store = MemoryStore(resolved_db_path, embedding_dim=resolved_embedding_dim)

        # --- Migrate PKL embedding cache into SQLite if DB is new ---
        if resolved_cache_dir is not None and store.num_memories == 0:
            pkl_path = Path(resolved_cache_dir) / "embeddings.pkl"
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    pkl_cache: dict[str, np.ndarray] = pickle.load(f)  # noqa: S301
                for text, emb in pkl_cache.items():
                    store.cache_embedding(text, np.array(emb, dtype=np.float32))

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

        _state = ServerState(store, engine, embedder_fn, tick_interval_seconds)
        _state.current_tick = state_tick

        yield

        # --- Shutdown: save tick to metadata and close ---
        if _state:
            _state.store.set_metadata("current_tick", str(_state.current_tick))
            _state.store.close()
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
            "num_memories": _state.store.num_memories,
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

        embedding = _state.embed(req.text)

        _state.store.add_memory(
            memory_id=memory_id,
            content=req.text,
            embedding=embedding,
            mtype=req.mtype,
            importance=req.importance,
            created_tick=req.created_tick if req.created_tick is not None else _state.current_tick,
            associations=associations,
            speaker=req.speaker or "",
        )

        return {"id": memory_id, "text": req.text, "tick": _state.current_tick}

    @app.post("/store-batch")
    def store_batch(items: list[StoreRequest]):
        if not _state:
            raise HTTPException(503, "Server not initialized")

        ids = []
        for req in items:
            memory_id = _state.next_memory_id()
            associations = None
            if req.associations:
                associations = [(a, 0.5) for a in req.associations]

            embedding = _state.embed(req.text)

            _state.store.add_memory(
                memory_id=memory_id,
                content=req.text,
                embedding=embedding,
                mtype=req.mtype,
                importance=req.importance,
                created_tick=req.created_tick if req.created_tick is not None else _state.current_tick,
                associations=associations,
                speaker=req.speaker or "",
            )
            ids.append(memory_id)

        return {"ids": ids, "count": len(ids), "tick": _state.current_tick}

    @app.post("/search")
    def search(req: SearchRequest):
        if not _state:
            raise HTTPException(503, "Server not initialized")

        query_embedding = _state.embed(req.query)
        params = _state.engine.get_params()

        results = _state.store.search(
            query_embedding=query_embedding,
            top_k=req.top_k,
            current_tick=_state.current_tick,
            activation_weight=params.get("activation_weight", 0.5),
        )

        return {"results": results}

    @app.post("/feedback")
    def feedback(req: FeedbackRequest):
        if not _state:
            raise HTTPException(503, "Server not initialized")
        if not req.items:
            raise HTTPException(422, "items must not be empty")

        tuples = [(item.memory_id, item.signal.value, item.strength) for item in req.items]
        applied = _state.store.adjust_scores(tuples, current_tick=_state.current_tick)
        return {"applied": applied, "total": len(tuples)}

    @app.post("/tick")
    def tick(req: TickRequest):
        if not _state:
            raise HTTPException(503, "Server not initialized")

        for _ in range(req.count):
            _state.engine.tick()
            _state.current_tick += 1
        _state.last_tick_time = time.time()

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

        return {
            "ticks_applied": ticks_due,
            "current_tick": _state.current_tick,
            "elapsed_seconds": round(elapsed, 1),
        }

    @app.delete("/forget/{memory_id}")
    def forget(memory_id: str):
        if not _state:
            raise HTTPException(503, "Server not initialized")

        node = _state.store.get_node(memory_id)
        if node is None:
            raise HTTPException(404, f"Memory {memory_id} not found")

        _state.store.delete_feedback_for(memory_id)
        _state.store.delete_memory(memory_id)

        return {"deleted": memory_id}

    @app.post("/reset")
    def reset():
        if not _state:
            raise HTTPException(503, "Server not initialized")

        cleared = _state.store.clear()
        _state.store.clear_feedback()
        _state.engine.reset()
        _state.current_tick = 0
        _state.last_tick_time = time.time()

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
    parser.add_argument("--persistence-dir", default=None,
                        help="(deprecated) Ignored, kept for backward compat")
    parser.add_argument("--cache-dir", default=None,
                        help="Path to embedding cache dir (default: auto-detect ./cache)")
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
    args = parser.parse_args()

    provider = create_embedding_provider(
        provider=args.embedding_provider,
        api_key=args.embedding_api_key,
        model=args.embedding_model,
    )

    # Auto-detect dimension from provider if not specified
    resolved_dim = args.embedding_dim or provider.dimension

    # Resolve db_path: explicit > cache_dir fallback > default
    resolved_db_path = args.db_path
    if resolved_db_path is None:
        if args.cache_dir:
            resolved_db_path = str(Path(args.cache_dir) / "memories.db")
        else:
            resolved_db_path = str(_DEFAULT_DB_DIR / "memories.db")

    app = create_app(
        embedding_provider=provider,
        cache_dir=args.cache_dir,
        tick_interval_seconds=args.tick_interval,
        experiment_dir=args.experiment_dir,
        db_path=resolved_db_path,
        embedding_dim=resolved_dim,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
