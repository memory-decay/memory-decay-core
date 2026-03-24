"""Pickle-based persistence for MemoryGraph state."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

from .graph import MemoryGraph


class MemoryPersistence:
    """Save and load MemoryGraph state to/from disk."""

    def __init__(self, directory: str | Path):
        self._dir = Path(directory)

    @property
    def graph_path(self) -> Path:
        return self._dir / "graph.pkl"

    @property
    def meta_path(self) -> Path:
        return self._dir / "meta.json"

    def save(self, graph: MemoryGraph, current_tick: int = 0) -> None:
        """Persist graph to disk."""
        self._dir.mkdir(parents=True, exist_ok=True)

        graph_data = {
            "nodes": dict(graph._graph.nodes(data=True)),
            "edges": list(graph._graph.edges(data=True)),
        }
        with open(self.graph_path, "wb") as f:
            pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        meta = {
            "current_tick": current_tick,
            "num_memories": graph.num_memories,
            "saved_at": time.time(),
            "saved_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self.meta_path.write_text(json.dumps(meta, indent=2))

    def load(self, graph: MemoryGraph) -> dict | None:
        """Load persisted state into graph. Returns metadata or None."""
        if not self.graph_path.exists():
            return None

        with open(self.graph_path, "rb") as f:
            graph_data = pickle.load(f)

        graph._graph.clear()
        for node_id, attrs in graph_data["nodes"].items():
            graph._graph.add_node(node_id, **attrs)
        for src, dst, attrs in graph_data["edges"]:
            graph._graph.add_edge(src, dst, **attrs)

        # Reset embedding matrix cache
        graph._emb_matrix = None
        graph._emb_node_count = 0

        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text())
        return {"current_tick": 0, "saved_at": 0}
