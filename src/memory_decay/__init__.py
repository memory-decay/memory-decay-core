"""Human-memory-inspired memory decay system.

Research code for a graph-based memory model with activation, stability,
reinforcement-aware reactivation, and LLM-driven parameter optimization.
"""

from .graph import MemoryGraph
from .decay import DecayEngine
from .evaluator import Evaluator
from .data_gen import SyntheticDataGenerator
from .auto_improver import AutoImprover

__all__ = [
    "MemoryGraph",
    "DecayEngine",
    "Evaluator",
    "MemoryStore",
    "EmbeddingProvider",
    "create_embedding_provider",
    "SyntheticDataGenerator",
    "AutoImprover",
]


def __getattr__(name: str):
    if name == "MemoryStore":
        from .memory_store import MemoryStore
        return MemoryStore
    if name == "EmbeddingProvider":
        from .embedding_provider import EmbeddingProvider
        return EmbeddingProvider
    if name == "create_embedding_provider":
        from .embedding_provider import create_embedding_provider
        return create_embedding_provider
    raise AttributeError(f"module 'memory_decay' has no attribute {name!r}")
