"""Shared test fixtures and constants for memory-decay server tests."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from memory_decay.server import create_app

DIM = 8


@pytest.fixture
def embedder():
    """Deterministic fake embedder for testing."""
    return lambda t: np.random.RandomState(hash(t) % 2**31).randn(DIM).astype(np.float32)


@pytest.fixture
def client(embedder):
    """Create test client with fake embedder."""
    app = create_app(embedding_provider=None, _test_embedder=embedder)
    with TestClient(app) as c:
        yield c
