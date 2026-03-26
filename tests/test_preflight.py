"""Tests for preflight checks and clear error messages."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from memory_decay.memory_store import MemoryStore


class TestSqliteExtensionCheck:
    def test_missing_enable_load_extension_raises_runtime_error(self):
        """MemoryStore must raise RuntimeError with clear message when
        enable_load_extension is not available."""
        mock_conn = MagicMock(spec=[])  # empty spec = no attributes
        mock_conn.execute = MagicMock()
        mock_conn.row_factory = None

        with patch("sqlite3.connect", return_value=mock_conn):
            with pytest.raises(RuntimeError, match="SQLite loadable extensions"):
                MemoryStore(":memory:", embedding_dim=8)

    def test_error_message_mentions_python_org_installer(self):
        """Error message should guide macOS users away from python.org installer."""
        mock_conn = MagicMock(spec=[])
        mock_conn.execute = MagicMock()
        mock_conn.row_factory = None

        with patch("sqlite3.connect", return_value=mock_conn):
            with pytest.raises(RuntimeError, match="python.org"):
                MemoryStore(":memory:", embedding_dim=8)

    def test_error_message_suggests_uv(self):
        """Error message should recommend uv as a fix."""
        mock_conn = MagicMock(spec=[])
        mock_conn.execute = MagicMock()
        mock_conn.row_factory = None

        with patch("sqlite3.connect", return_value=mock_conn):
            with pytest.raises(RuntimeError, match="uv venv"):
                MemoryStore(":memory:", embedding_dim=8)

    def test_normal_init_works(self):
        """MemoryStore should still work normally when extensions are supported."""
        store = MemoryStore(":memory:", embedding_dim=8)
        assert store.num_memories == 0
        store.close()


class TestLocalEmbeddingProviderCheck:
    def test_import_failure_raises_runtime_error(self):
        """LocalEmbeddingProvider should raise RuntimeError with guidance
        when sentence_transformers import fails."""
        from memory_decay.embedding_provider import LocalEmbeddingProvider

        provider = LocalEmbeddingProvider()

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(RuntimeError, match="sentence-transformers"):
                provider.embed("test")

    def test_error_mentions_python_313_regression(self):
        """Error message should mention CPython 3.13.8 AST regression."""
        from memory_decay.embedding_provider import LocalEmbeddingProvider

        provider = LocalEmbeddingProvider()

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(RuntimeError, match="3.13"):
                provider.embed("test")


class TestServerPreflightChecks:
    def test_sqlite_check_fails_gracefully(self):
        """_preflight_checks should sys.exit(1) when sqlite extensions missing."""
        from memory_decay.server import _preflight_checks

        mock_conn = MagicMock(spec=[])  # no enable_load_extension
        mock_conn.close = MagicMock()

        with patch("sqlite3.connect", return_value=mock_conn):
            with pytest.raises(SystemExit) as exc_info:
                _preflight_checks("local")
            assert exc_info.value.code == 1

    def test_torch_import_check_fails_gracefully(self):
        """_preflight_checks should sys.exit(1) when torch import fails."""
        from memory_decay.server import _preflight_checks

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return original_import(name, *args, **kwargs)

        import builtins
        original_import = builtins.__import__

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with pytest.raises(SystemExit) as exc_info:
                _preflight_checks("local")
            assert exc_info.value.code == 1

    def test_preflight_passes_for_nonlocal_provider(self):
        """_preflight_checks should not check torch for non-local providers."""
        from memory_decay.server import _preflight_checks

        # Should not raise for gemini/openai even if torch is missing
        _preflight_checks("gemini")
        _preflight_checks("openai")
