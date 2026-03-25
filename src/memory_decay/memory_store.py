"""SQLite + sqlite-vec backed memory store."""
from __future__ import annotations

import hashlib
import sqlite3
import struct
from typing import Optional

import numpy as np
import sqlite_vec


def _serialize_f32(vec: np.ndarray) -> bytes:
    """Serialize numpy float32 array to bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec.astype(np.float32))


def _deserialize_f32(data: bytes, dim: int) -> np.ndarray:
    """Deserialize bytes back to numpy float32 array."""
    return np.array(struct.unpack(f"{dim}f", data), dtype=np.float32)


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector. Pre-normalization makes L2 distance ≈ cosine distance."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


class MemoryStore:
    """SQLite-backed memory store with vector search via sqlite-vec.

    Drop-in replacement for MemoryGraph. Data lives on disk (or :memory:),
    no full-load needed, supports multi-user via user_id column.

    Embeddings are pre-normalized before storage so that sqlite-vec's L2
    distance approximates cosine distance:
        distance² = 2(1 - cos_sim)  =>  cos_sim = 1 - distance²/2
    """

    def __init__(self, db_path: str, embedding_dim: int = 3072):
        self._db_path = db_path
        self._embedding_dim = embedding_dim
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.enable_load_extension(True)
        sqlite_vec.load(self._db)
        self._db.enable_load_extension(False)
        self._init_schema()

    def _init_schema(self) -> None:
        # Create non-vector tables first
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id                   TEXT PRIMARY KEY,
                user_id              TEXT NOT NULL DEFAULT '',
                content              TEXT NOT NULL,
                mtype                TEXT DEFAULT 'episode',
                importance           REAL DEFAULT 0.7,
                speaker              TEXT DEFAULT '',
                created_tick         INTEGER DEFAULT 0,
                storage_score        REAL DEFAULT 1.0,
                retrieval_score      REAL DEFAULT 1.0,
                stability_score      REAL DEFAULT 0.0,
                last_activated_tick  INTEGER DEFAULT 0,
                last_reinforced_tick INTEGER DEFAULT 0,
                retrieval_count      INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS associations (
                source_id    TEXT,
                target_id    TEXT,
                weight       REAL DEFAULT 0.5,
                created_tick INTEGER DEFAULT 0,
                PRIMARY KEY (source_id, target_id)
            );

            CREATE TABLE IF NOT EXISTS metadata (
                key   TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash TEXT PRIMARY KEY,
                model     TEXT DEFAULT '',
                embedding BLOB NOT NULL
            );
        """)

        # Handle vec_memories with dimension change detection
        self._ensure_vec_table()
        self._db.commit()

    def _ensure_vec_table(self) -> None:
        """Create or recreate vec_memories if embedding dimension changed."""
        import sys

        stored_dim = self.get_metadata("embedding_dim")
        vec_exists = self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_memories'"
        ).fetchone() is not None

        needs_recreate = False
        if stored_dim and int(stored_dim) != self._embedding_dim:
            print(
                f"[memory-store] Embedding dimension changed: {stored_dim} → {self._embedding_dim}. "
                f"Recreating vec_memories and clearing embedding_cache.",
                file=sys.stderr,
            )
            needs_recreate = True
        elif not stored_dim and vec_exists:
            # Old DB without metadata — probe the existing dimension
            print(
                f"[memory-store] Existing vec_memories found without dimension metadata. "
                f"Recreating for {self._embedding_dim}-dim embeddings.",
                file=sys.stderr,
            )
            needs_recreate = True

        if needs_recreate:
            self._db.executescript(
                "DROP TABLE IF EXISTS vec_memories;"
                "DELETE FROM embedding_cache;"
            )

        self._db.executescript(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0("
            f"embedding float[{self._embedding_dim}]);"
        )
        self.set_metadata("embedding_dim", str(self._embedding_dim))

    def commit(self) -> None:
        """Explicitly commit the current transaction."""
        self._db.commit()

    def add_memory(
        self,
        memory_id: str,
        content: str,
        embedding: np.ndarray,
        *,
        user_id: str = "",
        mtype: str = "episode",
        importance: float = 0.7,
        speaker: str = "",
        created_tick: int = 0,
        associations: list[tuple[str, float]] | None = None,
        auto_commit: bool = True,
    ) -> None:
        # Insert into memories table first to get a rowid
        self._db.execute(
            """INSERT OR REPLACE INTO memories
               (id, user_id, content, mtype, importance, speaker, created_tick,
                last_activated_tick, last_reinforced_tick)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (memory_id, user_id, content, mtype, importance, speaker,
             created_tick, created_tick, created_tick),
        )
        # Get the rowid assigned by SQLite
        rowid = self._db.execute(
            "SELECT rowid FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()[0]

        # Pre-normalize embedding for cosine similarity via L2 distance
        normed = _normalize(embedding)

        # Insert into vec_memories with the same rowid
        # Use DELETE + INSERT because vec0 does not support OR REPLACE
        self._db.execute("DELETE FROM vec_memories WHERE rowid = ?", (rowid,))
        self._db.execute(
            "INSERT INTO vec_memories(rowid, embedding) VALUES (?, ?)",
            (rowid, _serialize_f32(normed)),
        )

        if associations:
            for target_id, weight in associations:
                self._db.execute(
                    "INSERT OR IGNORE INTO associations VALUES (?, ?, ?, ?)",
                    (memory_id, target_id, weight, created_tick),
                )
                self._db.execute(
                    "INSERT OR IGNORE INTO associations VALUES (?, ?, ?, ?)",
                    (target_id, memory_id, weight, created_tick),
                )
        if auto_commit:
            self._db.commit()

    def add_memories_batch(
        self,
        memories: list[dict],
    ) -> None:
        """Insert multiple memories in a single transaction.

        Each dict must have: memory_id, content, embedding.
        Optional: user_id, mtype, importance, speaker, created_tick, associations.
        """
        for mem in memories:
            self.add_memory(
                memory_id=mem["memory_id"],
                content=mem["content"],
                embedding=mem["embedding"],
                user_id=mem.get("user_id", ""),
                mtype=mem.get("mtype", "episode"),
                importance=mem.get("importance", 0.7),
                speaker=mem.get("speaker", ""),
                created_tick=mem.get("created_tick", 0),
                associations=mem.get("associations"),
                auto_commit=False,
            )
        self._db.commit()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        current_tick: int | None = None,
        activation_weight: float = 0.0,
        user_id: str | None = None,
    ) -> list[dict]:
        # Pre-normalize query embedding to match stored embeddings
        normed_query = _normalize(query_embedding)
        query_bytes = _serialize_f32(normed_query)

        # KNN search via sqlite-vec (returns L2 distance on normalized vectors)
        # sqlite-vec requires 'k = ?' constraint in WHERE clause (not LIMIT)
        fetch_k = top_k * 3  # fetch extra for filtering/reranking
        rows = self._db.execute(
            """SELECT v.rowid, v.distance, m.*
               FROM vec_memories v
               JOIN memories m ON m.rowid = v.rowid
               WHERE v.embedding MATCH ? AND k = ?
               ORDER BY v.distance""",
            (query_bytes, fetch_k),
        ).fetchall()

        results = []
        for row in rows:
            if current_tick is not None and row["created_tick"] > current_tick:
                continue
            if user_id is not None and row["user_id"] != user_id:
                continue

            # For pre-normalized vectors:
            # L2_distance² = 2(1 - cos_sim)
            # cos_sim = 1 - distance²/2
            distance = float(row["distance"])
            similarity = max(1.0 - (distance ** 2) / 2.0, 0.0)

            # Apply activation weight (boost by retrieval_score)
            if activation_weight > 0:
                retrieval_score = max(float(row["retrieval_score"]), 0.0)
                similarity *= retrieval_score ** activation_weight

            results.append({
                "id": row["id"],
                "text": row["content"],
                "score": round(similarity, 4),
                "storage_score": round(float(row["storage_score"]), 4),
                "retrieval_score": round(float(row["retrieval_score"]), 4),
                "category": row["mtype"],
                "created_tick": row["created_tick"],
                "speaker": row["speaker"] or "",
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_node(self, memory_id: str) -> dict | None:
        row = self._db.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_all_for_decay(self, current_tick: int) -> list[dict]:
        rows = self._db.execute(
            """SELECT id, retrieval_score, storage_score, stability_score,
                      importance, mtype
               FROM memories WHERE created_tick <= ?""",
            (current_tick,),
        ).fetchall()
        return [dict(r) for r in rows]

    def batch_update_scores(self, updates: list[tuple]) -> None:
        """Update scores in bulk. Each tuple: (retrieval, storage, stability, id)."""
        self._db.executemany(
            """UPDATE memories
               SET retrieval_score = ?, storage_score = ?, stability_score = ?
               WHERE id = ?""",
            updates,
        )
        self._db.commit()

    def set_retrieval_score(self, memory_id: str, score: float, *, auto_commit: bool = True) -> None:
        self._db.execute(
            "UPDATE memories SET retrieval_score = ? WHERE id = ?",
            (score, memory_id),
        )
        if auto_commit:
            self._db.commit()

    def set_storage_score(self, memory_id: str, score: float, *, auto_commit: bool = True) -> None:
        self._db.execute(
            "UPDATE memories SET storage_score = ? WHERE id = ?",
            (score, memory_id),
        )
        if auto_commit:
            self._db.commit()

    def set_activation(self, memory_id: str, score: float, *, auto_commit: bool = True) -> None:
        self._db.execute(
            "UPDATE memories SET retrieval_score = ? WHERE id = ?",
            (score, memory_id),
        )
        if auto_commit:
            self._db.commit()

    def reinforce(
        self,
        memory_id: str,
        retrieval_boost: float = 0.10,
        stability_gain: float = 0.05,
        stability_cap: float = 1.0,
        *,
        auto_commit: bool = True,
    ) -> None:
        """Boost a memory's retrieval and stability scores (retrieval consolidation).

        Called when a memory is successfully recalled, implementing the testing
        effect: successful retrieval strengthens the memory trace.
        """
        row = self._db.execute(
            "SELECT retrieval_score, storage_score, stability_score FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return

        retrieval = min(float(row[0]) + retrieval_boost, 1.0)
        storage = min(float(row[1]) + retrieval_boost * 0.25, 1.0)
        stability = float(row[2])
        saturation = max(1.0 - stability / max(stability_cap, 1e-9), 0.0)
        stability = min(stability + stability_gain * saturation, stability_cap)

        self._db.execute(
            """UPDATE memories
               SET retrieval_score = ?, storage_score = ?, stability_score = ?,
                   retrieval_count = retrieval_count + 1
               WHERE id = ?""",
            (retrieval, storage, stability, memory_id),
        )
        if auto_commit:
            self._db.commit()

    def delete_memory(self, memory_id: str) -> None:
        """Delete a single memory by ID."""
        rowid_row = self._db.execute(
            "SELECT rowid FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if rowid_row is not None:
            self._db.execute("DELETE FROM vec_memories WHERE rowid = ?", (rowid_row[0],))
        self._db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._db.execute("DELETE FROM associations WHERE source_id = ? OR target_id = ?",
                        (memory_id, memory_id))
        self._db.commit()

    def clear(self, user_id: str | None = None) -> int:
        if user_id:
            # Get rowids to delete from vec_memories
            rowids = [r[0] for r in self._db.execute(
                "SELECT rowid FROM memories WHERE user_id = ?", (user_id,)
            ).fetchall()]
            count = self._db.execute(
                "DELETE FROM memories WHERE user_id = ?", (user_id,)
            ).rowcount
            for rid in rowids:
                self._db.execute("DELETE FROM vec_memories WHERE rowid = ?", (rid,))
        else:
            count = self._db.execute("DELETE FROM memories").rowcount
            self._db.execute("DELETE FROM vec_memories")
            self._db.execute("DELETE FROM associations")
        self._db.commit()
        return count

    @property
    def num_memories(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def get_metadata(self, key: str, default: str = "") -> str:
        row = self._db.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else default

    def set_metadata(self, key: str, value: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._db.commit()

    # --- Embedding cache ---

    def cache_embedding(self, text: str, embedding: np.ndarray, model: str = "", *, auto_commit: bool = True) -> None:
        """Cache an embedding for a text string."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        self._db.execute(
            "INSERT OR REPLACE INTO embedding_cache (text_hash, model, embedding) VALUES (?, ?, ?)",
            (text_hash, model, _serialize_f32(embedding)),
        )
        if auto_commit:
            self._db.commit()

    def get_cached_embedding(self, text: str, model: str = "") -> np.ndarray | None:
        """Retrieve a cached embedding, or None if not found."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        row = self._db.execute(
            "SELECT embedding FROM embedding_cache WHERE text_hash = ?",
            (text_hash,),
        ).fetchone()
        if row is None:
            return None
        return _deserialize_f32(row[0], self._embedding_dim)

    def close(self) -> None:
        self._db.close()
