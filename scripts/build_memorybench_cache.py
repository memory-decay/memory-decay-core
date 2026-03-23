"""Build an embedding cache for raw MemoryBench benchmark transcripts.

This is intended for the MemoryBench <-> memory-decay bridge, which stores
conversation turns as prefixed transcript lines such as "[User] ...".
The existing auto-research cache contains converted memory nodes, which does
not align well enough with the raw benchmark transcript format.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Callable, Optional

import numpy as np


def load_questions(path: str) -> list[dict]:
    source = Path(path)

    if source.is_dir():
        return [
            json.loads(file.read_text(encoding="utf-8"))
            for file in sorted(source.glob("*.json"))
        ]

    data = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return [data]


def extract_texts(questions: list[dict]) -> list[str]:
    texts: set[str] = set()

    for item in questions:
        question = item.get("question")
        if question:
            texts.add(question)

        for session in item.get("haystack_sessions", []):
            for message in session:
                role = message.get("role", "user")
                prefix = "[User]" if role == "user" else "[Assistant]"
                content = message.get("content", "").strip()
                if content:
                    texts.add(f"{prefix} {content}")

    return sorted(texts)


def _load_embeddings(cache_path: Path) -> dict[str, np.ndarray]:
    if not cache_path.exists():
        return {}

    with open(cache_path, "rb") as f:
        embeddings = pickle.load(f)

    return {
        str(text): np.array(embedding, dtype=np.float32)
        for text, embedding in embeddings.items()
    }


def _persist_cache(
    out_dir: Path,
    embeddings: dict[str, np.ndarray],
    total_texts: int,
) -> None:
    embeddings_path = out_dir / "embeddings.pkl"
    manifest_path = out_dir / "manifest.json"

    tmp_embeddings_path = embeddings_path.with_suffix(".pkl.tmp")
    with open(tmp_embeddings_path, "wb") as f:
        pickle.dump(embeddings, f)
    tmp_embeddings_path.replace(embeddings_path)

    manifest = {
        "text_count": total_texts,
        "embedded_count": len(embeddings),
        "remaining_count": max(total_texts - len(embeddings), 0),
        "cache_type": "memorybench-transcript",
    }
    tmp_manifest_path = manifest_path.with_suffix(".json.tmp")
    tmp_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp_manifest_path.replace(manifest_path)


def build_embedding_cache(
    texts: list[str],
    output_dir: str,
    embedder: Optional[Callable[[str], np.ndarray]] = None,
    batch_embedder: Optional[Callable[[list[str]], list[np.ndarray]]] = None,
    batch_size: int = 100,
    checkpoint_every: int = 1000,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ordered_texts = list(dict.fromkeys(texts))
    total_texts = len(ordered_texts)
    embeddings = _load_embeddings(out / "embeddings.pkl")
    pending_texts = [text for text in ordered_texts if text not in embeddings]

    if embeddings:
        print(f"Loaded {len(embeddings)}/{total_texts} cached embeddings")

    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")

    if not pending_texts:
        _persist_cache(out, embeddings, total_texts)
        print(f"Embedded {total_texts}/{total_texts} texts")
        return

    completed_before_run = total_texts - len(pending_texts)
    dirty_count = 0

    if batch_embedder is not None:
        for start in range(0, len(pending_texts), batch_size):
            batch = pending_texts[start:start + batch_size]
            batch_embeddings = batch_embedder(batch)
            if len(batch_embeddings) != len(batch):
                raise ValueError("batch_embedder returned a mismatched number of embeddings")
            for text, emb in zip(batch, batch_embeddings):
                embeddings[text] = np.array(emb, dtype=np.float32)
            dirty_count += len(batch)
            completed = completed_before_run + min(start + batch_size, len(pending_texts))
            print(f"Embedded {completed}/{total_texts} texts")
            if dirty_count >= checkpoint_every:
                _persist_cache(out, embeddings, total_texts)
                dirty_count = 0
    else:
        if embedder is None:
            raise ValueError("embedder or batch_embedder is required")

        for index, text in enumerate(pending_texts, start=1):
            embeddings[text] = np.array(embedder(text), dtype=np.float32)
            dirty_count += 1
            completed = completed_before_run + index
            if index % 50 == 0 or index == len(pending_texts):
                print(f"Embedded {completed}/{total_texts} texts")
            if dirty_count >= checkpoint_every:
                _persist_cache(out, embeddings, total_texts)
                dirty_count = 0

    if dirty_count > 0 or len(embeddings) == total_texts:
        _persist_cache(out, embeddings, total_texts)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build raw transcript embedding cache for MemoryBench")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a benchmark question JSON file or a directory of split question JSON files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for embeddings.pkl",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "local", "gemini"],
        default="auto",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Texts per batch API call",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="Persist progress after this many newly embedded texts",
    )
    args = parser.parse_args()

    from memory_decay.graph import MemoryGraph

    questions = load_questions(args.input)
    texts = extract_texts(questions)
    print(f"Collected {len(texts)} unique texts")

    graph = MemoryGraph(embedding_backend=args.backend)
    build_embedding_cache(
        texts,
        args.output,
        embedder=graph._embed_text,
        batch_embedder=graph._gemini_embed_batch if args.backend in {"auto", "gemini"} else None,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"Cache written to {args.output}")


if __name__ == "__main__":
    main()
