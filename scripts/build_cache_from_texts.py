"""Build embedding cache from a JSON array of texts.

Reads a JSON file containing an array of strings, embeds them using OpenAI
text-embedding-3-small (batch API), and writes embeddings.pkl.

Note: pickle is used intentionally here to match the existing embedding cache
format used by memory_decay.cache_builder (embeddings.pkl). The cache only
contains trusted text->numpy array mappings generated locally.

Usage:
    python scripts/build_cache_from_texts.py \
        --input ../memorybench/data/cache-texts/all.json \
        --output cache/ \
        --api-key sk-...
"""

from __future__ import annotations

import argparse
import json
import pickle  # noqa: S403 - trusted local cache, matches existing format
import time
from pathlib import Path

import numpy as np


def load_existing_cache(cache_path: Path) -> dict[str, np.ndarray]:
    if not cache_path.exists():
        return {}
    with open(cache_path, "rb") as f:
        data = pickle.load(f)  # noqa: S301 - trusted local cache
    return {str(k): np.array(v, dtype=np.float32) for k, v in data.items()}


def save_cache(cache_path: Path, embeddings: dict[str, np.ndarray]) -> None:
    tmp = cache_path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(embeddings, f)
    tmp.replace(cache_path)


def embed_batch_openai(client, texts: list[str], model: str, dimensions: int | None) -> list[np.ndarray]:
    params: dict = {"model": model, "input": texts}
    if dimensions:
        params["dimensions"] = dimensions
    response = client.embeddings.create(**params)
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [np.array(item.embedding, dtype=np.float32) for item in sorted_data]


def main():
    parser = argparse.ArgumentParser(description="Build embedding cache from text list")
    parser.add_argument("--input", required=True, help="JSON file with array of strings")
    parser.add_argument("--output", required=True, help="Output directory for embeddings.pkl")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (or OPENAI_API_KEY env)")
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--dimensions", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    args = parser.parse_args()

    import os
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Provide --api-key or set OPENAI_API_KEY")

    import openai
    client = openai.OpenAI(api_key=api_key)

    # Load texts
    texts = json.loads(Path(args.input).read_text(encoding="utf-8"))
    print(f"Loaded {len(texts)} texts from {args.input}")

    # Load existing cache
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "embeddings.pkl"
    embeddings = load_existing_cache(cache_path)
    print(f"Existing cache: {len(embeddings)} entries")

    # Filter pending
    pending = [t for t in texts if t not in embeddings]
    print(f"Pending: {len(pending)} texts to embed")

    if not pending:
        print("All texts already cached!")
        return

    # Embed in batches
    total_done = len(texts) - len(pending)
    dirty = 0
    t0 = time.time()

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]

        try:
            vecs = embed_batch_openai(client, batch, args.model, args.dimensions)
            for text, vec in zip(batch, vecs):
                embeddings[text] = vec
            dirty += len(batch)
            total_done += len(batch)
        except Exception as e:
            print(f"Error at batch {start}: {e}")
            if dirty > 0:
                save_cache(cache_path, embeddings)
                dirty = 0
            time.sleep(5)
            continue

        elapsed = time.time() - t0
        rate = total_done / max(elapsed, 0.01)
        remaining = len(texts) - total_done
        eta = remaining / max(rate, 0.01)
        print(
            f"  {total_done}/{len(texts)} ({100*total_done/len(texts):.1f}%) "
            f"- {rate:.0f} texts/s - ETA {eta:.0f}s"
        )

        if dirty >= args.checkpoint_every:
            save_cache(cache_path, embeddings)
            dirty = 0

    if dirty > 0:
        save_cache(cache_path, embeddings)

    manifest = {
        "text_count": len(texts),
        "embedded_count": len(embeddings),
        "model": args.model,
        "dimensions": args.dimensions,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    elapsed = time.time() - t0
    print(f"\nDone! {len(embeddings)} embeddings in {elapsed:.1f}s")
    print(f"Cache: {cache_path}")


if __name__ == "__main__":
    main()
