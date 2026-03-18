"""Enrich memories_500.jsonl with entity-based associations.

Strategy (mirrors memories_50.jsonl patterns):
- Hub memories (high impact >= 0.7): get 1-2 cross-hub associations
- Leaf memories (low impact): get 1-3 associations to hubs sharing entities
- All associations are based on shared entities (deterministic)
- Weights are seeded random in [0.3, 1.0] range
- Max 4 associations per memory to avoid over-connection
"""

import json
import random
from pathlib import Path
from collections import defaultdict

SEED = 42
MAX_ASSOC = 4
HUB_IMPACT_THRESHOLD = 0.6


def load_jsonl(path: str) -> list[dict]:
    mems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                mems.append(json.loads(line))
    return mems


def save_jsonl(mems: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for m in mems:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def build_entity_index(mems: list[dict]) -> dict[str, list[str]]:
    """Map entity -> list of memory IDs containing it."""
    idx: dict[str, list[str]] = defaultdict(list)
    for m in mems:
        for ent in m.get("entities", []):
            idx[ent].append(m["id"])
    return idx


def find_shared_entity_peers(
    mem: dict, entity_index: dict[str, list[str]]
) -> list[str]:
    """Find all memory IDs sharing at least one entity with mem."""
    peers = set()
    for ent in mem.get("entities", []):
        for mid in entity_index[ent]:
            if mid != mem["id"]:
                peers.add(mid)
    return sorted(peers)


def enrich(mems: list[dict], rng: random.Random) -> list[dict]:
    entity_index = build_entity_index(mems)
    id_to_mem = {m["id"]: m for m in mems}

    # Classify hubs vs leaves
    hubs = {m["id"] for m in mems if m.get("impact", 0) >= HUB_IMPACT_THRESHOLD}

    for m in mems:
        peers = find_shared_entity_peers(m, entity_index)
        if not peers:
            continue

        is_hub = m["id"] in hubs

        if is_hub:
            # Hubs connect to other hubs first, then leaves
            hub_peers = [p for p in peers if p in hubs]
            leaf_peers = [p for p in peers if p not in hubs]
            # 1-2 hub connections + 0-1 leaf connections
            n_hub = min(len(hub_peers), rng.randint(1, 2))
            n_leaf = min(len(leaf_peers), max(0, MAX_ASSOC - n_hub))
            chosen = rng.sample(hub_peers, n_hub) if hub_peers else []
            if leaf_peers and n_leaf > 0:
                chosen += rng.sample(leaf_peers, min(n_leaf, len(leaf_peers)))
        else:
            # Leaves prefer hub connections
            hub_peers = [p for p in peers if p in hubs]
            other_peers = [p for p in peers if p not in hubs]
            n_hub = min(len(hub_peers), rng.randint(1, 2))
            n_other = min(len(other_peers), rng.randint(0, 1))
            chosen = rng.sample(hub_peers, n_hub) if hub_peers else []
            if other_peers and n_other > 0:
                chosen += rng.sample(other_peers, min(n_other, len(other_peers)))

            # If no hub peers, connect to any peer
            if not chosen and peers:
                chosen = rng.sample(peers, min(len(peers), rng.randint(1, 2)))

        # Cap at MAX_ASSOC
        chosen = chosen[:MAX_ASSOC]

        m["associations"] = [
            {"id": pid, "weight": round(rng.uniform(0.3, 1.0), 6)}
            for pid in chosen
        ]

    return mems


def main():
    rng = random.Random(SEED)
    src = Path("data/memories_500.jsonl")
    backup = Path("data/memories_500.jsonl.bak")

    mems = load_jsonl(str(src))
    print(f"Loaded {len(mems)} memories")

    # Count current associations
    before = sum(1 for m in mems if m.get("associations"))
    print(f"Before: {before} memories with associations")

    # Backup
    save_jsonl(mems, str(backup))
    print(f"Backup saved to {backup}")

    # Enrich
    mems = enrich(mems, rng)

    after = sum(1 for m in mems if m.get("associations"))
    total_assoc = sum(len(m.get("associations", [])) for m in mems)
    print(f"After: {after} memories with associations ({total_assoc} total links)")

    # Stats
    hub_count = sum(1 for m in mems if m.get("impact", 0) >= HUB_IMPACT_THRESHOLD)
    print(f"Hubs (impact >= {HUB_IMPACT_THRESHOLD}): {hub_count}")

    save_jsonl(mems, str(src))
    print(f"Saved enriched dataset to {src}")


if __name__ == "__main__":
    main()
