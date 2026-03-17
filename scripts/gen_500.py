"""Generate 500 memories in batches to avoid API response limits.

Strategy:
- Batch 0: generate hubs (10 high-impact memories)
- Batches 1-9: generate leaves referencing batch 0 hubs
- Cross-batch association via hub ID mapping
"""

import random
import time

import openai

from memory_decay.data_gen import SyntheticDataGenerator

gen = SyntheticDataGenerator()

# ── Phase 1: Generate hubs once ──
print("Phase 1: Generating hub memories...")
hub_dataset = gen.generate_dataset(
    num_memories=50,
    hub_ratio=0.4,
    ticks_range=(0, 200),
    seed=42,
)

# Extract hubs (high impact) and assign final IDs upfront
hub_memories = sorted(hub_dataset, key=lambda m: -m.get("impact", 0))[:15]
hub_id_map = {}
for j, m in enumerate(hub_memories):
    new_id = f"hub_{j+1:03d}"
    hub_id_map[m["id"]] = new_id
    m["id"] = new_id
print(f"  Created {len(hub_memories)} hubs: {[m['id'] for m in hub_memories[:3]]}...")

# Build a simple hub summary for the leaf prompt (content + entities)
hub_summary = [
    {"id": m["id"], "content": m["content"], "entities": m.get("entities", [])}
    for m in hub_memories
]

# ── Phase 2: Generate leaves in batches ──
all_leaves = []
batch_size = 50
num_batches = 10
global_counter = 0

for i in range(num_batches):
    print(f"Batch {i+1}/{num_batches}...")
    try:
        leaves = gen.generate_leaves_only(
            num_leaves=batch_size,
            hub_memories=hub_summary,
            max_associations=3,
            ticks_range=(0, 200),
            seed=100 + i,
        )
    except (openai.BadRequestError, openai.RateLimitError) as e:
        print(f"  API error: {e}. Retrying in 5s...")
        time.sleep(5)
        try:
            leaves = gen.generate_leaves_only(
                num_leaves=batch_size,
                hub_memories=hub_summary,
                max_associations=3,
                ticks_range=(0, 200),
                seed=200 + i,
            )
        except Exception as e2:
            print(f"  Failed after retry: {e2}. Skipping.")
            continue

    # Re-ID each leaf with globally unique ID
    for m in leaves:
        global_counter += 1
        new_id = f"mem_{global_counter:04d}"
        old_id = m["id"]
        m["id"] = new_id

        # Fix associations: map hub IDs
        new_assocs = []
        for assoc in m.get("associations", []):
            if isinstance(assoc, dict):
                # Check if it references a hub
                orig_id = assoc["id"]
                if orig_id in hub_id_map:
                    new_assocs.append({
                        "id": hub_id_map[orig_id],
                        "weight": assoc.get("weight", random.uniform(0.3, 0.9)),
                    })
            elif isinstance(assoc, str):
                if assoc in hub_id_map:
                    new_assocs.append({
                        "id": hub_id_map[assoc],
                        "weight": random.uniform(0.3, 0.9),
                    })
        m["associations"] = new_assocs

    all_leaves.extend(leaves)
    print(f"  Got {len(leaves)}, total leaves: {len(all_leaves)}")

# ── Phase 3: Merge and finalize ──
all_memories = hub_memories + all_leaves

# Add cross-associations between hubs
for i, hub in enumerate(hub_memories):
    others = [h["id"] for j, h in enumerate(hub_memories) if j != i]
    chosen = random.sample(others, min(3, len(others)))
    existing_ids = {a["id"] for a in hub.get("associations", [])}
    for cid in chosen:
        if cid not in existing_ids:
            hub.setdefault("associations", []).append({
                "id": cid, "weight": random.uniform(0.4, 0.9)
            })

all_memories.sort(key=lambda m: m.get("tick", 0))

total = len(all_memories)
facts = sum(1 for m in all_memories if m["type"] == "fact")
episodes = sum(1 for m in all_memories if m["type"] == "episode")
print(f"\nTotal: {total} (Facts: {facts}, Episodes: {episodes})")

gen.save_jsonl(all_memories, "data/memories_500.jsonl")

train, test = gen.split_test_train(all_memories, test_ratio=0.2, seed=42)
gen.save_jsonl(test, "data/test_500.jsonl")
print(f"Train: {len(train)}, Test: {len(test)}")
