"""Generate 500 memories in batches to avoid API response limits."""

import random
import time

import openai

from memory_decay.data_gen import SyntheticDataGenerator

gen = SyntheticDataGenerator()
all_memories = []

# Generate in batches of 50
batch_size = 50
total = 500
num_batches = total // batch_size  # 10 batches

hub_memories = None

for i in range(num_batches):
    print(f"Batch {i+1}/{num_batches}...")

    try:
        if i == 0:
            # First batch: generate full dataset (hubs + leaves)
            batch = gen.generate_dataset(
                num_memories=batch_size,
                hub_ratio=0.2,
                ticks_range=(0, 200),
                seed=42,
            )
            # Extract hub memories for reuse in later batches
            hub_memories = [m for m in batch if m.get("impact", 0) >= 0.7]
            if not hub_memories:
                # Fallback: take first 10 as hubs
                hub_memories = batch[:10]
        else:
            # Subsequent batches: only generate leaves referencing existing hubs
            num_leaves = batch_size
            batch = gen.generate_leaves_only(
                num_leaves=num_leaves,
                hub_memories=hub_memories,
                max_associations=4,
                ticks_range=(0, 200),
                seed=42 + i,
            )
    except (openai.BadRequestError, openai.RateLimitError) as e:
        print(f"  API error on batch {i+1}: {e}. Retrying in 5s...")
        time.sleep(5)
        try:
            if i == 0:
                batch = gen.generate_dataset(
                    num_memories=batch_size,
                    hub_ratio=0.2,
                    ticks_range=(0, 200),
                    seed=42,
                )
                hub_memories = [m for m in batch if m.get("impact", 0) >= 0.7]
                if not hub_memories:
                    hub_memories = batch[:10]
            else:
                batch = gen.generate_leaves_only(
                    num_leaves=batch_size,
                    hub_memories=hub_memories,
                    max_associations=4,
                    ticks_range=(0, 200),
                    seed=42 + i,
                )
        except Exception as e2:
            print(f"  Batch {i+1} failed after retry: {e2}. Skipping.")
            continue

    all_memories.extend(batch)
    print(f"  Generated {len(batch)}, total: {len(all_memories)}")

# De-duplicate by ID (in case LLM reuses IDs)
seen = set()
unique = []
for m in all_memories:
    if m["id"] not in seen:
        seen.add(m["id"])
        unique.append(m)

# Re-assign sequential IDs and fix associations
id_map = {}
for idx, m in enumerate(unique):
    old_id = m["id"]
    new_id = f"mem_{idx+1:04d}"
    id_map[old_id] = new_id
    m["id"] = new_id

# Fix association references
for m in unique:
    new_assocs = []
    for assoc in m.get("associations", []):
        if isinstance(assoc, dict):
            if assoc["id"] in id_map:
                new_assocs.append({"id": id_map[assoc["id"]], "weight": assoc.get("weight", 0.5)})
        elif isinstance(assoc, str) and assoc in id_map:
            new_assocs.append({"id": id_map[assoc], "weight": random.uniform(0.3, 0.9)})
    m["associations"] = new_assocs

unique.sort(key=lambda m: m["tick"])

print(f"\nTotal unique memories: {len(unique)}")
facts = sum(1 for m in unique if m["type"] == "fact")
episodes = sum(1 for m in unique if m["type"] == "episode")
print(f"Facts: {facts}, Episodes: {episodes}")

gen.save_jsonl(unique, "data/memories_500.jsonl")

train, test = gen.split_test_train(unique, test_ratio=0.2, seed=42)
gen.save_jsonl(test, "data/test_500.jsonl")
print(f"Train: {len(train)}, Test: {len(test)}")
