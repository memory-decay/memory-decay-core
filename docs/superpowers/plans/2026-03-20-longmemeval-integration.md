# LongMemEval Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the synthetic `memories_500.jsonl` dataset with the LongMemEval benchmark (ICLR 2025) and switch embeddings to `gemini-embedding-001`, enabling decay function research on a published, peer-reviewed dataset.

**Architecture:** Offline converter script transforms LongMemEval oracle JSON into existing JSONL schema. No changes to evaluator, runner, or decay engine. Cache rebuilt with Gemini embeddings for English text support.

**Tech Stack:** Python, `google-genai` (Gemini embeddings), existing `memory_decay` pipeline

**Spec:** `docs/superpowers/specs/2026-03-20-longmemeval-integration-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `scripts/convert_longmemeval.py` | Convert LongMemEval oracle JSON → `data/longmemeval.jsonl` |
| Create | `tests/test_convert_longmemeval.py` | Tests for converter |
| Create | `data/longmemeval_oracle.json` | Downloaded source dataset |
| Create | `data/longmemeval.jsonl` | Converted output (generated, not committed) |
| Modify | `program.md` | Update preflight dataset path |
| Preserve | `data/memories_500.jsonl` | Keep old dataset as fallback |
| Rebuild | `cache/` | New Gemini embeddings for English content |

---

## Task 1: Download LongMemEval Dataset

**Files:**
- Create: `data/longmemeval_oracle.json`

- [ ] **Step 1: Download the oracle dataset from HuggingFace**

```bash
curl -sL "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json" \
  -o data/longmemeval_oracle.json
```

- [ ] **Step 2: Verify download**

```bash
python3 -c "
import json
with open('data/longmemeval_oracle.json') as f:
    data = json.load(f)
print(f'Questions: {len(data)}')
assert len(data) == 500, f'Expected 500, got {len(data)}'
print('OK')
"
```

Expected: `Questions: 500` then `OK`

- [ ] **Step 3: Commit**

```bash
git add data/longmemeval_oracle.json
git commit -m "data: add LongMemEval oracle dataset (ICLR 2025, 500 questions)"
```

---

## Task 2: Write Converter Tests

**Files:**
- Create: `tests/test_convert_longmemeval.py`

- [ ] **Step 1: Write test fixtures and core tests**

Create `tests/test_convert_longmemeval.py`:

```python
"""Tests for LongMemEval → memory-decay JSONL converter."""

import json
import pytest

# We'll import after implementation exists
# from scripts.convert_longmemeval import convert, parse_date, map_type

# Minimal LongMemEval fixture matching real schema
FIXTURE = [
    {
        "question_id": "test_q001",
        "question_type": "single-session-user",
        "question": "What degree did I graduate with?",
        "answer": "Business Administration",
        "question_date": "2023/04/10 (Mon) 23:07",
        "haystack_dates": ["2023/04/05 (Wed) 14:00", "2023/04/10 (Mon) 17:50"],
        "haystack_session_ids": ["sess_001", "sess_002"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I enjoy hiking on weekends.", "has_answer": False},
                {"role": "assistant", "content": "That sounds fun!", "has_answer": False},
            ],
            [
                {"role": "user", "content": "I graduated with a degree in Business Administration.", "has_answer": True},
                {"role": "assistant", "content": "Great field of study!", "has_answer": False},
                {"role": "user", "content": "Thanks, I really enjoyed my time at university.", "has_answer": False},
            ],
        ],
    },
    {
        "question_id": "test_q002",
        "question_type": "temporal-reasoning",
        "question": "What was the first thing I mentioned about my trip?",
        "answer": "Booking flights",
        "question_date": "2023/04/12 (Wed) 10:00",
        "haystack_dates": ["2023/04/08 (Sat) 09:00"],
        "haystack_session_ids": ["sess_003"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I just booked flights for my vacation.", "has_answer": True},
                {"role": "assistant", "content": "Where are you going?", "has_answer": False},
                {"role": "user", "content": "Planning to visit Japan next month.", "has_answer": False},
            ],
        ],
    },
]


def test_parse_date():
    from scripts.convert_longmemeval import parse_date

    dt = parse_date("2023/04/10 (Mon) 23:07")
    assert dt.year == 2023
    assert dt.month == 4
    assert dt.day == 10
    assert dt.hour == 23
    assert dt.minute == 7


def test_map_type():
    from scripts.convert_longmemeval import map_type

    assert map_type("single-session-user") == "fact"
    assert map_type("single-session-assistant") == "fact"
    assert map_type("single-session-preference") == "fact"
    assert map_type("knowledge-update") == "fact"
    assert map_type("temporal-reasoning") == "episode"
    assert map_type("multi-session") == "episode"


def test_convert_produces_valid_jsonl():
    from scripts.convert_longmemeval import convert

    memories = convert(FIXTURE)

    # Only user messages become memory nodes
    # q001: 3 user msgs (sess_001 has 1 user, sess_002 has 2 user)
    # q002: 2 user msgs (sess_003 has 2 user)
    # Total: 5 user messages
    assert len(memories) == 5

    # Check required fields on every node
    for mem in memories:
        assert "id" in mem
        assert "type" in mem and mem["type"] in ("fact", "episode")
        assert "content" in mem and len(mem["content"]) > 0
        assert "tick" in mem and 0 <= mem["tick"] <= 200
        assert "impact" in mem and 0.0 <= mem["impact"] <= 1.0
        assert "associations" in mem


def test_recall_query_assigned_to_has_answer_message():
    from scripts.convert_longmemeval import convert

    memories = convert(FIXTURE)

    # Find the memory with recall_query for q001
    recall_mems = [m for m in memories if m.get("recall_query")]
    assert len(recall_mems) == 2  # one per question

    q001_recall = [m for m in recall_mems if "Business Administration" in m["content"]]
    assert len(q001_recall) == 1
    assert q001_recall[0]["recall_query"] == "What degree did I graduate with?"
    assert q001_recall[0]["recall_answer"] == "Business Administration"


def test_type_mapping_from_question_type():
    from scripts.convert_longmemeval import convert

    memories = convert(FIXTURE)

    # q001 is single-session-user → fact
    q001_mems = [m for m in memories if m["id"].startswith("lme_test_q001")]
    for m in q001_mems:
        assert m["type"] == "fact"

    # q002 is temporal-reasoning → episode
    q002_mems = [m for m in memories if m["id"].startswith("lme_test_q002")]
    for m in q002_mems:
        assert m["type"] == "episode"


def test_impact_based_on_has_answer():
    from scripts.convert_longmemeval import convert

    memories = convert(FIXTURE)

    for mem in memories:
        if mem.get("recall_query"):
            # has_answer=true messages get high impact
            assert mem["impact"] == 0.8
        else:
            # non-answer messages get low impact
            assert mem["impact"] == 0.3


def test_tick_range():
    from scripts.convert_longmemeval import convert

    memories = convert(FIXTURE)
    ticks = [m["tick"] for m in memories]
    assert min(ticks) == 0
    assert max(ticks) <= 200


def test_assistant_messages_excluded():
    from scripts.convert_longmemeval import convert

    memories = convert(FIXTURE)

    for mem in memories:
        assert "That sounds fun" not in mem["content"]
        assert "Great field of study" not in mem["content"]
        assert "Where are you going" not in mem["content"]


def test_deduplication_across_questions():
    """Same session appearing in two questions should not produce duplicate nodes."""
    from scripts.convert_longmemeval import convert

    # Create two questions sharing the same session
    shared_session = [
        {"role": "user", "content": "I live in Seoul.", "has_answer": True},
        {"role": "assistant", "content": "Nice city!", "has_answer": False},
    ]
    data = [
        {
            "question_id": "dup_q1",
            "question_type": "single-session-user",
            "question": "Where do I live?",
            "answer": "Seoul",
            "question_date": "2023/05/01 (Mon) 10:00",
            "haystack_dates": ["2023/04/20 (Thu) 10:00"],
            "haystack_session_ids": ["shared_sess"],
            "haystack_sessions": [shared_session],
        },
        {
            "question_id": "dup_q2",
            "question_type": "single-session-user",
            "question": "What city am I in?",
            "answer": "Seoul",
            "question_date": "2023/05/02 (Tue) 10:00",
            "haystack_dates": ["2023/04/20 (Thu) 10:00"],
            "haystack_session_ids": ["shared_sess"],
            "haystack_sessions": [shared_session],
        },
    ]

    memories = convert(data)

    # "I live in Seoul." should appear only once
    seoul_mems = [m for m in memories if "Seoul" in m["content"]]
    assert len(seoul_mems) == 1

    # But it should have a recall_query (from first question)
    assert seoul_mems[0]["recall_query"] == "Where do I live?"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src:. uv run pytest tests/test_convert_longmemeval.py -v
```

Expected: All tests FAIL with `ModuleNotFoundError: No module named 'scripts.convert_longmemeval'`

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_convert_longmemeval.py
git commit -m "test: add converter tests for LongMemEval integration"
```

---

## Task 3: Implement Converter Script

**Files:**
- Create: `scripts/__init__.py` (if missing)
- Create: `scripts/convert_longmemeval.py`

- [ ] **Step 1: Create `scripts/__init__.py` if it doesn't exist**

```bash
touch scripts/__init__.py
```

- [ ] **Step 2: Write the converter**

Create `scripts/convert_longmemeval.py`:

```python
"""Convert LongMemEval oracle JSON to memory-decay JSONL format.

Input:  data/longmemeval_oracle.json
Output: data/longmemeval.jsonl

Each user message becomes a memory node. Assistant messages are excluded.
Deduplicates across questions by (session_id, message_index).
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path


def parse_date(date_str: str) -> datetime:
    """Parse LongMemEval date format: '2023/04/10 (Mon) 23:07'."""
    # Strip day-of-week in parentheses
    cleaned = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", date_str).strip()
    return datetime.strptime(cleaned, "%Y/%m/%d %H:%M")


TYPE_MAP = {
    "single-session-user": "fact",
    "single-session-assistant": "fact",
    "single-session-preference": "fact",
    "knowledge-update": "fact",
    "temporal-reasoning": "episode",
    "multi-session": "episode",
}


def map_type(question_type: str) -> str:
    """Map LongMemEval question type to memory type."""
    return TYPE_MAP.get(question_type, "fact")


def convert(data: list[dict]) -> list[dict]:
    """Convert LongMemEval questions to memory-decay JSONL records.

    Returns a list of memory node dicts matching the memories_500.jsonl schema.
    """
    # Pass 1: collect all dates for tick normalization
    all_dates: list[datetime] = []
    for q in data:
        for date_str in q.get("haystack_dates", []):
            all_dates.append(parse_date(date_str))

    if not all_dates:
        return []

    min_date = min(all_dates)
    max_date = max(all_dates)
    date_range = (max_date - min_date).total_seconds()
    if date_range == 0:
        date_range = 1.0  # avoid division by zero

    def date_to_tick(dt: datetime) -> int:
        elapsed = (dt - min_date).total_seconds()
        return int(elapsed / date_range * 200)

    # Pass 2: deduplicate and build memory nodes
    # Key: (session_id, user_msg_index_within_session) → memory dict
    seen: dict[tuple[str, int], dict] = {}
    # Track which memory IDs are associated per question
    question_associations: dict[str, list[str]] = {}

    for q in data:
        qid = q["question_id"]
        qtype = map_type(q["question_type"])
        session_ids = q.get("haystack_session_ids", [])
        sessions = q.get("haystack_sessions", [])
        dates = q.get("haystack_dates", [])

        answer_mem_ids: list[str] = []

        for s_idx, session in enumerate(sessions):
            sess_id = session_ids[s_idx] if s_idx < len(session_ids) else f"sess_{s_idx}"
            sess_date = parse_date(dates[s_idx]) if s_idx < len(dates) else min_date
            tick = date_to_tick(sess_date)

            user_msg_idx = 0
            for msg in session:
                if msg["role"] != "user":
                    continue

                dedup_key = (sess_id, user_msg_idx)
                mem_id = f"lme_{qid}_s{s_idx}_m{user_msg_idx}"
                has_answer = msg.get("has_answer", False)

                if dedup_key not in seen:
                    mem = {
                        "id": mem_id,
                        "type": qtype,
                        "content": msg["content"],
                        "entities": [],
                        "tick": tick,
                        "impact": 0.8 if has_answer else 0.3,
                        "associations": [],
                    }

                    if has_answer:
                        mem["recall_query"] = q["question"]
                        mem["recall_answer"] = q["answer"]
                        answer_mem_ids.append(mem_id)

                    seen[dedup_key] = mem
                else:
                    # Already seen this message from another question
                    existing = seen[dedup_key]
                    if has_answer and "recall_query" not in existing:
                        existing["recall_query"] = q["question"]
                        existing["recall_answer"] = q["answer"]
                        existing["impact"] = 0.8
                    if has_answer:
                        answer_mem_ids.append(existing["id"])

                user_msg_idx += 1

        question_associations[qid] = answer_mem_ids

    # Pass 3: build associations between answer-bearing messages of the same question
    memories = list(seen.values())
    mem_by_id = {m["id"]: m for m in memories}

    for qid, answer_ids in question_associations.items():
        if len(answer_ids) < 2:
            continue
        for i, aid in enumerate(answer_ids):
            if aid not in mem_by_id:
                continue
            for j, other_id in enumerate(answer_ids):
                if i == j or other_id not in mem_by_id:
                    continue
                existing_assoc_ids = {a["id"] for a in mem_by_id[aid]["associations"]}
                if other_id not in existing_assoc_ids:
                    mem_by_id[aid]["associations"].append(
                        {"id": other_id, "weight": 0.7}
                    )

    # Sort by tick for deterministic output
    memories.sort(key=lambda m: (m["tick"], m["id"]))

    return memories


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert LongMemEval oracle JSON to memory-decay JSONL"
    )
    parser.add_argument(
        "--input",
        default="data/longmemeval_oracle.json",
        help="Path to LongMemEval oracle JSON",
    )
    parser.add_argument(
        "--output",
        default="data/longmemeval.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    memories = convert(data)

    with open(args.output, "w", encoding="utf-8") as f:
        for mem in memories:
            f.write(json.dumps(mem, ensure_ascii=False) + "\n")

    # Stats
    with_query = sum(1 for m in memories if "recall_query" in m)
    facts = sum(1 for m in memories if m["type"] == "fact")
    episodes = sum(1 for m in memories if m["type"] == "episode")
    print(f"Converted {len(memories)} memories ({facts} facts, {episodes} episodes)")
    print(f"Recall queries: {with_query}")
    print(f"Tick range: {min(m['tick'] for m in memories)}-{max(m['tick'] for m in memories)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests to verify they pass**

```bash
PYTHONPATH=src:. uv run pytest tests/test_convert_longmemeval.py -v
```

Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/__init__.py scripts/convert_longmemeval.py
git commit -m "feat: add LongMemEval to memory-decay JSONL converter"
```

---

## Task 4: Convert Dataset and Rebuild Cache

**Files:**
- Create: `data/longmemeval.jsonl` (generated)
- Rebuild: `cache/` (Gemini embeddings)

- [ ] **Step 1: Run the converter on the real dataset**

```bash
PYTHONPATH=src:. uv run python scripts/convert_longmemeval.py \
  --input data/longmemeval_oracle.json \
  --output data/longmemeval.jsonl
```

Expected output like:
```
Converted NNNN memories (NNN facts, NNN episodes)
Recall queries: ~500
Tick range: 0-200
```

- [ ] **Step 2: Validate output schema**

```bash
python3 -c "
import json
with open('data/longmemeval.jsonl') as f:
    memories = [json.loads(line) for line in f if line.strip()]
print(f'Total: {len(memories)}')
required = {'id', 'type', 'content', 'tick', 'impact', 'associations'}
for m in memories:
    missing = required - set(m.keys())
    assert not missing, f'{m[\"id\"]} missing: {missing}'
with_query = [m for m in memories if 'recall_query' in m]
print(f'With recall_query: {len(with_query)}')
print('Schema OK')
"
```

Expected: `Schema OK`

- [ ] **Step 3: Backup current cache**

```bash
# cache_backup_local already exists from earlier, verify
ls cache_backup_local/embeddings.pkl && echo "Backup exists" || \
  cp -r cache cache_backup_local
```

- [ ] **Step 4: Rebuild cache with Gemini embeddings**

```bash
PYTHONPATH=src uv run python -m memory_decay.cache_builder \
  --dataset data/longmemeval.jsonl \
  --output cache \
  --backend gemini
```

Expected: `Cached NNNN embeddings` (should be ~4000-5000 texts: contents + recall queries)

Note: This calls the Gemini API for each unique text. May take a few minutes.

- [ ] **Step 5: Verify cache**

```bash
python3 -c "
import pickle, json
with open('cache/embeddings.pkl', 'rb') as f:
    emb = pickle.load(f)
with open('cache/dataset.json') as f:
    ds = json.load(f)
print(f'Embeddings: {len(emb)}')
print(f'Dataset entries: {len(ds)}')
print(f'Embedding dim: {list(emb.values())[0].shape}')
# Verify test_queries.json exists
with open('cache/test_queries.json') as f:
    tq = json.load(f)
print(f'Test queries: {len(tq)}')
print('Cache OK')
"
```

- [ ] **Step 6: Commit converted dataset (not cache)**

```bash
git add data/longmemeval.jsonl
git commit -m "data: add converted LongMemEval dataset (JSONL format)"
```

---

## Task 5: Reset Experiment Baseline

**Files:**
- Archive: `experiments/history.jsonl`
- Reset: `experiments/best/`

- [ ] **Step 1: Archive old experiment history**

```bash
mkdir -p experiments/archive_memories500
cp experiments/history.jsonl experiments/archive_memories500/ 2>/dev/null || true
cp -r experiments/best experiments/archive_memories500/ 2>/dev/null || true
```

- [ ] **Step 2: Clear history for fresh start**

```bash
> experiments/history.jsonl
```

- [ ] **Step 3: Create baseline experiment with default exponential decay**

Use the same `decay_fn.py` as `experiments/exp_0000/`:

```bash
mkdir -p experiments/exp_lme_0000
```

Write `experiments/exp_lme_0000/decay_fn.py` — copy from `experiments/exp_0000/decay_fn.py`:

```bash
cp experiments/exp_0000/decay_fn.py experiments/exp_lme_0000/decay_fn.py
```

Write `experiments/exp_lme_0000/params.json`:

```bash
cat > experiments/exp_lme_0000/params.json << 'EOF'
{
  "lambda_fact": 0.02,
  "lambda_episode": 0.035,
  "stability_weight": 0.8,
  "stability_decay": 0.01,
  "reinforcement_gain_direct": 0.2,
  "reinforcement_gain_assoc": 0.05,
  "stability_cap": 1.0,
  "activation_weight": 0.5,
  "assoc_boost": 0.0
}
EOF
```

Write `experiments/exp_lme_0000/hypothesis.txt`:

```bash
echo "Baseline: default exponential decay on LongMemEval dataset with Gemini embeddings. Establishes reference scores for the new benchmark." > experiments/exp_lme_0000/hypothesis.txt
```

- [ ] **Step 4: Run baseline experiment**

```bash
PYTHONPATH=src uv run python -m memory_decay.runner experiments/exp_lme_0000 --cache cache --force
```

Expected: `Done: overall=X.XXXX retrieval=X.XXXX plausibility=X.XXXX`

- [ ] **Step 5: Set as new best**

```bash
rm -f experiments/best
ln -s exp_lme_0000 experiments/best
```

- [ ] **Step 6: Record in history**

```bash
python3 -c "
import json
with open('experiments/exp_lme_0000/results.json') as f:
    r = json.load(f)
entry = {
    'exp': 'exp_lme_0000',
    'overall': round(r['overall_score'], 4),
    'retrieval': round(r['retrieval_score'], 4),
    'plausibility': round(r['plausibility_score'], 4),
    'status': 'baseline',
    'hypothesis': 'LongMemEval baseline with default exponential decay + Gemini embeddings'
}
with open('experiments/history.jsonl', 'a') as f:
    f.write(json.dumps(entry) + '\n')
print(json.dumps(entry, indent=2))
"
```

- [ ] **Step 7: Commit**

```bash
git add experiments/exp_lme_0000/ experiments/history.jsonl experiments/archive_memories500/
git commit -m "feat: LongMemEval baseline experiment (exp_lme_0000) with Gemini embeddings

Replaced memories_500.jsonl with LongMemEval (ICLR 2025, 500 questions).
Old experiments archived to experiments/archive_memories500/.

Constraint: Gemini embedding-001 required for English text
Rejected: ko-sroberta-multitask | Korean-only, incompatible with English dataset
Confidence: high
Scope-risk: moderate
Tested: baseline experiment completed successfully
Directive: all future experiments use exp_lme_ prefix to distinguish from old exp_ series"
```

---

## Task 6: Update program.md and Run Best Decay Function

**Files:**
- Modify: `program.md`

- [ ] **Step 1: Update preflight dataset reference in program.md**

In `program.md`, the preflight section references `data/memories_500.jsonl` and `data/memories_50.jsonl`. Update to reflect LongMemEval:

Change the preflight section to note the new dataset:

```
## Preflight (run once before the loop if missing)

If `cache/embeddings.pkl` does not exist or needs rebuilding, run:

1. Convert source data (if not done):
   ```bash
   PYTHONPATH=src:. uv run python scripts/convert_longmemeval.py
   ```

2. Build embedding cache:
   ```bash
   PYTHONPATH=src uv run python -m memory_decay.cache_builder \
     --dataset data/longmemeval.jsonl --output cache --backend gemini
   ```
```

- [ ] **Step 2: Run current best decay function (exp_0315 Jost+sigmoid) on new data**

```bash
mkdir -p experiments/exp_lme_0001
cp experiments/exp_0315/decay_fn.py experiments/exp_lme_0001/decay_fn.py
cp experiments/exp_0315/params.json experiments/exp_lme_0001/params.json
echo "Run current best (Jost+sigmoid power=4.0) on LongMemEval to compare with old dataset performance." > experiments/exp_lme_0001/hypothesis.txt
PYTHONPATH=src uv run python -m memory_decay.runner experiments/exp_lme_0001 --cache cache
```

- [ ] **Step 3: Compare and record**

```bash
python3 -c "
import json
with open('experiments/exp_lme_0000/results.json') as f:
    baseline = json.load(f)
with open('experiments/exp_lme_0001/results.json') as f:
    jost = json.load(f)
print('=== LongMemEval Comparison ===')
print(f'Baseline:  overall={baseline[\"overall_score\"]:.4f}  retrieval={baseline[\"retrieval_score\"]:.4f}  plausibility={baseline[\"plausibility_score\"]:.4f}')
print(f'Jost 4.0:  overall={jost[\"overall_score\"]:.4f}  retrieval={jost[\"retrieval_score\"]:.4f}  plausibility={jost[\"plausibility_score\"]:.4f}')
print(f'sim_recall baseline={baseline[\"similarity_recall_rate\"]:.4f}  jost={jost[\"similarity_recall_rate\"]:.4f}')
"
```

- [ ] **Step 4: Update best symlink if Jost is better, record in history**

```bash
# If Jost outperforms baseline:
python3 -c "
import json
with open('experiments/exp_lme_0000/results.json') as f:
    b = json.load(f)
with open('experiments/exp_lme_0001/results.json') as f:
    j = json.load(f)
if j['overall_score'] > b['overall_score']:
    import os
    os.remove('experiments/best')
    os.symlink('exp_lme_0001', 'experiments/best')
    print(f'Updated best → exp_lme_0001 (overall {j[\"overall_score\"]:.4f})')
else:
    print(f'Baseline still best (baseline={b[\"overall_score\"]:.4f}, jost={j[\"overall_score\"]:.4f})')

entry = {
    'exp': 'exp_lme_0001',
    'overall': round(j['overall_score'], 4),
    'retrieval': round(j['retrieval_score'], 4),
    'plausibility': round(j['plausibility_score'], 4),
    'status': 'improved' if j['overall_score'] > b['overall_score'] else 'recorded',
    'hypothesis': 'Jost+sigmoid power=4.0 on LongMemEval'
}
with open('experiments/history.jsonl', 'a') as f:
    f.write(json.dumps(entry) + '\n')
"
```

- [ ] **Step 5: Commit**

```bash
git add experiments/exp_lme_0001/ experiments/history.jsonl experiments/best program.md
git commit -m "feat: run Jost+sigmoid on LongMemEval, update program.md preflight

Constraint: old pre_program_pipeline not applicable to new dataset
Tested: baseline + Jost decay on LongMemEval with Gemini embeddings
Directive: future experiments use exp_lme_ prefix"
```

---

## Task 7: Reset Memory Chain for New Dataset

**Files:**
- Archive and reset: `memory_chain/`

- [ ] **Step 1: Archive old memory chain**

```bash
mkdir -p memory_chain/archive_memories500
mv memory_chain/round_*.md memory_chain/archive_memories500/ 2>/dev/null || true
mv memory_chain/failure_patterns.md memory_chain/archive_memories500/ 2>/dev/null || true
cp memory_chain/memory_index.jsonl memory_chain/archive_memories500/ 2>/dev/null || true
> memory_chain/memory_index.jsonl
```

- [ ] **Step 2: Create initial failure_patterns.md for new dataset**

Create `memory_chain/failure_patterns.md`:

```markdown
# Memory Chain - Failure Pattern Analysis (LongMemEval)

## Overview

Fresh start with LongMemEval dataset (ICLR 2025). Previous analysis from memories_500.jsonl archived to `archive_memories500/`.

## What Worked (from prior dataset — needs revalidation)

- Jost Law decay with power=4.0 + sigmoid floor was best on old synthetic data
- To be confirmed on LongMemEval

## What Failed (from prior dataset — needs revalidation)

Previous failures may not apply to new dataset. Re-test before ruling out:
- Gompertz floor
- Decoupled impact/stability
- Memory-type-specific Jost curvature
- Dual-sigmoid floor

## New Findings

(To be populated as experiments run on LongMemEval)
```

- [ ] **Step 3: Write initial memory chain round**

Create `memory_chain/round_0000.md` for the baseline:

```bash
python3 -c "
import json
with open('experiments/best/results.json') as f:
    r = json.load(f)
best_exp = 'exp_lme_0000'
# Check if exp_lme_0001 is best
import os
if os.path.islink('experiments/best'):
    best_exp = os.readlink('experiments/best')

content = f'''# Memory Chain — Round 0000

## Experiment: {best_exp}
**Date**: 2026-03-20
**Parent**: none

## Scores
| Metric | Value |
|--------|-------|
| overall_score | {r[\"overall_score\"]:.4f} |
| retrieval_score | {r[\"retrieval_score\"]:.4f} |
| plausibility_score | {r[\"plausibility_score\"]:.4f} |
| recall_mean | {r[\"recall_mean\"]:.3f} |
| mrr_mean | {r[\"mrr_mean\"]:.3f} |
| precision_lift | {r[\"precision_lift\"]:.3f} |
| similarity_recall_rate | {r[\"similarity_recall_rate\"]:.3f} |

## Hypothesis
LongMemEval baseline — fresh start with published benchmark dataset and Gemini embeddings.

## Self-Criticism
- This is the initial baseline on a new dataset. No comparison to make yet.
- Key question: does the embedding ceiling differ from the old dataset?
- similarity_recall_rate will tell us the new embedding quality bound.

## Decisions Made
- Switched from memories_500.jsonl (synthetic Korean) to LongMemEval (published English)
- Switched from ko-sroberta to gemini-embedding-001

## What To Avoid
- Nothing yet — clean slate on new data

## Next Step Direction
Run Jost+sigmoid (old best) to see if it transfers, then explore from there.
'''
with open('memory_chain/round_0000.md', 'w') as f:
    f.write(content)

# Write index entry
import datetime
entry = {
    'round': 0,
    'experiment': best_exp,
    'next_direction': 'Run old best (Jost+sigmoid) on new dataset',
    'timestamp': datetime.datetime.now().isoformat()
}
with open('memory_chain/memory_index.jsonl', 'w') as f:
    f.write(json.dumps(entry) + '\n')
print('Memory chain initialized')
"
```

- [ ] **Step 4: Commit**

```bash
git add memory_chain/
git commit -m "chore: reset memory chain for LongMemEval dataset

Old memories_500.jsonl analysis archived to memory_chain/archive_memories500/.
Fresh failure_patterns.md and round_0000.md for new benchmark.

Constraint: prior failure patterns may not apply to new dataset
Directive: re-test previously-failed approaches before ruling them out"
```

---

## Verification Checklist

After all tasks complete, verify:

- [ ] `data/longmemeval_oracle.json` exists (500 questions)
- [ ] `data/longmemeval.jsonl` exists with valid schema
- [ ] `cache/embeddings.pkl` contains Gemini embeddings
- [ ] `experiments/exp_lme_0000/results.json` has `status: "completed"`
- [ ] `experiments/best` symlink points to best experiment
- [ ] `experiments/history.jsonl` has baseline entry
- [ ] `memory_chain/round_0000.md` has baseline scores
- [ ] `program.md` preflight references new dataset
- [ ] Old data preserved in `data/memories_500.jsonl` and `experiments/archive_memories500/`
