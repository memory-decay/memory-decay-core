# LongMemEval Integration Design

**Date**: 2026-03-20
**Status**: Draft
**Goal**: Replace synthetic `memories_500.jsonl` with LongMemEval benchmark to validate decay functions against a published, peer-reviewed dataset (ICLR 2025).

## Motivation

The current dataset (`memories_500.jsonl`, 500 synthetic Korean memories) has hit a ceiling:
- `recall_mean` = `similarity_recall_rate` = 0.402 — decay optimization maxed out at embedding quality
- 20+ consecutive experiment failures suggest the search surface is exhausted
- The dataset is self-generated, so quality issues may be masking or inflating scores

LongMemEval provides 500 questions across 5 memory ability types with 4,865 unique user messages from multi-session chat histories. It directly tests what the decay system is designed for: long-term memory retention and retrieval over time.

## Data Source

- **Dataset**: `longmemeval_oracle.json` from HuggingFace (`xiaowu0162/longmemeval-cleaned`)
- **Size**: 500 questions, 4,865 unique user messages, 842 answer-bearing messages
- **Language**: English
- **License**: Research use

## Architecture Decision: Converter Script

**Chosen approach**: Offline converter script (`scripts/convert_longmemeval.py`) that transforms LongMemEval JSON into the existing `memories_NNNN.jsonl` schema.

**Why**: Zero changes to evaluator, runner, decay engine, or cross-validator. The entire integration is isolated to:
1. One converter script (new)
2. Cache rebuild (existing `cache_builder.py`)
3. Embedding backend switch to `gemini-embedding-001` (multilingual, already supported)

**Rejected**: Runtime adapter in `cache_builder.py` — couples dataset format to the pipeline, harder to debug.

## Data Mapping

### Memory Node Schema (output)

Each user message becomes a memory node in `data/longmemeval.jsonl`:

```json
{
  "id": "lme_q0042_s1_m3",
  "type": "fact" | "episode",
  "content": "I graduated with a degree in Business Administration.",
  "entities": [],
  "tick": 15,
  "impact": 0.8,
  "associations": [{"id": "lme_q0042_s0_m1", "weight": 0.7}],
  "recall_query": "What degree did I graduate with?",
  "recall_answer": "Business Administration"
}
```

### Field Mapping Rules

| LongMemEval field | → | memory-decay field | Transformation |
|---|---|---|---|
| `message.content` (user role only) | → | `content` | Direct copy |
| `message.has_answer` | → | `impact` | `true` → 0.8, `false` → 0.3 |
| `haystack_dates` | → | `tick` | Linear: earliest date → tick 0, latest → tick 200 |
| `question` | → | `recall_query` | Only on the target memory (first `has_answer=true` user message) |
| `answer` | → | `recall_answer` | Same as above |
| `question_type` | → | `type` | See type mapping below |
| Messages sharing `answer_session_ids` | → | `associations` | Bidirectional links, weight 0.7 |

### Type Mapping

| question_type | → | memory type | Rationale |
|---|---|---|---|
| `single-session-user` | → | `fact` | Single factual mention |
| `single-session-assistant` | → | `fact` | Information provided by assistant |
| `single-session-preference` | → | `fact` | Preferences = stable facts |
| `knowledge-update` | → | `fact` | Updatable knowledge |
| `temporal-reasoning` | → | `episode` | Time-ordered experiential memory |
| `multi-session` | → | `episode` | Cross-session experiential memory |

### Tick Mapping

LongMemEval dates span days/weeks. The converter normalizes all dates to the 0-200 tick range:

```python
all_dates = sorted(set of all message dates)
tick = int((date - min_date) / (max_date - min_date) * 200)
```

Messages on the same date get the same tick. This preserves temporal ordering while fitting the existing 200-tick simulation window.

### Deduplication

The same message may appear across multiple questions (shared haystack sessions). The converter deduplicates by `(session_id, message_index)` to avoid duplicate memory nodes. When a message is referenced by multiple questions, only the first question's `recall_query` is assigned; subsequent questions reference the same memory node by ID.

### Assistant Messages

Excluded. Only `role: "user"` messages become memory nodes. Rationale:
- The system models user memories, not assistant-generated content
- Assistant messages are typically generic responses, not memorable facts
- `has_answer=true` on assistant messages is rare (54/10960 = 0.5%)

## Embedding Backend

**Switch to `gemini-embedding-001`**:
- Current `ko-sroberta-multitask` is Korean-only, won't work with English data
- `gemini-embedding-001` is already supported in `graph.py` and `cache_builder.py`
- Requires `GEMINI_API_KEY` (already configured)

Cache rebuild command:
```bash
PYTHONPATH=src uv run python -m memory_decay.cache_builder \
  --dataset data/longmemeval.jsonl \
  --output cache \
  --backend gemini
```

## Impact on Existing System

### No changes required
- `evaluator.py` — reads `(recall_query, expected_id)` pairs, unchanged
- `runner.py` — loads from cache, unchanged
- `decay.py` / `DecayEngine` — operates on graph nodes, unchanged
- `graph.py` / `MemoryGraph` — schema-compatible
- `cross_validator.py` — uses `cache/dataset.json`, unchanged
- `program.md` experiment loop — unchanged (reads cache, runs experiments)

### Changes required
- `scripts/convert_longmemeval.py` — **new file**
- `cache/` — rebuilt with new embeddings
- `data/longmemeval.jsonl` — **new file** (converter output)
- `program.md` preflight section — update dataset path reference
- `experiments/best/` — reset baseline (new dataset = new scores)
- `experiments/history.jsonl` — archive and restart
- `outputs/pre_program_pipeline/` — re-run with new dataset

### Backup plan
- Current cache backed up to `cache_backup_local/`
- Current dataset remains at `data/memories_500.jsonl`
- Can revert by pointing cache_builder back to old dataset

## Converter Script Design

`scripts/convert_longmemeval.py`:

```
Input:  data/longmemeval_oracle.json (downloaded from HuggingFace)
Output: data/longmemeval.jsonl (one memory node per line, existing schema)

Steps:
1. Load oracle JSON
2. Parse all dates, compute tick mapping (min_date=0, max_date=200)
3. Deduplicate messages across questions by (session_id, msg_index)
4. For each question:
   a. Extract user messages from haystack_sessions
   b. Assign memory IDs: lme_q{qid}_s{session}_m{msg_idx}
   c. Map type from question_type
   d. Set impact: has_answer → 0.8, else → 0.3
   e. Build associations from co-referenced answer messages
   f. Assign recall_query/recall_answer to first has_answer user message
5. Write JSONL output
```

## Migration Sequence

1. Download `longmemeval_oracle.json` to `data/`
2. Run converter → `data/longmemeval.jsonl`
3. Backup current cache → `cache_backup_local/`
4. Rebuild cache with gemini backend
5. Run baseline experiment (`exp_0000` equivalent) on new data
6. Archive old `experiments/history.jsonl`
7. Update `program.md` preflight references
8. Re-run pre-program pipeline with new dataset
9. Resume experiment loop with fresh baseline

## Success Criteria

- Converter produces valid JSONL matching existing schema
- `runner.py` executes without changes on new cache
- Baseline experiment completes with `status: "completed"`
- `similarity_recall_rate` > 0.40 (embedding quality sanity check)
- Cross-validator runs successfully on new data
