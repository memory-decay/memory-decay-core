# MemoryBench Integration Design

## Overview

Integrate the memory-decay system as a custom provider in the [MemoryBench](https://github.com/supermemoryai/memorybench) framework to evaluate decay-based retrieval against the LongMemEval benchmark, enabling standardized comparison with Supermemory, Mem0, and Zep.

## Architecture

Two repositories, connected via HTTP:

```
memorybench (fork)                         memory-decay (existing repo)
├── src/providers/memory-decay/            ├── src/memory_decay/server.py
│   ├── index.ts   ── HTTP ──────────────► │   POST /reset
│   └── prompts.ts                         │   POST /store  (+ created_tick)
│                                          │   POST /tick   (existing, count=N)
│                                          │   POST /search
│                                          │   GET  /health
└── .env.local (OPENAI_API_KEY)            └── experiments/exp_lme_0292/
```

## Per-Question Execution Flow

MemoryBench evaluates each question in isolation. For each of the ~500 LongMemEval questions:

1. **clear()** → `POST /reset` — wipe graph, engine, and all caches
2. **ingest(sessions)** → date-to-tick mapping, then `POST /store` per message
3. **awaitIndexing()** → `POST /tick` with `{ "count": N }` — run decay to question_date tick
4. **search(query)** → `POST /search` with `{ "top_k": 7 }` — decay-weighted retrieval
5. MemoryBench generates answer via judge LLM (GPT-4o)
6. MemoryBench evaluates answer against ground truth

`containerTag` is ignored by our provider because the memory-decay server manages a single global graph. Per-question isolation is achieved by calling `/reset` before each question's ingest.

## Memory-Decay Server Changes

### New Endpoint

**POST /reset**

Clears the entire memory graph, decay engine, and all caches. Resets to a clean initial state.

```
Request:  (empty body)
Response: { "status": "ok", "cleared": <num_deleted> }
```

Reset must clear:
- `MemoryGraph`: NetworkX graph, BM25 index, precomputed similarity matrix (`_emb_matrix`, `_emb_nids`, etc.)
- `DecayEngine`: `current_tick` reset to 0, pre-extracted tick arrays (`_tick_nids`, `_tick_*`)
- `ServerState`: `current_tick` reset to 0, `_memory_counter` reset to 0

The embedding cache (`_embedding_cache`) is intentionally **preserved** across resets for performance — identical texts across questions will not be re-embedded. This is safe because embeddings are deterministic.

Implementation: add `MemoryGraph.clear()` and `DecayEngine.reset()` methods, called by the `/reset` handler.

### Modified Endpoint

**POST /store** — add optional `created_tick` parameter:

```json
{
  "text": "I prefer dark roast coffee",
  "importance": 0.5,
  "mtype": "episode",
  "created_tick": 14
}
```

When `created_tick` is provided, the memory is created at that tick instead of the server's current tick. This enables temporal ordering of ingested sessions.

**Invariant**: `/store` with `created_tick` does NOT advance the server's `current_tick`. After all stores, `current_tick` remains 0. Memories with `created_tick > 0` will not participate in decay until the server tick reaches that value via `/tick`.

### Unchanged

- `POST /tick` — runs `count` ticks of decay simulation (existing endpoint, no changes)
- `POST /search` — decay-weighted retrieval (no changes)
- `GET /health` — health check (no changes)
- 0292 decay parameters (decay_fn.py + params.json) — used as-is

### Search Response Contract

The TS provider depends on this response shape from `POST /search`:

```json
{
  "results": [
    {
      "id": "mem_abc123",
      "text": "I prefer dark roast coffee",
      "score": 0.8234,
      "storage_score": 0.75,
      "retrieval_score": 0.75,
      "category": "episode",
      "created_tick": 14
    }
  ]
}
```

The custom answer prompt uses `r.score` and `r.text` from each result.

### Embedding Provider

LongMemEval is an English benchmark. The server must use an English-capable embedding model, not the default Korean model (`jhgan/ko-sroberta-multitask`). Options:

- `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim) — used in existing cache
- Gemini embedding via `GEMINI_API_KEY` (3072-dim)

The server startup command should specify the embedding backend explicitly, or ensure the cached embeddings are compatible.

## TS Provider Implementation

### Provider Class

```typescript
// src/providers/memory-decay/index.ts
class MemoryDecayProvider implements Provider {
  name = "memory-decay"
  prompts: ProviderPrompts  // custom answer prompt
  private baseUrl: string   // from MEMORY_DECAY_BASE_URL or default http://localhost:8100
  private simulateTicks: number = 0  // computed during ingest, used in awaitIndexing

  initialize(config)      // GET /health check; fail with guidance if server not running
  ingest(sessions, opts)  // POST /reset + POST /store per message
  awaitIndexing(result)   // POST /tick with { count: simulateTicks }
  search(query, opts)     // POST /search with { top_k: 7 }
  clear(tag)              // POST /reset
}
```

### Date-to-Tick Mapping

Performed in the TS provider during `ingest()`:

- 1 tick = 1 day
- Earliest session date = tick 0
- Each session's date maps to `floor((date - earliest) / (1 day))`
- `question_date` (from MemoryBench metadata) = target tick for simulation
- `simulateTicks` = target tick (since current_tick is 0 after ingest)

Example: sessions on Jan 1, Jan 15, Feb 1; question on Mar 1
- created_ticks: [0, 14, 31]
- simulateTicks: 60

### Session-to-Memory Conversion

Each message in a session becomes an individual memory:

```json
{
  "text": "[User] Do you know my coffee preference?",
  "importance": 0.5,
  "mtype": "episode",
  "created_tick": 14
}
```

Importance assignment by role:
- User statements of fact/preference → `importance: 0.7` (higher — these are the recall targets)
- Assistant responses → `importance: 0.4` (lower — supporting context)
- Default fallback → `importance: 0.5`

This leverages the 0292 decay function's `importance_scaled_boost` logic to differentiate decay rates.

Rationale: message-level granularity matches other providers (Supermemory, Mem0) and allows decay to differentiate individual statements.

### Custom Answer Prompt

```typescript
// src/providers/memory-decay/prompts.ts
answerPrompt: (question, context, questionDate) => {
  const formatted = context
    .map((r, i) => `[Memory ${i+1}] (score: ${r.score.toFixed(3)}) ${r.text}`)
    .join("\n\n")

  return `Answer based only on the provided memories.
If the memories don't contain the answer, say "I don't know".

Memories:
${formatted}

Question: ${question}
Answer:`
}
```

### Registration

- Add `MemoryDecayProvider` to `src/providers/index.ts`
- Add `"memory-decay"` to `ProviderName` union in `src/types/provider.ts`
- Add `MEMORY_DECAY_BASE_URL` config in `src/utils/config.ts` (default: `http://localhost:8100`)

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Server not running | `initialize()` fails: "Start server: python -m memory_decay.server --port 8100 --experiment-dir experiments/exp_lme_0292" |
| Request timeout | 30s per request, no retry, skip question |
| `/store` failure | Skip that memory, log warning |
| `/search` empty results | Return empty array → judge handles "I don't know" |

## Running an Evaluation

```bash
# Terminal 1: Start memory-decay server with 0292 params
cd memory-decay
python -m memory_decay.server --port 8100 \
  --experiment-dir experiments/exp_lme_0292

# Terminal 2: Run MemoryBench evaluation
cd memorybench
bun run src/index.ts run \
  -p memory-decay \
  -b longmemeval \
  -j gpt-4o \
  -r decay-0292-baseline
```

### Viewing Results

```bash
bun run src/index.ts status -r decay-0292-baseline
bun run src/index.ts show-failures -r decay-0292-baseline
bun run src/index.ts serve  # web dashboard at localhost:3000
```

### Multi-Provider Comparison

```bash
bun run src/index.ts compare \
  -p memory-decay,supermemory,mem0 \
  -b longmemeval \
  -j gpt-4o
```

## Implementation Scope

| Location | Change | Size |
|----------|--------|------|
| `memory-decay/server.py` | `/reset` endpoint, `/store` created_tick | ~50 lines |
| `memory-decay/graph.py` | `clear()` method | ~15 lines |
| `memory-decay/decay.py` | `reset()` method | ~10 lines |
| `memorybench/src/providers/memory-decay/index.ts` | Provider implementation | ~150 lines |
| `memorybench/src/providers/memory-decay/prompts.ts` | Custom answer prompt | ~20 lines |
| `memorybench/src/providers/index.ts` | Provider registration | 2 lines |
| `memorybench/src/types/provider.ts` | ProviderName union | 1 line |
| `memorybench/src/utils/config.ts` | Base URL config | 3 lines |

Total: ~250 lines of new code.

## Verification Plan

1. **Smoke test**: 5 questions, verify full pipeline (ingest → search → answer → evaluate)
2. **Full run**: All LongMemEval questions (6 question types)
3. **Comparison**: Run same benchmark with supermemory and mem0 providers

## Decisions

- **Per-question isolation** over single-graph mode — enables fair comparison with other providers at the cost of not showcasing cross-memory decay effects
- **HTTP bridge** over subprocess — leverages existing FastAPI server, clean separation
- **Message-level granularity** over session-level — matches other providers, finer decay differentiation
- **1 tick = 1 day** mapping — intuitive temporal scaling for LongMemEval date ranges
- **Reuse `/tick`** over new `/simulate` endpoint — avoids redundant endpoint with subtly different schema
- **Preserve embedding cache across resets** — performance optimization, safe because embeddings are deterministic
- **Role-based importance** (0.7 user / 0.4 assistant) — leverages 0292's importance_scaled_boost for meaningful decay differentiation
