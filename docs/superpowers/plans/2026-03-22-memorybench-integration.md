# MemoryBench Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the memory-decay system as a custom MemoryBench provider for standardized LongMemEval evaluation against Supermemory, Mem0, and Zep.

**Architecture:** HTTP bridge — a TypeScript provider in a memorybench fork calls the existing memory-decay FastAPI server. New `/reset` endpoint + `created_tick` support on the Python side; ~150-line provider class on the TS side.

**Tech Stack:** Python/FastAPI (server), TypeScript/Bun (MemoryBench provider), NetworkX, sentence-transformers

**Spec:** `docs/superpowers/specs/2026-03-22-memorybench-integration-design.md`

---

## Chunk 1: Python-Side Changes (memory-decay repo)

### Task 1: Add `MemoryGraph.clear()` method

**Files:**
- Modify: `src/memory_decay/graph.py` (add method after `__init__`, around line 45)

- [ ] **Step 1: Add `clear()` method to MemoryGraph**

Add after `__init__` (line 45 of `graph.py`):

```python
def clear(self) -> int:
    """Clear all nodes/edges and caches, preserving the embedding cache.

    Returns the number of nodes that were removed.
    """
    count = self._graph.number_of_nodes()
    self._graph.clear()
    # Reset precomputed similarity matrix
    self._emb_matrix = None
    self._emb_nids = []
    self._emb_nid_to_idx = {}
    self._emb_created_ticks = None
    self._emb_retrieval_scores = None
    self._emb_node_count = 0
    # Reset BM25 index
    self._bm25_idf = None
    self._bm25_doc_tokens = None
    self._bm25_avgdl = 0.0
    # Reset query stats
    self._query_similarity_total_time = 0.0
    self._query_similarity_call_count = 0
    # NOTE: _embedding_cache is intentionally preserved
    return count
```

- [ ] **Step 2: Verify the method works**

Run in Python REPL or a quick test:
```bash
cd /home/roach/.openclaw/workspace/memory-decay
python3 -c "
from src.memory_decay.graph import MemoryGraph
g = MemoryGraph(embedder=lambda t: [0.0]*384)
g.add_memory('m1', 'fact', 'hello', 0.5, 0)
print(f'Before: {g._graph.number_of_nodes()} nodes')
cleared = g.clear()
print(f'Cleared: {cleared}, After: {g._graph.number_of_nodes()} nodes')
assert g._graph.number_of_nodes() == 0
assert g._emb_node_count == 0
print('OK')
"
```
Expected: `Before: 1 nodes`, `Cleared: 1, After: 0 nodes`, `OK`

---

### Task 2: Add `DecayEngine.reset()` method

**Files:**
- Modify: `src/memory_decay/decay.py` (add method after `set_params`, around line 122)

- [ ] **Step 1: Add `reset()` method to DecayEngine**

Add after `set_params()` (line 122 of `decay.py`):

```python
def reset(self) -> None:
    """Reset tick counter and cached arrays to initial state."""
    self.current_tick = 0
    self._tick_arrays_built = False
    self._tick_nids = []
    self._tick_retrieval = None
    self._tick_storage = None
    self._tick_stability = None
    self._tick_impact = None
    self._tick_created = None
    self._tick_is_fact = None
```

- [ ] **Step 2: Verify the method works**

```bash
python3 -c "
from src.memory_decay.graph import MemoryGraph
from src.memory_decay.decay import DecayEngine
g = MemoryGraph(embedder=lambda t: [0.0]*384)
e = DecayEngine(g)
e.current_tick = 100
e._tick_arrays_built = True
e.reset()
assert e.current_tick == 0
assert e._tick_arrays_built == False
print('OK')
"
```
Expected: `OK`

---

### Task 3: Add `/reset` endpoint and `created_tick` to `/store`

**Files:**
- Modify: `src/memory_decay/server.py`

- [ ] **Step 1: Add `created_tick` field to `StoreRequest`**

In `StoreRequest` class (line 61-66 of `server.py`), add one field:

```python
class StoreRequest(BaseModel):
    text: str
    importance: float = Field(default=0.7, ge=0.0, le=1.0)
    category: str = "other"
    mtype: str = "fact"
    associations: list[str] | None = None
    created_tick: int | None = None  # NEW: if set, memory is created at this tick
```

- [ ] **Step 2: Update `/store` handler to use `created_tick`**

Change line 201 in the `/store` handler from:

```python
created_tick=_state.current_tick,
```

to:

```python
created_tick=req.created_tick if req.created_tick is not None else _state.current_tick,
```

- [ ] **Step 3: Add `/reset` endpoint**

Add before the `return app` line (before line 287 of `server.py`):

```python
@app.post("/reset")
def reset():
    if not _state:
        raise HTTPException(503, "Server not initialized")

    cleared = _state.graph.clear()
    _state.engine.reset()
    _state.current_tick = 0
    _state._memory_counter = 0
    _state.last_tick_time = time.time()

    return {"status": "ok", "cleared": cleared}
```

- [ ] **Step 4: Verify server endpoints manually**

**Note:** The server must use an English-capable embedding model for LongMemEval (not the default Korean `ko-sroberta`). Either set `GEMINI_API_KEY` for auto-detection, or the server will fall back to the MiniLM model from the embedding cache. For this test, the dummy embedder loaded via `--experiment-dir` suffices.

```bash
# Start server in background
python3 -m memory_decay.server --port 8199 --experiment-dir experiments/exp_lme_0292 &
SERVER_PID=$!
sleep 3

# Store with created_tick
curl -s -X POST http://localhost:8199/store \
  -H "Content-Type: application/json" \
  -d '{"text":"test memory","importance":0.7,"mtype":"fact","created_tick":5}' | python3 -m json.tool

# Check stats
curl -s http://localhost:8199/stats | python3 -m json.tool

# Reset
curl -s -X POST http://localhost:8199/reset | python3 -m json.tool

# Confirm cleared
curl -s http://localhost:8199/stats | python3 -m json.tool

# Cleanup
kill $SERVER_PID
```

Expected:
- `/store` returns `{"id": "mem_...", "text": "test memory", "tick": 0}` (tick stays 0 even though created_tick=5)
- `/stats` shows `"num_memories": 1`
- `/reset` returns `{"status": "ok", "cleared": 1}`
- Second `/stats` shows `"num_memories": 0`

- [ ] **Step 5: Commit Python-side changes**

```bash
git add src/memory_decay/graph.py src/memory_decay/decay.py src/memory_decay/server.py
git commit -m "feat: add /reset endpoint and created_tick support for MemoryBench integration

MemoryGraph.clear() wipes graph and index caches but preserves
embedding cache for performance. DecayEngine.reset() zeros tick
and clears pre-extracted arrays. /store now accepts optional
created_tick for temporal ordering of ingested sessions.

Constraint: embedding cache preserved across resets (deterministic)
Constraint: created_tick does not advance server current_tick
Confidence: high
Scope-risk: narrow
Reversibility: clean
Tested: manual curl verification of /reset and /store created_tick
Not-tested: high-volume reset cycles (500+ questions)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Chunk 2: MemoryBench Provider (memorybench fork)

### Task 4: Clone and set up memorybench

**Files:**
- New repo: clone `supermemoryai/memorybench` to a sibling directory

- [ ] **Step 1: Clone the repository**

```bash
cd /home/roach/.openclaw/workspace
git clone https://github.com/supermemoryai/memorybench.git
cd memorybench
```

- [ ] **Step 2: Install dependencies**

```bash
bun install
```

- [ ] **Step 3: Create `.env.local` with API key**

```bash
echo "OPENAI_API_KEY=$OPENAI_API_KEY" > .env.local
echo "MEMORY_DECAY_BASE_URL=http://localhost:8100" >> .env.local
```

- [ ] **Step 4: Verify setup**

```bash
bun run src/index.ts help
```

Expected: CLI help output listing available commands.

---

### Task 5: Explore memorybench provider patterns

Before writing our provider, read the existing ones to understand exact import paths, type names, and registration patterns.

- [ ] **Step 1: Read the existing provider index to understand registration**

Check `src/providers/index.ts` for the provider map pattern.

- [ ] **Step 2: Read an existing provider for reference**

Check one of: `src/providers/supermemory/index.ts` or `src/providers/rag/index.ts` to understand:
- Import paths for `Provider`, `ProviderConfig`, `IngestOptions`, `SearchOptions`, `IngestResult`
- How `initialize()`, `ingest()`, `awaitIndexing()`, `search()`, `clear()` are structured
- How `prompts` are defined

- [ ] **Step 3: Read the type definitions**

Check `src/types/provider.ts` for `ProviderName` union and `src/utils/config.ts` for how provider config is loaded.

---

### Task 6: Create the memory-decay provider

**Files:**
- Create: `src/providers/memory-decay/index.ts`
- Create: `src/providers/memory-decay/prompts.ts`

- [ ] **Step 1: Create the prompts file**

```typescript
// src/providers/memory-decay/prompts.ts
import type { ProviderPrompts } from "../../types/provider"

interface MemoryDecayResult {
  id: string
  text: string
  score: number
  storage_score: number
  retrieval_score: number
  category: string
  created_tick: number
}

export const memoryDecayPrompts: ProviderPrompts = {
  answerPrompt: (question: string, context: unknown[], questionDate?: string) => {
    const results = context as MemoryDecayResult[]
    const formatted = results
      .map((r, i) => `[Memory ${i + 1}] (score: ${r.score.toFixed(3)}) ${r.text}`)
      .join("\n\n")

    return `Answer based only on the provided memories.
If the memories don't contain the answer, say "I don't know".

Memories:
${formatted}

Question: ${question}
Answer:`
  },
}
```

- [ ] **Step 2: Create the provider implementation**

**Note:** The code below is a draft based on the MemoryBench Provider interface from documentation. After Task 5 (explore patterns), adjust import paths, type names, and method signatures (especially `awaitIndexing` parameter count) to match the actual codebase.

```typescript
// src/providers/memory-decay/index.ts
import type {
  Provider,
  ProviderConfig,
  IngestOptions,
  IngestResult,
  SearchOptions,
  ProviderPrompts,
} from "../../types/provider"
import type { UnifiedSession } from "../../types/unified"
import { memoryDecayPrompts } from "./prompts"

const ONE_DAY_MS = 24 * 60 * 60 * 1000

export class MemoryDecayProvider implements Provider {
  name = "memory-decay"
  prompts: ProviderPrompts = memoryDecayPrompts
  private baseUrl = "http://localhost:8100"
  private simulateTicks = 0

  async initialize(config: ProviderConfig): Promise<void> {
    if (config.baseUrl) {
      this.baseUrl = config.baseUrl as string
    }
    // Health check
    try {
      const res = await fetch(`${this.baseUrl}/health`, { signal: AbortSignal.timeout(5000) })
      if (!res.ok) throw new Error(`Health check failed: ${res.status}`)
    } catch (e) {
      throw new Error(
        `Cannot reach memory-decay server at ${this.baseUrl}. ` +
        `Start it with: python -m memory_decay.server --port 8100 --experiment-dir experiments/exp_lme_0292`
      )
    }
  }

  async ingest(sessions: UnifiedSession[], options: IngestOptions): Promise<IngestResult> {
    // Reset graph for per-question isolation
    await fetch(`${this.baseUrl}/reset`, { method: "POST" })

    // Compute date-to-tick mapping
    const sessionDates = this.extractSessionDates(sessions)
    const earliestMs = Math.min(...sessionDates.map(d => d.getTime()))
    const questionDate = this.extractQuestionDate(options)

    // Compute target ticks for simulation
    if (questionDate) {
      this.simulateTicks = Math.max(1, Math.floor((questionDate.getTime() - earliestMs) / ONE_DAY_MS))
    } else {
      // Fallback: use latest session date + 30 days
      const latestMs = Math.max(...sessionDates.map(d => d.getTime()))
      this.simulateTicks = Math.max(1, Math.floor((latestMs - earliestMs) / ONE_DAY_MS) + 30)
    }

    // Ingest each message as a memory
    const documentIds: string[] = []
    for (let si = 0; si < sessions.length; si++) {
      const session = sessions[si]
      const sessionDate = sessionDates[si]
      const createdTick = Math.floor((sessionDate.getTime() - earliestMs) / ONE_DAY_MS)

      for (const msg of session.messages) {
        const importance = msg.role === "user" ? 0.7 : 0.4
        const prefix = msg.role === "user" ? "[User]" : "[Assistant]"
        const text = `${prefix} ${msg.content}`

        try {
          const res = await fetch(`${this.baseUrl}/store`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              text,
              importance,
              mtype: "episode",
              created_tick: createdTick,
            }),
            signal: AbortSignal.timeout(30000),
          })
          if (res.ok) {
            const data = await res.json() as { id: string }
            documentIds.push(data.id)
          } else {
            console.warn(`[memory-decay] /store failed for session ${session.sessionId}: ${res.status}`)
          }
        } catch (e) {
          console.warn(`[memory-decay] /store error: ${e}`)
        }
      }
    }

    return { documentIds }
  }

  async awaitIndexing(_result: IngestResult, _containerTag: string): Promise<void> {
    // Run decay simulation to the question date.
    // Server caps /tick at 1000, so loop for large ranges (LongMemEval can span years).
    let remaining = this.simulateTicks
    while (remaining > 0) {
      const batch = Math.min(remaining, 1000)
      await fetch(`${this.baseUrl}/tick`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ count: batch }),
        signal: AbortSignal.timeout(30000),
      })
      remaining -= batch
    }
  }

  async search(query: string, options: SearchOptions): Promise<unknown[]> {
    try {
      const res = await fetch(`${this.baseUrl}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: options.limit || 7 }),
        signal: AbortSignal.timeout(30000),
      })
      if (!res.ok) return []
      const data = await res.json() as { results: unknown[] }
      return data.results
    } catch {
      return []
    }
  }

  async clear(_containerTag: string): Promise<void> {
    await fetch(`${this.baseUrl}/reset`, { method: "POST" })
  }

  // --- Private helpers ---

  private extractSessionDates(sessions: UnifiedSession[]): Date[] {
    return sessions.map(session => {
      // Try metadata.date or metadata.formattedDate
      const meta = session.metadata || {}
      const dateStr = (meta.date || meta.formattedDate || meta.iso_date) as string | undefined
      if (dateStr) {
        const parsed = new Date(dateStr)
        if (!isNaN(parsed.getTime())) return parsed
      }
      // Try first message timestamp
      if (session.messages.length > 0 && session.messages[0].timestamp) {
        const parsed = new Date(session.messages[0].timestamp)
        if (!isNaN(parsed.getTime())) return parsed
      }
      // Fallback to epoch
      return new Date(0)
    })
  }

  private extractQuestionDate(options: IngestOptions): Date | null {
    const meta = options.metadata || {}
    const dateStr = (meta.questionDate || meta.question_date) as string | undefined
    if (dateStr) {
      const parsed = new Date(dateStr)
      if (!isNaN(parsed.getTime())) return parsed
    }
    return null
  }
}
```

---

### Task 7: Register the provider

**Files:**
- Modify: `src/providers/index.ts`
- Modify: `src/types/provider.ts`
- Modify: `src/utils/config.ts`

- [ ] **Step 1: Add to provider index**

In `src/providers/index.ts`, add the import and registration:

```typescript
import { MemoryDecayProvider } from "./memory-decay"

// Add to the providers map:
"memory-decay": new MemoryDecayProvider(),
```

Follow the exact pattern of other providers in that file.

- [ ] **Step 2: Add to ProviderName type**

In `src/types/provider.ts`, add `"memory-decay"` to the `ProviderName` union type.

- [ ] **Step 3: Add config**

In `src/utils/config.ts`, add:

```typescript
MEMORY_DECAY_BASE_URL: process.env.MEMORY_DECAY_BASE_URL || "http://localhost:8100",
```

And pass it as `baseUrl` in the provider config when `providerName === "memory-decay"`.

- [ ] **Step 4: Verify provider loads**

```bash
bun run src/index.ts run -p memory-decay -b longmemeval -j gpt-4o -r test-load --help
```

Should not crash with "unknown provider" error.

- [ ] **Step 5: Commit provider**

```bash
git add src/providers/memory-decay/ src/providers/index.ts src/types/provider.ts src/utils/config.ts
git commit -m "feat: add memory-decay provider for decay-based retrieval evaluation

HTTP bridge provider connecting to memory-decay FastAPI server.
Per-question isolation via /reset, date-to-tick mapping for
temporal decay simulation, role-based importance (0.7 user / 0.4
assistant).

Constraint: server must be running separately on localhost:8100
Rejected: subprocess bridge | too slow, model loading per call
Rejected: session-level granularity | message-level matches other providers
Confidence: high
Scope-risk: narrow
Reversibility: clean
Tested: provider registration and import
Not-tested: full LongMemEval pipeline (requires running server)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Chunk 3: Integration Test

### Task 8: End-to-end smoke test

- [ ] **Step 1: Start the memory-decay server**

Ensure `GEMINI_API_KEY` is set for English-capable embeddings, or the server will fall back to the local Korean model which will produce poor results on the English LongMemEval benchmark.

```bash
cd /home/roach/.openclaw/workspace/memory-decay
# Ensure English embedding model is used (auto-detects GEMINI_API_KEY)
python -m memory_decay.server --port 8100 --experiment-dir experiments/exp_lme_0292 &
```

- [ ] **Step 2: Run MemoryBench with 5 questions**

```bash
cd /home/roach/.openclaw/workspace/memorybench
bun run src/index.ts run \
  -p memory-decay \
  -b longmemeval \
  -j gpt-4o \
  -r smoke-test-5 \
  --limit 5
```

Verify: all 6 phases complete (ingest → index → search → answer → evaluate → report).

- [ ] **Step 3: Check results**

```bash
bun run src/index.ts status -r smoke-test-5
cat data/runs/smoke-test-5/report.json | python3 -m json.tool
```

Verify: report contains accuracy score and per-question results.

- [ ] **Step 4: Run full LongMemEval evaluation**

```bash
bun run src/index.ts run \
  -p memory-decay \
  -b longmemeval \
  -j gpt-4o \
  -r decay-0292-full
```

- [ ] **Step 5: Review results and compare**

```bash
bun run src/index.ts status -r decay-0292-full
bun run src/index.ts show-failures -r decay-0292-full
```
