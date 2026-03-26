# Session Handoff Summary — Design

## Status

Approved for planning.

## Background

The **session amnesia problem** is acute in OpenClaw because sessions are long-running and multi-step. When an agent resumes after a break, it has no automatic way to reconstruct context — it either starts from scratch or the user re-explains everything.

The current memory-decay system handles storage and retrieval well, but lacks a mechanism for **structured session context transfer** between sessions.

## Goal

Provide a single endpoint that a consuming agent (openclaw-memory-decay, claude-code-memory-decay) can call at:
- **Session end** — capture final state
- **Periodic intervals** — capture state every N hours during long sessions

The result is stored as a memory the next session can retrieve.

---

## API: `POST /session-summary`

### Request

```json
{
  "hours": 24,
  "min_activation": 0.1,
  "top_k": 50
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `hours` | 24 | How far back to look for recent memories |
| `min_activation` | 0.1 | Skip memories with activation below this threshold |
| `top_k` | 50 | Maximum number of memories to include |

### Response

```json
{
  "summary": "## Session Handoff\n\n### Active Tasks\n- ...\n\n### Recent Decisions\n- ...\n\n### Current Context\n- ...",
  "memory_id": "sess_<uuid>",
  "memories_included": 12
}
```

| Field | Description |
|-------|-------------|
| `summary` | Markdown-formatted handoff text |
| `memory_id` | ID of the stored session_summary memory |
| `memories_included` | Count of individual memories used to build the summary |

### Parameter Validation

| Field | Validation |
|-------|------------|
| `hours` | Must be positive integer (> 0). Default is 24. |
| `min_activation` | Must be in [0.0, 1.0]. Default 0.1. |
| `top_k` | Must be positive integer (> 0). Default 50. |

Invalid parameters return HTTP 422 with a descriptive error.

### Behavior

1. **Query** recent memories from the last `hours` with `activation >= min_activation`, ordered by activation descending, limited to `top_k`
2. **Group** by category:
   - `task` / `episode` → **Active Tasks**
   - `decision` → **Recent Decisions**
   - `fact` / `preference` → **Current Context**
3. **Format** into a markdown handoff summary with the structure above
4. **Store** as a new memory:
   - `memory_id`: `sess_<uuid>`
   - `mtype`: `fact`
   - `category`: `session_summary`
   - `importance`: `0.95`
   - `content`: the summary text
5. **Return** the summary text, memory ID, and count of included memories

### Error Handling

| Condition | Response |
|-----------|----------|
| Query yields 0 memories | HTTP 200 with `{"summary": "## Session Handoff\n\nNo active memories found.", "memory_id": "sess_<uuid>", "memories_included": 0}` — still stores an empty summary |
| Storage fails | HTTP 500 — the endpoint must succeed in storing the summary; callers depend on having it retrievable |
| Invalid parameters | HTTP 422 with validation error details |

### Example

**Request:** `POST /session-summary` with default params, 3 relevant memories in store.

**Memories found:**
- `{"id": "m1", "category": "task", "content": "Fix auth middleware", "activation": 0.85}`
- `{"id": "m2", "category": "decision", "content": "Chose SQLite over Postgres", "activation": 0.72}`
- `{"id": "m3", "category": "preference", "content": "User prefers dark mode", "activation": 0.60}`

**Response:**
```json
{
  "summary": "## Session Handoff\n\n### Active Tasks\n- Fix auth middleware\n\n### Recent Decisions\n- Chose SQLite over Postgres\n\n### Current Context\n- User prefers dark mode",
  "memory_id": "sess_abc123",
  "memories_included": 3
}
```

### Storage Memory Properties

The stored session_summary memory uses:
- `mtype=fact` — decays slowly (lambda_fact=0.02)
- `importance=0.95` — near the ceiling of the soft-floor formula `A(t+1) = floor(impact) + (A(t) - floor(impact)) * exp(-rate)`, so it converges toward 0.95 as the activation floor and remains highly retrievable
- `category=session_summary` — searchable via `/search` with category filter

This means the handoff summary persists across ticks and is easy to retrieve at session start.

---

## Trigger Logic (Plugin Layer)

Timer-based triggering is **not** implemented in memory-decay-core. It lives in the consuming agent:

- **openclaw-memory-decay** / **claude-code-memory-decay**: call `POST /session-summary` on session-end event and on a periodic timer (e.g., every N hours)
- The plugin also calls `GET /search` with `category=session_summary` at session start to retrieve the last handoff

This keeps memory-decay-core focused on storage and retrieval; session lifecycle management stays in the plugin.

---

## What This Does NOT Do

- It does NOT replace memories — the original memories remain in the store with their own decay
- It does NOT do LLM summarization — the summary is structured grouping, not generative compression
- It does NOT handle timer logic — that is the plugin's responsibility

---

## What This Assumes

- The memory store is already populated with relevant memories before the endpoint is called
- `min_activation` is tuned to the deployed decay curve — if the tick interval is very short (e.g., minutes), memories decay faster and `min_activation` should be lowered accordingly
- Plugins retrieve the last session_summary via `/search` with `category=session_summary` and `top_k=1` ordered by `created_tick` descending

---

## Open Questions

None at this time.

---

## Alternatives Considered

**B: Utility class `SessionSummaryGenerator`**
- Core provides a class, plugin calls it and stores the result
- More flexible but adds integration burden per plugin
- Rejected in favor of self-contained endpoint

**C: New memory type + query helpers**
- Expose `get_active_tasks()`, `get_recent_decisions()` helpers
- Plugin assembles the summary
- Adds complexity without reducing plugin burden
- Rejected
