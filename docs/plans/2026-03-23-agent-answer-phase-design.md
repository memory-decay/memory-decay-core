# Agent-based MemoryBench Answer Phase

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace MemoryBench's single-LLM-call answer generation with Claude Code CLI agent that autonomously searches, re-queries, and reasons over memories.

**Architecture:** Provider-level `answerFunction` override. When defined, answer.ts calls it instead of `generateText()`. The memory-decay provider implements `answerFunction` by invoking `claude -p` with a memory-retrieval skill that teaches the agent how to use the `/search` API.

**Tech Stack:** Claude Code CLI (`claude -p`), memory-decay FastAPI server, MemoryBench TypeScript

---

## Components

### 1. Claude Code Skill: `memory-retrieval`

Located at project root or `~/.claude/skills/memory-retrieval/`.

Teaches Claude Code:
- How to call `/search` endpoint (curl)
- How to interpret results (score, date, speaker)
- Re-query strategy: if initial results are insufficient, try different keywords
- Temporal reasoning: compute date differences using memory dates
- Knowledge updates: latest date wins
- Answer format: concise, factual, direct

### 2. MemoryBench Provider Interface Change

Add optional `answerFunction` to `Provider` interface:

```typescript
interface Provider {
  // ... existing
  answerFunction?: (
    question: string,
    context: unknown[],
    questionDate?: string
  ) => Promise<string>
}
```

### 3. MemoryBench answer.ts Change

In `runAnswerPhase`, before `generateText()`:

```typescript
if (provider?.answerFunction) {
  const text = await provider.answerFunction(question, context, questionDate)
  // save hypothesis, skip generateText
}
```

### 4. memory-decay Provider `answerFunction`

Calls `claude -p` with:
- System prompt from the skill
- Previous conversations as context
- The question
- Server URL for `/search`
- `--output-format json` for clean parsing
- `--allowedTools "Bash"` so agent can curl the server
- `--max-turns` to limit cost

## Data Flow

```
MemoryBench orchestrator
  → answer phase
    → provider.answerFunction(question, searchResults, questionDate)
      → builds prompt with conversation context + question
      → spawns: claude -p "<prompt>" --output-format json --allowedTools "Bash"
        → Claude Code agent reads prompt
        → agent calls curl localhost:8100/search with its own queries
        → agent reasons over results
        → agent outputs answer
      → parses JSON response, extracts result text
    → saves hypothesis to checkpoint
  → evaluate phase (unchanged)
```

## CLI Invocation Pattern

```bash
claude -p "$(cat <<'EOF'
<previous_conversations>
[Memory 1] (speaker: User, date: 2023-05-28) I just finished binge-watching...
[Memory 2] ...
</previous_conversations>

Question: Which show did I start watching first?
Today's date: 2023-06-15

Use the memory-decay server at http://localhost:8100 to search for additional context if needed.
EOF
)" --output-format json --allowedTools "Bash" --max-turns 5
```

## Cost Control

- `--max-turns 5`: limits agentic loops
- `--max-budget-usd 0.50`: per-question spending cap
- Initial search results passed as context to reduce re-queries
