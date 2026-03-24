# Auto-Research Loop: MemoryBench-Driven Memory Decay Optimization

## Goal
Achieve ≥70% accuracy on all three MemoryBench benchmarks (LongMemEval, LoCoMo, ConvoMem) by optimizing decay functions and retrieval parameters.

**Current baseline** (experiments/best → exp_bench_0001):
- LongMemEval: 70% (7/10)
- LoCoMo: 100% (10/10)
- ConvoMem: 100% (10/10)
- Composite bench_score: 0.85

**Target**: bench_score ≥ 0.70, with each benchmark ≥ 70%. ✅ Achieved on 10-question samples.
Next goal: validate stability with 20-50 question runs.

## Operating Principle

The human defines the closed loop. The AI agent works only inside the allowed search surface.

- Evaluation uses MemoryBench (external benchmark framework) — 3 datasets, end-to-end QA accuracy
- Embedding cache is pre-built (`cache/openai/`, text-embedding-3-small, 334K texts) — no embedding API costs
- Only answer/judge LLM calls cost money (~$0.60 per 20-question evaluation)
- The agent explores decay functions, retrieval parameters, and server behavior within the allowed surface

## Infrastructure

**Embedding cache**: `cache/openai/embeddings.pkl` (pre-built, DO NOT rebuild)
**Server**: `memory-decay` FastAPI server on port 8100
**Benchmarks**: MemoryBench at `~/memorybench/`
**Embedding model**: OpenAI text-embedding-3-small (1536 dims)

### Server startup
```bash
pkill -f "memory_decay.server" 2>/dev/null; sleep 1
export OPENAI_API_KEY="$OPENAI_API_KEY"
.venv/bin/python -m memory_decay.server \
  --port 8100 \
  --cache-dir cache/openai \
  --embedding-provider openai \
  --embedding-api-key "$OPENAI_API_KEY" \
  --experiment-dir experiments/exp_bench_NNNN \
  &
sleep 3
```

### MemoryBench evaluation (per benchmark)
```bash
cd ~/memorybench && OPENAI_API_KEY="$OPENAI_API_KEY" bun run src/index.ts run \
  -p memory-decay -b <benchmark> -j gpt-4o -m gpt-4o \
  -r <run-id> -l 20 --force
```

## Fixed Dual-State Architecture

Two state variables per memory:
- `storage_score`: decays over time, used for threshold gating
- `retrieval_score`: decays over time, used for similarity ranking (`score = cosine_sim * retrieval_score^activation_weight`)

The decay function is applied independently to both states by the fixed engine.

## Protocol (each cycle)

### 1. Read State
- Read `experiments/history.jsonl` for previous experiment results
- Read `experiments/best/decay_fn.py` for the current best function
- Read `experiments/best/params.json` for the current best parameters
- Read `experiments/best/bench_results.json` for the current best MemoryBench scores
- Read `memory_chain/failure_patterns.md` and last 5 `memory_chain/round_NNNN.md` files

### 2. Analyze & Hypothesize

**Lessons from optimization (2026-03-23):**
- activation_weight 0.35→0.05: retrieval Hit@10 jumped to 100%. Decay was hiding relevant memories.
- BM25 (weight 0.1-0.3): hurt accuracy. Short messages ("Wow!", "Cool!") dominated keyword matching. Disabled.
- Hybrid chunking (individual + session chunks, MIN_CHUNK_SIZE=15): LoCoMo 30%→60%. Chunks absorb noise in long dialogues.
- Prompt v2 (temporal reasoning + inference): LoCoMo 60%→100%. Retrieval was already good — answer generation was the bottleneck.
- ConvoMem chunking fix: skip undated sessions (sessionDateMs=0) to avoid one giant unusable chunk.

**Key parameters (current best):**
- `activation_weight`: 0.05 — controls `retrieval_score^weight` penalty. Lower = less decay impact.
- `bm25_weight`: 0.0 (disabled) — hurts more than helps.
- `lambda_fact` / `lambda_episode`: 0.05 / 0.2 — base decay speeds.
- `floor_max`: 0.45 — minimum activation floor.

**Memory chain check** (mandatory):
- Does `failure_patterns.md` list this approach as non-viable?
- Do recent round files warn against this direction?
- If yes: choose a different direction.

### 3. Write Experiment Files
Create `experiments/exp_bench_NNNN/` (next sequential number) with:

**decay_fn.py** — Same interface as before:
```python
def compute_decay(activation, impact, stability, mtype, params):
    """Returns float 0-1, new activation (must be ≤ activation for pure decay)."""
```

**params.json** — Must include at minimum the standard keys. May also include:
- `bm25_weight`: float 0-1, enables BM25 lexical re-ranking
- `bm25_candidates`: int, number of candidates for BM25 stage
- All other standard keys (`lambda_fact`, `activation_weight`, etc.)

**hypothesis.txt** — What you're trying and why.

### 4. Run Experiment

Start the server with the new experiment, then run MemoryBench.

**Current focus: LongMemEval** (70% on 10q — weakest benchmark, highest weight 0.50).
Validate all benchmarks at 20-50 questions to confirm stability.

**IMPORTANT: Always use random sampling (`--sample-type random`) to prevent overfitting to specific questions.**

**Answer mode**: Use Claude Code agent mode (`MEMORY_DECAY_AGENT_MODE=1`).
**Judge**: gpt-4o-mini (`-j gpt-4o-mini`).
**Embedding model**: text-embedding-3-large (cache is 3072 dims).

**Stage A (quick screen)**: 3 questions/category × LongMemEval only.

```bash
# 1. Kill old server and start new one
pkill -f "memory_decay.server" 2>/dev/null; sleep 1
.venv/bin/python -m memory_decay.server \
  --port 8100 --cache-dir cache/openai \
  --embedding-provider openai --embedding-api-key "$OPENAI_API_KEY" \
  --embedding-model text-embedding-3-large \
  --experiment-dir experiments/exp_bench_NNNN &
sleep 3

# 2. Stage A: LongMemEval, 3/category random, agent mode
cd ~/memorybench
OPENAI_API_KEY="$OPENAI_API_KEY" MEMORY_DECAY_AGENT_MODE=1 bun run src/index.ts run \
  -p memory-decay -b longmemeval -j gpt-4o-mini -m sonnet \
  -s 3 --sample-type random \
  -r exp_bench_NNNN-stageA --force
```

**Stage B (confirmation)**: only if Stage A shows improvement, run 5/category × all 3 benchmarks:
```bash
for BENCH in longmemeval locomo convomem; do
  OPENAI_API_KEY="$OPENAI_API_KEY" MEMORY_DECAY_AGENT_MODE=1 bun run src/index.ts run \
    -p memory-decay -b $BENCH -j gpt-4o-mini -m sonnet \
    -s 5 --sample-type random \
    -r exp_bench_NNNN-stageB-$BENCH --force
done
```

### 5. Read Results

Parse LoCoMo report:
```bash
cd ~/memorybench
python3 -c "
import json
r = json.load(open('data/runs/exp_bench_NNNN-locomo/report.json'))
s = r['summary']
ret = r.get('retrieval', {})
print(f'LoCoMo: accuracy={s[\"accuracy\"]*100:.1f}% ({s[\"correctCount\"]}/{s[\"totalQuestions\"]})')
print(f'  MRR={ret.get(\"mrr\", 0):.3f}, Hit@10={ret.get(\"hitAtK\", 0)*100:.0f}%')
"
```

**Stage A metric**: LongMemEval accuracy (target ≥ 80%).
Current baseline: 70% (7/10) with 10 questions.

### 6. Judge

**Stage A**: Compare LongMemEval accuracy with best.
- If lme_acc > best_lme_acc: proceed to **Stage B** (20 questions × all 3 benchmarks with gpt-4o)
- If not improved: record in history, move on

**Stage B**: Compute full `bench_score` from 3 benchmarks:
```
bench_score = 0.50 × longmemeval_acc + 0.30 × locomo_acc + 0.20 × convomem_acc
```
- If bench_score > best_bench_score: **accept**
- If not: **reject** (Stage A was a false positive)

**On accept**, save results:
```bash
# Save bench_results.json
python3 -c "
import json
results = {
  'bench_score': SCORE,
  'benchmarks': {
    'longmemeval': {'accuracy': X, 'total': 20, 'correct': N, 'run_id': 'exp_bench_NNNN-longmemeval'},
    'locomo': {'accuracy': X, 'total': 20, 'correct': N, 'run_id': 'exp_bench_NNNN-locomo'},
    'convomem': {'accuracy': X, 'total': 20, 'correct': N, 'run_id': 'exp_bench_NNNN-convomem'},
  }
}
json.dump(results, open('experiments/exp_bench_NNNN/bench_results.json', 'w'), indent=2)
"

# Update best symlink
cd experiments && rm -f best && ln -s exp_bench_NNNN best
```

### 7. Write Memory Chain Round

Create `memory_chain/round_NNNN.md` with:

```markdown
# Memory Chain — Round NNNN

## Experiment: exp_bench_XXXX
**Date**: YYYY-MM-DD

## Scores
| Benchmark | Accuracy | Previous |
|-----------|----------|----------|
| LongMemEval | XX% | XX% |
| LoCoMo | XX% | XX% |
| ConvoMem | XX% | XX% |
| **bench_score** | **X.XX** | **X.XX** |

## Hypothesis
<what was tried and why>

## Self-Criticism
<analyze WHY this experiment succeeded or failed>
- Which benchmark improved/regressed and why?
- What parameter change had the biggest effect?

## What To Avoid
<approaches that should NOT be retried>

## Next Step Direction
<concrete direction for the next experiment>
```

### 8. Record
Append to `experiments/history.jsonl`:
```json
{"exp": "exp_bench_NNNN", "bench_score": 0.51, "lme_acc": 0.55, "locomo_acc": 0.15, "convomem_acc": 0.95, "status": "improved", "hypothesis": "short summary"}
```

### 9. Repeat or Stop
Continue unless:
- 20 cycles completed this session
- 10+ consecutive experiments with no improvement
- All benchmarks ≥ 70% achieved

## Allowed Search Surface

The agent may change:
- `experiments/exp_bench_NNNN/decay_fn.py` — decay function
- `experiments/exp_bench_NNNN/params.json` — parameters (including `activation_weight`, `bm25_weight`, etc.)
- `experiments/exp_bench_NNNN/hypothesis.txt` — rationale

## Rules
- NEVER modify `src/memory_decay/evaluator.py`, `graph.py`, `decay.py`, `server.py`, or `runner.py`
- NEVER modify the embedding cache (`cache/openai/`)
- NEVER modify MemoryBench code (`~/memorybench/`)
- NEVER modify the dataset
- Each experiment is independent — always start from fresh server state
- Be creative with decay formulas and params, but respect the interface contract
- Track what you've tried to avoid repeating failed approaches

## Escalation Rule

If the best next move requires changing files outside the allowed search surface:
1. Do not make the change
2. Record why the loop seems insufficient
3. Stop and ask for a human decision
