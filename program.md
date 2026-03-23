# Auto-Research Loop: MemoryBench-Driven Memory Decay Optimization

## Goal
Achieve ≥70% accuracy on all three MemoryBench benchmarks (LongMemEval, LoCoMo, ConvoMem) by optimizing decay functions and retrieval parameters.

**Current baseline** (experiments/best):
- LongMemEval: 55% (11/20)
- LoCoMo: 15% (3/20)
- ConvoMem: 95% (19/20)
- Composite bench_score: 0.51

**Target**: bench_score ≥ 0.70, with each benchmark ≥ 70%.

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

**Known issues from baseline analysis (2026-03-23):**
- LoCoMo 15% is lowest — decay suppresses old memories via `activation_weight=0.35`. Old memories with low retrieval_score get penalized in ranking.
- LongMemEval 55% — room for improvement in retrieval quality
- ConvoMem 95% — already near-optimal

**Key parameters that affect benchmark performance:**
- `activation_weight`: controls `retrieval_score^weight` penalty in ranking. Lower = less decay impact on retrieval. Currently 0.35.
- `bm25_weight`: lexical keyword matching. Currently 0.0 (disabled). Can help or hurt depending on weight.
- `bm25_candidates`: how many candidates to fetch for BM25 re-ranking. Currently 30.
- `lambda_fact` / `lambda_episode`: base decay speed. Currently 0.05 / 0.2.
- `floor_max`: minimum activation floor. Currently 0.45.
- `assoc_boost`: spreading activation. Currently 0.0.

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

**Current focus: LoCoMo** (15% → 70% target). This is the weakest benchmark.
Other benchmarks will be validated after LoCoMo reaches ≥70%.

**Stage A (quick screen)**: 10 questions × LoCoMo only (~1 min).

```bash
# 1. Kill old server and start new one
pkill -f "memory_decay.server" 2>/dev/null; sleep 1
.venv/bin/python -m memory_decay.server \
  --port 8100 --cache-dir cache/openai \
  --embedding-provider openai --embedding-api-key "$OPENAI_API_KEY" \
  --experiment-dir experiments/exp_bench_NNNN &
sleep 3

# 2. Stage A: LoCoMo only, 10 questions, gpt-4o-mini
cd ~/memorybench
OPENAI_API_KEY="$OPENAI_API_KEY" bun run src/index.ts run \
  -p memory-decay -b locomo -j gpt-4o-mini -m gpt-4o-mini \
  -r exp_bench_NNNN-locomo -l 10 --force
```

**Stage B (confirmation)**: only if Stage A shows improvement, run 20 questions × all 3 benchmarks with gpt-4o:
```bash
for BENCH in longmemeval locomo convomem; do
  OPENAI_API_KEY="$OPENAI_API_KEY" bun run src/index.ts run \
    -p memory-decay -b $BENCH -j gpt-4o -m gpt-4o \
    -r exp_bench_NNNN-stageB-$BENCH -l 20 --force
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

**Stage A metric**: LoCoMo accuracy (target ≥ 70%).
Current baseline: 15% (3/20) with 20 questions, 30% (3/10) with 10 questions.

### 6. Judge

**Stage A**: Compare LoCoMo accuracy with best.
- If locomo_acc > best_locomo_acc: proceed to **Stage B** (20 questions × all 3 benchmarks with gpt-4o)
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
