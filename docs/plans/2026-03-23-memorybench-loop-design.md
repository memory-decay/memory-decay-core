# MemoryBench-Driven Auto-Improvement Loop

## Goal

Replace the internal evaluator (proxy metrics on LongMemEval fixed split) with MemoryBench end-to-end evaluation across 3 benchmarks, using the pre-built OpenAI embedding cache.

## Objective Function

```
bench_score = 0.50 × longmemeval_acc + 0.30 × locomo_acc + 0.20 × convomem_acc
```

LongMemEval gets highest weight: hardest benchmark, most questions (500), best signal.

## Two-Stage Validation

| Stage | Questions/dataset | Total | Cost/iter | Purpose |
|-------|------------------|-------|-----------|---------|
| A (screen) | 20 | 60 | ~$0.60 | Quick reject bad experiments |
| B (confirm) | 50 | 150 | ~$1.50 | Confirm improvement is real |

Stage B only runs if Stage A shows improvement over current best.

## Pipeline Per Iteration

```
1. AutoImprover proposes params.json changes (or decay_fn.py changes)
2. Write experiment files to experiments/exp_bench_NNNN/
3. Restart memory-decay server with new experiment dir
4. Run MemoryBench CLI for 3 benchmarks (Stage A: 20 questions each)
5. Parse report.json → compute bench_score
6. If bench_score > best: run Stage B (50 questions each)
7. If Stage B confirms: update experiments/best symlink
8. Record in history.jsonl
```

## Server Configuration

```bash
.venv/bin/python -m memory_decay.server \
  --port 8100 \
  --cache-dir cache/openai \
  --embedding-provider openai \
  --embedding-api-key $OPENAI_API_KEY \
  --experiment-dir experiments/exp_bench_NNNN
```

## MemoryBench CLI

```bash
cd ~/memorybench && bun run src/index.ts run \
  -p memory-decay -b locomo -j gpt-4o -m gpt-4o \
  -r exp_bench_NNNN-locomo -l 20 --force
```

## Key Constraints

- `decay_fn.py` interface unchanged (compute_decay signature)
- `params.json` structure unchanged
- Embedding cache is fixed (cache/openai/, text-embedding-3-small)
- Server code (graph.py, decay.py, server.py) is FIXED
- Only the decay function and its parameters are in the search space

## Files

- `src/memory_decay/bench_evaluator.py` — orchestrates server restart + MemoryBench runs + score parsing
- `scripts/run_bench_loop.py` — main loop with AutoImprover integration
