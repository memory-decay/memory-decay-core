# Auto-Research Loop Design

> Based on Karpathy's autoresearch pattern: fixed evaluation + open search space + meta-program

## Core Idea

Claude Code acts as the orchestrator in a closed loop, proposing new decay functions,
running experiments, and deciding keep/discard. Python code is a pure simulation runner.

## Architecture

```
Claude Code (orchestrator)
  |-- reads history.jsonl + best/decay_fn.py
  |-- writes new decay_fn.py + params.json
  |-- runs: python -m memory_decay.runner experiments/exp_NNNN
  |-- reads results.json
  |-- judges keep/discard
  |-- repeats (max 20 cycles/session)

/loop 30m program.md  -->  autonomous daemon
```

## Boundary (autoresearch mapping)

| autoresearch   | memory-decay              | status   |
|----------------|---------------------------|----------|
| prepare.py     | evaluator.py + graph.py   | FIXED    |
| train.py       | function slot (decay_fn)  | OPEN     |
| program.md     | program.md                | NEW      |
| result.tsv     | experiments/exp_NNN/      | NEW      |

## Function Slot Interface

```python
def compute_decay(
    activation: float,   # current activation_score (0-1)
    impact: float,       # memory importance (0-1)
    stability: float,    # stability score (0-1)
    mtype: str,          # "fact" | "episode"
    params: dict,        # freely defined parameters
) -> float:              # new activation_score (0-1)
```

## Experiment Storage

```
experiments/
  best/                 -> symlink to best experiment
  history.jsonl         -> one line per experiment
  exp_0001/
    decay_fn.py
    params.json
    hypothesis.txt
    results.json
```

## Caching

- cache/embeddings.pkl: pre-computed embeddings for all 500 memories + queries
- cache/dataset.json: parsed dataset
- cache/test_queries.json: (query, expected_id) pairs

## Stopping Conditions

- Convergence: 20 consecutive experiments with no improvement
- Validation failure: 10 consecutive failures
- User interrupt (Ctrl+C or session end)

## Decisions

- Search space: parameters + algorithms (decay formulas)
- LLM: Claude Code (no separate API calls)
- Experiment tracking: snapshots always, git on improvement only
- Execution: /loop 30m, ~20 cycles per session
