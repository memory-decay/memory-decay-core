# Auto-Research Loop: Memory Decay Function Exploration

## Goal
Discover better decay functions by iterating: hypothesize -> implement -> test -> judge.

## Protocol (each cycle)

### 1. Read State
- Read `experiments/history.jsonl` for previous experiment results
- Read `experiments/best/decay_fn.py` for the current best function
- Read `experiments/best/results.json` for the current best scores
- If no `experiments/` directory exists, create it and run the baseline first

### 2. Analyze & Hypothesize
Based on previous results, form a hypothesis for improvement. Consider:
- Which thresholds have low recall? Can a different curve shape help?
- Is the decay too fast or too slow for facts vs episodes?
- Could a different mathematical form (hyperbolic, stretched exponential,
  logarithmic saturation) perform better?
- Are impact and stability modifiers being used effectively?

### 3. Write Experiment Files
Create `experiments/exp_NNNN/` (next sequential number) with:

**decay_fn.py** — Must follow this exact interface:
```python
def compute_decay(activation, impact, stability, mtype, params):
    """
    Args:
        activation: float 0-1, current activation score
        impact: float 0-1, memory importance
        stability: float 0-1, reinforcement stability
        mtype: "fact" or "episode"
        params: dict of tunable parameters
    Returns:
        float 0-1, new activation score (must be < activation for decay)
    """
    # Your implementation here
    ...
```

**params.json** — Parameters the function uses. Must include at minimum:
```json
{
  "lambda_fact": 0.02,
  "lambda_episode": 0.035,
  "stability_weight": 0.8,
  "stability_decay": 0.01,
  "reinforcement_gain_direct": 0.2,
  "reinforcement_gain_assoc": 0.05,
  "stability_cap": 1.0
}
```
(reinforcement/stability params are used by the simulation loop, not the decay function)

**hypothesis.txt** — One paragraph explaining what you're trying and why.

### 4. Run Experiment
```bash
PYTHONPATH=src uv run python -m memory_decay.runner experiments/exp_NNNN --cache cache
```

### 5. Read Results
Read `experiments/exp_NNNN/results.json`. Key metrics:
- `overall_score`: main metric (0.7 * retrieval + 0.3 * plausibility)
- `retrieval_score`: 0.7 * recall_mean + 0.3 * precision_mean (across thresholds)
- `plausibility_score`: 0.6 * correlation + 0.4 * smoothness
- `status`: "completed" or "validation_failed"

### 6. Judge
Compare `overall_score` with `experiments/best/results.json`:
- **Improved**: update `experiments/best/` symlink, git commit
- **No gain**: record in history, move on
- **Validation failed**: record error, adjust next hypothesis

### 7. Record
Append one line to `experiments/history.jsonl`:
```json
{"exp": "exp_NNNN", "overall": 0.74, "retrieval": 0.71, "plausibility": 0.81, "status": "improved", "hypothesis": "short summary"}
```

### 8. Repeat or Stop
Continue to next cycle unless:
- 20 cycles completed this session
- 20+ consecutive experiments with no improvement (convergence)
- 10+ consecutive validation failures

## Baseline (first run only)
If `experiments/best/` doesn't exist:
1. Create `experiments/exp_0000/` with the default exponential decay
2. Run it as the baseline
3. Set it as `experiments/best/`

## Rules
- NEVER modify evaluator.py, graph.py, or runner.py
- NEVER modify the dataset or cache
- Each experiment is independent — always start from fresh graph state
- Be creative with decay formulas but respect the interface contract
- Track what you've tried to avoid repeating failed approaches
