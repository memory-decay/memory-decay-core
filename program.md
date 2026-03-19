# Auto-Research Loop: Memory Decay Function Exploration

## Goal
Discover better decay functions by iterating: hypothesize -> implement -> test -> judge.

## Operating Principle

The human defines the closed loop. The AI agent works only inside the allowed search surface.

- The loop itself is fixed by humans
- Evaluation, datasets, bootstrap artifacts, and experiment protocol are fixed
- The agent may explore only the allowed algorithm and weight space
- If an apparent improvement requires changing the loop, evaluator, or datasets, stop and report it instead of modifying them

## Preflight (run once before the loop if missing)

If `outputs/pre_program_pipeline/suite_summary.json` does not exist, run:

```bash
PYTHONPATH=src uv run python scripts/run_pre_program_pipeline.py --embedding-backend local
```

This bootstrap step does all required pre-program work:

- human calibration on `data/human_reviews_smoke.jsonl`
- baseline vs calibrated comparison on `data/memories_50.jsonl`
- baseline vs calibrated comparison on `data/memories_500.jsonl`
- artifact export under `outputs/pre_program_pipeline/`

After it completes, read:

- `outputs/pre_program_pipeline/suite_summary.json`
- `outputs/pre_program_pipeline/memories_500/comparison_summary.json`

Use the `memories_500` comparison as the canonical pre-loop reference point.

Treat all files under `outputs/pre_program_pipeline/` as read-only reference artifacts during the loop.

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
  "stability_cap": 1.0,
  "activation_weight": 0.5,
  "assoc_boost": 0.0
}
```
(reinforcement/stability params are used by the simulation loop, not the decay function)
(`activation_weight` controls how strongly activation influences similarity ranking: score = cosine_sim * activation^weight)
(`assoc_boost` enables spreading-activation retrieval: score *= (1 + assoc_boost * mean_neighbor_activation))

**hypothesis.txt** — One paragraph explaining what you're trying and why.

### 4. Run Experiment
```bash
PYTHONPATH=src uv run python -m memory_decay.runner experiments/exp_NNNN --cache cache
```

### 5. Read Results
Read `experiments/exp_NNNN/results.json`. Key metrics:
- `overall_score`: main metric (retrieval_score * (0.85 + 0.15 * plausibility_score))
- `retrieval_score`: 0.40 * recall_mean + 0.30 * mrr_mean + 0.30 * precision_lift
- `plausibility_score`: 0.50 * correlation + 0.50 * smoothness (correlation allows negative)
- `precision_lift`: how much the decay engine pruned distractors above the baseline (null_precision)
- `mrr_mean`: Mean Reciprocal Rank across thresholds
- `precision_strict`: exact-match precision only; used to calculate precision_lift
- `precision_associative`: diagnostic precision where associated neighbors also count
- `similarity_recall_rate`: threshold-free similarity retrieval rate; diagnostic only
- `status`: "completed" or "validation_failed"

### 6. Judge

**Stage A — Quick screen** (single fixed split):
Compare `overall_score` with `experiments/best/results.json`:
- If score is **not higher**: record in history, move on (skip Stage B)
- If **validation failed**: record error, adjust next hypothesis

**Stage B — Cross-validation gate** (only if Stage A passed):
Run 5-fold CV to confirm the improvement is robust, not overfitting to the fixed test split:
```bash
PYTHONPATH=src uv run python -c "
from pathlib import Path
from memory_decay.cross_validator import run_kfold
r = run_kfold(Path('experiments/exp_NNNN'), k=5, cache_dir=Path('cache'))
print(f'CV overall: {r[\"mean\"][\"overall_score\"]:.4f} +/- {r[\"std\"][\"overall_score\"]:.4f}')
cv_pct = r['std']['overall_score'] / r['mean']['overall_score'] * 100
print(f'CV%: {cv_pct:.1f}%')
for i, f in enumerate(r['fold_scores']):
    print(f'  fold {i}: {f[\"overall_score\"]:.4f}')
"
```

Compare CV mean with current best's CV mean (stored in `experiments/best/cv_results.json`).

**Decision rules:**
- CV mean improved AND CV% < 30%: **accept** — update symlink, commit
- CV mean improved but CV% >= 30%: **unstable** — record but do not update best
- CV mean not improved: **reject** — the fixed-split gain was overfitting

On accept, save CV results alongside the experiment:
```bash
# Save CV results for future comparison
PYTHONPATH=src uv run python -c "
import json
from pathlib import Path
from memory_decay.cross_validator import run_kfold
r = run_kfold(Path('experiments/exp_NNNN'), k=5, cache_dir=Path('cache'))
Path('experiments/exp_NNNN/cv_results.json').write_text(json.dumps(r, indent=2))
"
```

**Git commit format (Lore)** — only on improvement:
```
exp_NNNN: overall 0.24→0.31 (+0.07), CV 0.22→0.25 via <short description>

<what changed in the decay function and why>

Rejected: <alternative tried earlier> | <why it didn't work>
Confidence: <high | medium | low>
Scope-risk: narrow
Tested: 200-tick simulation, threshold sweep [0.2,0.3,0.4,0.5], 5-fold CV
Directive: <any warning for future experiments>
```

### 7. Record
Append one line to `experiments/history.jsonl`:
```json
{"exp": "exp_NNNN", "overall": 0.74, "retrieval": 0.71, "plausibility": 0.81, "cv_mean": 0.68, "cv_std": 0.03, "status": "improved", "hypothesis": "short summary"}
```
Include `cv_mean` and `cv_std` when CV was run (Stage B). Omit them for rejected experiments (Stage A only).

Operational note:
- `experiments/history.jsonl` is append-only session history, not the canonical source for rerun state
- Never rerun an existing `experiments/exp_NNNN/` directory in place, because that can overwrite `results.json` and make the directory disagree with prior history
- If an experiment must be rerun intentionally, do it only with an explicit overwrite action and record that decision separately

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

## Important: Simulation Characteristics

- The simulation with `scheduled_query` reactivation policy is **fully deterministic** — changing `--seed` does not alter results. The seed only affects the `random` reactivation policy.
- Therefore, **multi-seed runs are not meaningful** for statistical validation.
- **K-fold cross-validation** (varying the train/test split) is the correct way to assess robustness. Use `memory_decay.cross_validator.run_kfold()`.
- Prior finding: `assoc_boost=2.0` (exp_0338) showed CV=38% instability and scored 0.076 on CV vs 0.252 for `assoc_boost=0` (exp_0315, CV=4.8%). The fixed-split gain was overfitting.

## Rules
- NEVER modify evaluator.py, graph.py, or runner.py
- NEVER modify the dataset or cache
- NEVER modify `main.py` benchmark protocol during the loop
- NEVER modify files under `outputs/pre_program_pipeline/`
- NEVER modify bootstrap scripts to improve scores mid-loop
- Each experiment is independent — always start from fresh graph state
- Be creative with decay formulas but respect the interface contract
- Track what you've tried to avoid repeating failed approaches

## Allowed Search Surface

The agent is allowed to change only these experiment-local files:

- `experiments/exp_NNNN/decay_fn.py`
- `experiments/exp_NNNN/params.json`
- `experiments/exp_NNNN/hypothesis.txt`

Interpretation:

- `decay_fn.py` = algorithm slot
- `params.json` = weight/parameter slot
- `hypothesis.txt` = rationale only

Everything else is part of the closed loop and should be treated as fixed.

## Escalation Rule

If the best next move appears to require changing any file outside the allowed search surface:

1. Do not make the change
2. Record why the loop seems insufficient
3. Stop and ask for a human decision on whether to widen the search space
