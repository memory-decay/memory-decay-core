# Auto-Research Loop: Dual-State Memory Decay Exploration

## Goal
Discover better dual-state memory dynamics by iterating: hypothesize -> implement -> test -> judge.

## Operating Principle

The human defines the closed loop. The AI agent works only inside the allowed search surface.

- The loop itself is fixed by humans
- Evaluation, datasets, bootstrap artifacts, and experiment protocol are fixed
- The fixed protocol now includes the dual-state memory architecture
- The agent may explore only the allowed algorithm and weight space inside that architecture
- If an apparent improvement requires changing the loop, evaluator, or datasets, stop and report it instead of modifying them

## Fixed Dual-State Architecture

The current protocol uses two state variables per memory:

- `storage_score`: decays over time and is used for threshold gating in evaluation
- `retrieval_score`: decays over time and is used for similarity ranking and activation-weighted retrieval

Compatibility note:

- `activation_score` is retained as a compatibility alias of `retrieval_score`
- Future experiments should reason in terms of `storage_score` and `retrieval_score`, not a single activation value

Protocol semantics:

- Threshold-dependent recall, precision, MRR, and correlation use `storage_score`
- Similarity ranking uses `retrieval_score`
- Standard scheduled reactivation policies (`random`, `scheduled_query`, `scheduled_query_all`, `scheduled_query_plus_test`) boost both states unless explicitly changed by humans
- `retrieval_consolidation` behavior is controlled by `params.json` via `retrieval_consolidation_mode`

Allowed retrieval consolidation modes:

- `activation_and_stability`: boosts retrieval and reinforces stability
- `retrieval_only`: boosts retrieval and reinforces stability without directly raising storage
- `stability_only_direct`: reinforces stability only, without directly raising retrieval or storage

## Preflight (run once before the loop if missing)

If `cache/embeddings.pkl` does not exist or needs rebuilding, run:

1. Convert source data (if not done):
   ```bash
   PYTHONPATH=src:. uv run python scripts/convert_longmemeval.py
   ```

2. Build embedding cache:
   ```bash
   PYTHONPATH=src uv run python -m memory_decay.cache_builder \
     --dataset data/longmemeval.jsonl --output cache --backend gemini
   ```

**Dataset**: LongMemEval (ICLR 2025), 500 questions, 5432 memory nodes, Gemini embeddings.
**Baseline**: `experiments/exp_lme_0000` (default exponential decay, overall=0.0374).

## Protocol (each cycle)

### 1. Read State
- Read `experiments/history.jsonl` for previous experiment results
- Read `experiments/best/decay_fn.py` for the current best function
- Read `experiments/best/results.json` for the current best scores
- If no `experiments/` directory exists, create it and run the baseline first

**Memory chain** (M27-style cross-session learning):
- Read `memory_chain/failure_patterns.md` for accumulated failure analysis and convergence diagnosis
- Read the last 5 `memory_chain/round_NNNN.md` files (by number) for recent self-criticism and next-step directions
- Use these to avoid repeating failed approaches and to inform the current hypothesis

### 2. Analyze & Hypothesize
Based on previous results **and memory chain feedback**, form a hypothesis for improvement.

**Memory chain check** (mandatory before hypothesizing):
- Does `failure_patterns.md` already list this approach as non-viable?
- Do recent round files show self-criticism that warns against this direction?
- If yes to either: choose a different direction. Do not repeat known failures.

Consider:
- Which thresholds have low recall? Is storage decaying too aggressively or not selectively enough?
- Is retrieval ranking strong enough even when storage becomes selective?
- Is the decay too fast or too slow for facts vs episodes?
- Is the gap between storage and retrieval state helping or hurting selectivity?
- Would a different `retrieval_consolidation_mode` improve the storage/retrieval tradeoff?
- Are `retrieval_boost`, `activation_weight`, and the importance-scaled boost parameters amplifying the right memories?
- Could a different mathematical form (hyperbolic, stretched exponential,
  logarithmic saturation) perform better?
- Are impact and stability modifiers being used effectively?
- What do the self-criticism notes from recent rounds suggest as unexplored territory?

### 3. Write Experiment Files
Create `experiments/exp_lme_NNNN/` (next sequential number) with:

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

Interpretation:
- The fixed engine applies this scalar decay function independently to `storage_score` and `retrieval_score`
- Experiment-local `decay_fn.py` should not try to implement its own second state variable
- The return value must remain monotone non-increasing for pure decay

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
  "assoc_boost": 0.0,
  "retrieval_boost": 0.10,
  "retrieval_consolidation_mode": "activation_and_stability"
}
```
(reinforcement/stability params are used by the simulation loop, not the decay function)
(`activation_weight` controls how strongly retrieval_score influences similarity ranking: score = cosine_sim * retrieval_score^weight)
(`assoc_boost` enables spreading-activation retrieval: score *= (1 + assoc_boost * mean_neighbor_activation))
(`retrieval_boost` controls the direct retrieval-state bump used by retrieval_consolidation modes that still boost retrieval)
(`retrieval_consolidation_mode` selects how successful recall reinforces memory state)

Common optional experiment-local params:
```json
{
  "importance_scaled_boost": true,
  "importance_boost_min_scale": 0.52,
  "importance_boost_max_scale": 1.0,
  "test_reactivation_start_tick": 40,
  "test_reactivation_interval": 10
}
```

**hypothesis.txt** — One paragraph explaining what you're trying and why.

### 4. Run Experiment
```bash
PYTHONPATH=src uv run python -m memory_decay.runner experiments/exp_lme_NNNN --cache cache
```

### 5. Read Results
Read `experiments/exp_lme_NNNN/results.json`. Key metrics:
- `overall_score`: main metric = 0.40 * retrieval_score + 0.35 * forgetting_score + 0.25 * plausibility_score
- `retrieval_score`: 0.55 * recall_mean + 0.45 * mrr_mean — measures "do you remember what you should?"
- `forgetting_score`: 1 - mean(storage_score of non-target memories) — measures "do you forget what you should?" Higher = better selective forgetting. Creates tension with retrieval: boosting all storage helps retrieval but hurts forgetting.
- `plausibility_score`: correlation between activation scores and actual retrievability (allows negative)
- `non_target_mean_storage`: raw mean storage of non-target memories; diagnostic for forgetting_score
- `mrr_mean`: Mean Reciprocal Rank across thresholds
- `precision_strict`: exact-match precision only; diagnostic
- `precision_lift`: diagnostic only (not used in overall_score)
- `precision_associative`: diagnostic precision where associated neighbors also count
- `similarity_recall_rate`: threshold-free similarity retrieval rate; diagnostic only
- `threshold_discrimination`: spread of thresholded recall across the fixed threshold sweep; in the dual-state protocol this reflects storage-thresholded recall
- `storage_std`, `storage_iqr`, `storage_gini`: spread of storage state at the final tick
- `retrieval_std`, `retrieval_iqr`, `retrieval_gini`: spread of retrieval state at the final tick
- `activation_std`, `activation_iqr`, `activation_gini`: backward-compatible aliases of the storage spread metrics
- `status`: "completed" or "validation_failed"

NOTE: Scores from experiments before exp_lme_0218 used a different formula and are not directly comparable.

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
r = run_kfold(Path('experiments/exp_lme_NNNN'), k=5, cache_dir=Path('cache'))
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
r = run_kfold(Path('experiments/exp_lme_NNNN'), k=5, cache_dir=Path('cache'))
Path('experiments/exp_lme_NNNN/cv_results.json').write_text(json.dumps(r, indent=2))
"
```

**Git commit format (Lore)** — only on improvement:
```
exp_lme_NNNN: overall 0.24→0.31 (+0.07), CV 0.22→0.25 via <short description>

<what changed in the decay function and why>

Rejected: <alternative tried earlier> | <why it didn't work>
Confidence: <high | medium | low>
Scope-risk: narrow
Tested: 200-tick simulation, threshold sweep [0.2,0.3,0.4,0.5], 5-fold CV
Directive: <any warning for future experiments>
```

### 7. Write Memory Chain Round

After judging, create a memory chain round file for M27-style cross-session learning.

**Determine the round number**: read `memory_chain/memory_index.jsonl`, find the highest `round` value, and use `round + 1`. If the file doesn't exist, start at round 0.

**Create `memory_chain/round_NNNN.md`** with this template:

```markdown
# Memory Chain — Round NNNN

## Experiment: exp_lme_XXXX
**Date**: YYYY-MM-DD
**Parent**: [round_PPPP.md](round_PPPP.md)

## Scores
| Metric | Value |
|--------|-------|
| overall_score | X.XXXX |
| retrieval_score | X.XXXX |
| plausibility_score | X.XXXX |
| recall_mean | X.XXX |
| mrr_mean | X.XXX |
| precision_lift | X.XXX |

## Hypothesis
<what was tried and why — copy from hypothesis.txt>

## Self-Criticism
<M27 core: analyze WHY this experiment succeeded or failed>
- What specific mechanism caused the score change?
- Did the hypothesis hold? If not, what was the actual failure mode?
- What does this reveal about the search surface structure?

## Decisions Made
- <concrete decision: e.g., "Jost power=4.0 is near-optimal, further increases not viable">

## What To Avoid
- <specific approaches that this experiment proves should NOT be retried>

## Next Step Direction
<concrete direction for the next experiment, informed by self-criticism>
```

**Append to `memory_chain/memory_index.jsonl`** (one line, no duplicates):
```json
{"round": NNNN, "experiment": "exp_lme_XXXX", "next_direction": "<short summary>", "timestamp": "<ISO 8601>"}
```

**Important**: Do NOT re-append the entire history on session start. Only append the single new round entry.

**Update `memory_chain/failure_patterns.md`** if a new failure pattern is discovered:
- Add the pattern to the relevant section
- Update the convergence diagnosis if the failure changes the structural understanding
- This file is cumulative — append new insights, don't rewrite existing analysis

### 8. Record
Append one line to `experiments/history.jsonl`:
```json
{"exp": "exp_lme_NNNN", "overall": 0.74, "retrieval": 0.71, "plausibility": 0.81, "cv_mean": 0.68, "cv_std": 0.03, "status": "improved", "hypothesis": "short summary"}
```
Include `cv_mean` and `cv_std` when CV was run (Stage B). Omit them for rejected experiments (Stage A only).

Operational note:
- `experiments/history.jsonl` is append-only session history, not the canonical source for rerun state
- Never rerun an existing `experiments/exp_lme_NNNN/` directory in place, because that can overwrite `results.json` and make the directory disagree with prior history
- If an experiment must be rerun intentionally, do it only with an explicit overwrite action and record that decision separately

### 9. Repeat or Stop
Continue to next cycle unless:
- 20 cycles completed this session
- 20+ consecutive experiments with no improvement (convergence)
- 10+ consecutive validation failures

## Baseline (first run only)
If `experiments/best/` doesn't exist:
1. Create `experiments/exp_lme_0000/` with the default exponential decay
2. Run it as the baseline
3. Set it as `experiments/best/`

## Important: Simulation Characteristics

- The simulation with `scheduled_query` reactivation policy is **fully deterministic** — changing `--seed` does not alter results. The seed only affects the `random` reactivation policy.
- Therefore, **multi-seed runs are not meaningful** for statistical validation.
- **K-fold cross-validation** (varying the train/test split) is the correct way to assess robustness. Use `memory_decay.cross_validator.run_kfold()`.
- Prior finding (memories_500 dataset, needs revalidation on LongMemEval): `assoc_boost=2.0` (exp_0338) showed CV=38% instability and scored 0.076 on CV vs 0.252 for `assoc_boost=0` (exp_0315, CV=4.8%). The fixed-split gain was overfitting.

## Rules
- During ordinary experiment loops, NEVER modify `src/memory_decay/evaluator.py`, `src/memory_decay/graph.py`, `src/memory_decay/decay.py`, `src/memory_decay/main.py`, or `src/memory_decay/runner.py`
- These files define the fixed dual-state protocol
- NEVER modify the dataset or cache
- NEVER modify `main.py` benchmark protocol during the loop
- NEVER modify files under `outputs/pre_program_pipeline/`
- NEVER modify bootstrap scripts to improve scores mid-loop
- Each experiment is independent — always start from fresh graph state
- Be creative with decay formulas and experiment-local params, but respect the interface contract
- Track what you've tried to avoid repeating failed approaches

## Allowed Search Surface

The agent is allowed to change only these experiment-local files:

- `experiments/exp_lme_NNNN/decay_fn.py`
- `experiments/exp_lme_NNNN/params.json`
- `experiments/exp_lme_NNNN/hypothesis.txt`

Interpretation:

- `decay_fn.py` = scalar decay-law slot applied to both storage and retrieval by the fixed engine
- `params.json` = weight/parameter slot, including dual-state policy params such as `retrieval_consolidation_mode`
- `hypothesis.txt` = rationale only

Everything else is part of the closed loop and should be treated as fixed.

## Escalation Rule

If the best next move appears to require changing any file outside the allowed search surface:

1. Do not make the change
2. Record why the loop seems insufficient
3. Stop and ask for a human decision on whether to widen the search space

Examples that require escalation:

- changing which state (`storage_score` vs `retrieval_score`) an evaluator metric reads
- changing how `query_by_similarity()` scores candidates
- introducing a third state variable or changing the fixed dual-state protocol
- modifying scheduled reactivation behavior outside experiment-local params
