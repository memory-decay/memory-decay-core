# Evaluator 3-Pillar Redesign

**Date**: 2026-03-22
**Status**: Approved
**Scope**: `src/memory_decay/evaluator.py`, `program.md`

## Problem

The current evaluator has two structural bottlenecks that cap CV at ~0.51:

1. **precision_lift = 0 always** — 30% of retrieval_score is dead weight. `precision_lift = max(0, strict_precision - recall/top_k)`, but strict_precision equals the null baseline exactly because storage scores are too uniform to filter anything via thresholding.

2. **smoothness clips to 0 in 3/5 folds** — `smoothness_score = max(1.0 - variance * 10, 0.0)` is too aggressive. Fold-dependent recall jitter triggers the hard clip, causing plausibility to swing from 0.43 to 0.93 across folds. This is the dominant source of CV%.

More fundamentally: the evaluator only tests "do you remember what you should?" — there's no cost to keeping everything alive, so the optimal strategy converges to "forget nothing" (storage_scale=2.0 confirming this).

## Design: 3-Pillar Scoring

Replace the current 2-component formula with three orthogonal pillars:

### Pillar 1: Retrieval (weight 0.40)

```python
retrieval_score = 0.55 * recall_mean + 0.45 * mrr_mean
```

- Removes precision_lift (replaced by forgetting pillar)
- recall: "did you find the target?"
- mrr: "how high did you rank it?"

### Pillar 2: Forgetting (weight 0.35)

```python
forgetting_score = 1.0 - mean(storage_score of non-target memories)
```

- non-target = all typed graph nodes created before current tick, excluding test query expected_ids
- Penalizes keeping irrelevant memories alive
- Creates tension: boosting all storage helps retrieval but hurts forgetting
- Optimal strategy: selectively retain targets, let non-targets decay

Edge case: if non-target set is empty, return 0.5 (neutral).

### Pillar 3: Plausibility (weight 0.25)

```python
plausibility_score = corr_score
```

- correlation between activation/storage scores and actual retrievability
- Removes smoothness entirely (fold variance source, low discriminating power)
- corr_score has fold spread of only 0.04 — very stable

### Overall

```python
overall_score = 0.40 * retrieval_score + 0.35 * forgetting_score + 0.25 * plausibility_score
```

## Changes

### evaluator.py

1. Add `_forgetting_score(self, test_queries, current_tick)` method:
   - Extract target_ids from test_queries
   - Iterate graph nodes, collect storage_score of non-targets
   - Return `1.0 - mean(non_target_storage)`, or 0.5 if empty

2. Modify `score_summary()`:
   - New retrieval_score formula: `0.55 * recall_mean + 0.45 * mrr_mean`
   - New plausibility_score: `corr_score` only
   - Compute and include forgetting_score
   - New overall_score: 3-pillar formula
   - Keep old metrics in result dict as diagnostics (precision_lift, smoothness_score, etc.)
   - Add new fields: `forgetting_score`, `non_target_mean_storage`

### program.md

- Update overall_score, retrieval_score, plausibility_score descriptions
- Add forgetting_score description
- Note that old experiment scores are not comparable to new ones

### cross_validator.py

- Add `"forgetting_score"` to METRICS list
- No other changes needed (already calls score_summary with test_queries)

### No changes to

- runner.py, main.py, decay.py, graph.py
- decay_fn.py interface
- Dataset or cache
- CV acceptance criteria (CV% < 30%)

## After deployment

1. Re-evaluate exp_lme_0205 under new formula as baseline
2. Re-evaluate exp_lme_0008 (old single-score best) for comparison
3. Resume experiment loop — the forgetting pillar creates new gradient for decay function optimization
