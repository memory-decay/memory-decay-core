# Eval v2 Design

## Goal

Replace the current mostly static retrieval leaderboard with an evaluation that answers three questions at once:

1. Does the model forget in a realistic time-graded way?
2. Does it separate targets from distractors instead of winning by broad similarity?
3. Does the improvement survive split variation, rather than only one fixed split?

This is a protocol redesign, not a continuation of the current closed loop. Results from Eval v2 should not be placed on the same leaderboard as the current Eval v1 experiments.

## Why Eval v1 Is No Longer Enough

The current loop converged at [`experiments/exp_0315`](../../experiments/exp_0315/), but the search exposed evaluation weaknesses:

- Fixed-split gains can be misleading. [`experiments/exp_0338`](../../experiments/exp_0338/) improved on the fixed split but was later rejected as unstable under cross-validation.
- Threshold metrics are often degenerate. In late experiments, recall and precision often remain effectively identical across the current threshold sweep, leaving `threshold_discrimination` at or near zero.
- `precision_lift` frequently collapses to zero, which hides meaningful ranking differences.
- Under `scheduled_query`, multi-seed variation is not a real uncertainty signal. Split variation is the meaningful robustness axis.

In short: Eval v1 is too easy to game with split-specific retrieval gains and too weak at distinguishing meaningful selectivity improvements from cosmetic score changes.

## Design Principles

- Make cross-validation the center of acceptance, not a late afterthought
- Measure forgetting over time, not just a final state snapshot
- Separate strict retrieval from associative retrieval
- Reward target selectivity under distractor pressure
- Penalize instability across folds and degenerate threshold behavior
- Keep the first version narrow enough to implement without rebuilding the whole project

## Recommended Eval v2 Structure

Eval v2 should score three primary dimensions:

1. `retention_auc`
2. `selectivity_score`
3. `robustness_score`

A fourth metric, `associative_score`, should remain diagnostic or lightly weighted until its behavior is better understood.

### 1. Retention

Purpose:
- Measure realistic forgetting across time rather than only final retrieval at one tick

Proposed task:
- Evaluate the same held-out query set at multiple delays, for example ticks `40, 80, 120, 160, 200`

Proposed metrics:
- strict recall at each delay
- strict `mrr_mean` at each delay
- optional type-specific curves for `fact` vs `episode`

Proposed aggregate:
- `retention_auc`: area under the strict retrieval curve across delays

Why:
- This rewards models that preserve useful memories over time without flattening into “everything survives forever”

### 2. Selectivity

Purpose:
- Distinguish true target retrieval from broad semantic matching

Proposed task:
- Add lure or interference probes built from near-neighbor distractors, paraphrase-adjacent distractors, or same-entity wrong memories

Proposed metrics:
- exact-target `precision@k` or `mrr`
- lure false-positive rate
- strict-minus-associative gap as a penalty when associative retrieval grows without strict retrieval support

Proposed aggregate:
- `selectivity_score = exact_target_quality - lure_penalty`

Why:
- Current `precision_lift` is often clipped to zero and loses information
- Selectivity is the missing dimension behind many false improvements

### 3. Robustness

Purpose:
- Prevent fixed-split wins from becoming accepted model changes

Proposed task:
- Use 5-fold CV as the primary experiment acceptance surface

Proposed metrics:
- fold mean
- fold standard deviation
- worst fold score
- paired fold deltas against the current champion

Proposed aggregate:
- `robustness_score` based on CV mean with an explicit variance penalty

Why:
- In the current protocol, CV is already the signal that separated `exp_0315` from the unstable `exp_0338`
- Eval v2 should formalize that instead of treating it as a secondary safety check

## Recommended Composite Score

Initial recommended weighting:

```text
eval_v2_score
= 0.45 * retention_auc
+ 0.35 * selectivity_score
+ 0.20 * robustness_score
```

Notes:
- `associative_score` is reported separately at first
- If associative retrieval later proves important, it can be reintroduced as a small positive term

## Acceptance Rule

Eval v2 should use a two-stage decision rule:

### Stage A: Screen

Use the existing fixed split only as a cheap screen.

Rule:
- If the candidate is not at least directionally better on the screen, stop early
- Do not accept any model based only on Stage A

### Stage B: Main Decision

Use 5-fold CV as the real acceptance gate.

A candidate is accepted only if all conditions hold:

1. CV mean `eval_v2_score` improves by at least `+0.01`
2. The one-sided 95% lower bound of the paired fold delta is greater than `0`
3. At least `4/5` folds improve over the current champion
4. No fold is worse than baseline by more than `0.02` on either retention or selectivity
5. Threshold/selectivity diagnostics are non-degenerate

This keeps the protocol practical while still blocking `exp_0338`-style overfitting.

## Metric Changes Relative to Eval v1

### Replace coarse threshold discrimination

Current issue:
- Four thresholds are too sparse and often collapse to a flat line

Change:
- Replace the current max-minus-min recall summary with a denser threshold curve summary
- Example outputs:
  - `threshold_auc`
  - low-vs-high threshold slope
  - optional threshold sensitivity penalty when the curve is nearly flat

### Replace clipped precision lift

Current issue:
- `precision_lift` often collapses to zero and hides useful variation

Change:
- Replace it with a selectivity metric that preserves negative and positive signal
- Candidate forms:
  - baseline-normalized strict precision gain without clipping
  - strict `precision@k`
  - strict `mrr` under lure pressure

### Demote seed-based uncertainty for `scheduled_query`

Current issue:
- multi-seed confidence intervals imply false robustness for a deterministic policy

Change:
- For `scheduled_query`, use split resampling and fold variation as the uncertainty source
- Keep seed sweeps only as diagnostics for non-deterministic policies

## Proposed Output Shape

Eval v2 results should include fields like:

```json
{
  "retention_auc": 0.41,
  "selectivity_score": 0.28,
  "robustness_score": 0.22,
  "eval_v2_score": 0.34,
  "cv": {
    "mean": 0.34,
    "std": 0.02,
    "worst_fold": 0.30,
    "fold_deltas": [0.01, 0.02, -0.00, 0.03, 0.01]
  },
  "retention_curve": {
    "40": 0.62,
    "80": 0.49,
    "120": 0.41,
    "160": 0.33,
    "200": 0.29
  },
  "selectivity": {
    "strict_mrr": 0.31,
    "lure_fp_rate": 0.08,
    "associative_gap": 0.04
  },
  "threshold_summary": {
    "threshold_auc": 0.27,
    "slope": -0.09
  }
}
```

The exact field names can change during implementation, but the structure should preserve these four layers:
- primary sub-scores
- CV summary
- retention curve
- selectivity diagnostics

## Scope Recommendation for Implementation

Implement Eval v2 in two phases.

### Phase 1: Minimal useful version

- Add multi-delay retention scoring
- Replace coarse threshold summary
- Replace clipped `precision_lift`
- Make CV the primary acceptance rule

This is the recommended first implementation.

### Phase 2: Richer memory realism probes

- Add lure/interference tasks
- Add spaced vs massed reactivation tests
- Add noisy cue/paraphrase probes

These should be deferred until the minimal version is working and producing stable rankings.

## Success Criteria

Eval v2 is successful if:

- `exp_0338`-style fixed-split gains no longer survive unless they generalize across folds
- The protocol can distinguish “better ranking/selectivity” from “same retrieval with cosmetic score changes”
- The project can state clearly whether a new candidate improves retention, selectivity, robustness, or some tradeoff between them
- The new leaderboard is stable enough that late-stage experiments no longer converge to indistinguishable scores due to metric collapse

## Recommendation

Adopt the recommended three-part Eval v2 design:

- retention over time
- selectivity under distractor pressure
- CV-centered robustness

This is the most direct response to the failure mode exposed by the current loop and is a better next move than immediately changing embeddings.
