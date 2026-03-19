# Convergence Summary: Auto-Research Loop as of 2026-03-19

## Decision

Treat [`experiments/exp_0315`](../experiments/exp_0315/) as the best result inside the current closed-loop search surface.

- Fixed-split score: `overall_score=0.2228`
- Fixed-split retrieval: `0.2351`
- Fixed-split plausibility: `0.6511`
- 5-fold CV mean overall: `0.2515`
- 5-fold CV std overall: `0.0120`
- CV variation: about `4.8%`

This is the most robust result found within the allowed search surface in [`program.md`](../program.md).

## Why `exp_0315` Is the Termination Point

The late-stage search tested multiple structural alternatives after `exp_0315` and none beat it on Stage A:

- [`experiments/exp_0339`](../experiments/exp_0339/): Gompertz floor, `overall_score=0.2152`
- [`experiments/exp_0340`](../experiments/exp_0340/): decoupled impact/stability, `overall_score=0.1506`
- [`experiments/exp_0341`](../experiments/exp_0341/): bounded interaction term, `overall_score=0.1989`
- [`experiments/exp_0342`](../experiments/exp_0342/): type-specific Jost curvature, `overall_score=0.1871`
- [`experiments/exp_0343`](../experiments/exp_0343/): piecewise Jost near the floor, `overall_score=0.1989`
- [`experiments/exp_0344`](../experiments/exp_0344/): dual-sigmoid floor, `overall_score=0.1916`
- [`experiments/exp_0345`](../experiments/exp_0345/): nonlinear retention weighting, `overall_score=0.1992`

The search pattern is consistent:

- Coupled impact/stability remains better than separated roles
- The original sigmoid floor remains better than Gompertz or dual-sigmoid variants
- A single global Jost curve remains better than the tested piecewise or type-specific variants
- Recent alternatives cluster below `exp_0315`, mostly in the `0.19x` range

## Interpretation of `exp_0338`

[`experiments/exp_0338`](../experiments/exp_0338/) showed a fixed-split gain (`overall_score=0.2259`), but it is not the accepted best result.

Per the loop rules in [`program.md`](../program.md):

- `exp_0338` corresponds to `assoc_boost=2.0`
- It was later judged unstable under cross-validation
- The accepted robust best remains `exp_0315`

So `exp_0338` should be treated as a useful overfitting case, not the final model choice.

## What Changed Operationally

To prevent future record corruption:

- [`src/memory_decay/runner.py`](../src/memory_decay/runner.py) now refuses to overwrite an existing `results.json` unless rerun is explicitly forced
- [`tests/test_runner.py`](../tests/test_runner.py) covers that protection
- [`program.md`](../program.md) now states that `history.jsonl` is append-only and existing `exp_NNNN` directories should not be rerun in place

## What Happens Next

If the goal is a better score than `exp_0315`, the next move is no longer inside the current search surface.

Human decision required:

1. Widen the search surface
   - Example: new embedding backend or retrieval representation
2. Change the evaluation setup
   - Example: different benchmark split, dataset, or protocol
3. Accept convergence and move on
   - Use `exp_0315` as the final in-loop result

## Recommended Project Position

Use this wording in reports and discussions:

> Within the current closed-loop protocol, the Jost-plus-sigmoid formulation converged at `exp_0315`. Later structural variations failed to improve on it robustly, and the strongest fixed-split challenger (`exp_0338`) was rejected due to instability.
