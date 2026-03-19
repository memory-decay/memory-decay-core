# Memory Chain - Failure Pattern Analysis (LongMemEval)

## Overview

Fresh start with LongMemEval dataset (ICLR 2025). Previous analysis from memories_500.jsonl archived to `archive_memories500/`.

## What Worked (from prior dataset — needs revalidation)

- Jost Law decay with power=4.0 + sigmoid floor was best on old synthetic data
- To be confirmed on LongMemEval

## What Failed (from prior dataset — needs revalidation)

Previous failures may not apply to new dataset. Re-test before ruling out:
- Gompertz floor
- Decoupled impact/stability
- Memory-type-specific Jost curvature
- Dual-sigmoid floor

## New Findings

(To be populated as experiments run on LongMemEval)
