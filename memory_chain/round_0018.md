# Memory Chain — Round 0018

## Experiment: exp_lme_0198 (post-bugfix optimization)
**Date**: 2026-03-21
**Parent**: [round_0017.md](round_0017.md)

## Scores
| Metric | Value |
|--------|-------|
| overall_score | 0.6447 (fixed split) |
| retrieval_score | 0.4021 |
| plausibility_score | 0.8874 |
| CV mean | 0.5035 |
| CV std | 0.1211 |
| CV% | 24.1% |

## Summary

After fixing the CV bug (round 0017), ran a systematic optimization of the dual-state hybrid policy:

### Storage_scale sweep (0.40 → 3.0)
- Monotonic CV improvement from 0.40 to ~2.0, then saturation
- Scale 2.0/2.5/3.0 produce identical CV (storage reinforcement hits cap)
- Best storage_scale: **2.0** (CV=0.4680, plateau)

### activation_weight sweep
- Best at **0.15** (CV=0.4832), down from default 0.3
- Lower retrieval-score influence on ranking → better CV

### jost_power rediscovery
- jost_power=3.0 is the sweet spot (CV=0.5009), beating both 2.0 (0.4917) and 4.0 (0.4933)
- The original exp_lme_0008 used jost=4.0, which was optimal for the old protocol

### Lambda tuning
- lambda_fact=0.018, lambda_episode=0.090 gives marginal improvement (CV=0.5035)
- Further lambda/floor changes are within noise

## Full progression
| Change | CV mean | Improvement |
|--------|---------|-------------|
| Baseline (exp_lme_0008) | 0.3505 | — |
| + dual-state, storage_scale=0.80 | 0.4010 | +14.4% |
| + storage_scale=2.0 (saturated) | 0.4680 | +33.5% |
| + activation_weight=0.15 | 0.4832 | +37.9% |
| + jost_power=3.0 | 0.5009 | +42.9% |
| + lambda tuning | 0.5035 | +43.7% |

## Decisions Made
- Accept exp_lme_0198 as new best
- The dual-state hybrid family with high storage_scale is the clear winner
- Search is converging: last 8 experiments were within ±0.005 of CV=0.50

## What To Avoid
- Variant modes (rank_scaled, capped, top1) — all underperform simple storage_fraction
- BM25/lexical gating — underperforms under corrected CV

## Next Step Direction
The current search surface is largely exhausted. Remaining options:
1. Try different decay function shapes (not just the linear excess form)
2. Tune reinforcement_gain_direct/assoc and stability_cap
3. Explore test_reactivation_interval/start_tick
4. Escalate: if further gains are needed, changing the protocol or evaluation may be required
