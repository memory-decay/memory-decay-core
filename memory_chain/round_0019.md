# Memory Chain — Round 0019

## Experiments: exp_lme_0218-0229 (H1-H7 hypothesis sweep)
**Date**: 2026-03-22
**Parent**: [round_0018.md](round_0018.md)

## Results Summary
| Exp | Hypothesis | CV | Retrieval | Forget | Plausibility | Key Finding |
|-----|-----------|-----|---------|--------|--------------|-------------|
| 0222 | H2b: 3.5x faster decay | **0.6757** | 0.614 | 0.632 | 0.836 | **WINNER — +7.6% over baseline** |
| 0225 | H4: act=0.3 + scale=0.5 | 0.6486 | 0.614 | 0.504 | 0.907 | Good retrieval, no forgetting |
| 0221 | H2a: 2x faster decay | 0.6465 | 0.602 | 0.561 | 0.837 | Moderate improvement |
| 0205 | baseline | 0.6282 | 0.588 | 0.504 | 0.866 | — |
| 0218 | H1a: scale=0.5 only | 0.6243 | 0.585 | 0.504 | 0.855 | Scale alone doesn't help forgetting |
| 0228 | H6: retrieval_only | 0.4641 | 0.520 | 0.504 | 0.318 | Kills plausibility — storage essential |
| 0220 | H1c: scale=0.0 | 0.4655 | 0.527 | 0.315 | — | Zero storage kills plausibility |

## Critical Insight
**Base decay rate is the real lever, not storage_scale.**

storage_scale lowering (H1a/b/c) does NOT force non-target forgetting — non-targets simply don't receive consolidation anyway, so adjusting the scale only affects targets. The actual mechanism for getting non-targets to decay faster is raising `lambda_fact` and `lambda_episode`.

Dose-response confirmed: 2x → 3.5x decay → better performance monotonically.

## Decisions Made
- Accept exp_lme_0222 (3.5x decay) as new best — CV=0.6757 (+34% over round_0018's best 0.5035)
- Update `best` symlink to exp_lme_0222
- Storage scale below 1.0 does NOT improve forgetting — retract H1 hypothesis direction
- retrieval_only mode destroys plausibility — storage reinforcement is essential, not optional

## What To Avoid
- storage_scale tuning below 1.0 as a forgetting lever — proven ineffective
- retrieval_only mode — it destroys plausibility entirely
- Claims of "search converging" — round_0018 was wrong at CV=0.5035

## Next Step Direction
1. Fine-tune decay rate between 2x and 3.5x (experiments 0230–0234) to map full dose-response and find optimal multiplier
2. If 4.0x+ extends the curve, continue that direction
3. Combine H2b's fast decay with H4's activation_weight tuning (0225 had good retrieval but no forgetting — combine with faster decay?)
