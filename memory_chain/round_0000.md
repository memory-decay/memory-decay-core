# Memory Chain — Round 0000

## Experiment: exp_lme_0000
**Date**: 2026-03-20
**Parent**: none

## Scores
| Metric | Value |
|--------|-------|
| overall_score | 0.0374 |
| retrieval_score | 0.0401 |
| plausibility_score | 0.5589 |
| recall_mean | 0.068 |
| mrr_mean | 0.043 |
| precision_lift | 0.000 |
| similarity_recall_rate | 0.111 |

## Hypothesis
LongMemEval baseline — fresh start with published benchmark dataset and Gemini embeddings.

## Self-Criticism
- This is the initial baseline on a new dataset. No comparison to make yet.
- Key question: does the embedding ceiling differ from the old dataset?
- similarity_recall_rate=0.111 is much lower than old dataset (0.402) — need to investigate whether this is dataset difficulty or embedding quality.
- With 5432 memories and top_k=5, random baseline recall would be ~0.001, so 0.111 shows the embeddings have signal.

## Decisions Made
- Switched from memories_500.jsonl (synthetic Korean) to LongMemEval (published English)
- Switched from ko-sroberta to gemini-embedding-001

## What To Avoid
- Nothing yet — clean slate on new data

## Next Step Direction
Run Jost+sigmoid (old best) to see if it transfers, then explore from there.
