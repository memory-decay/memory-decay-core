# Human-Memory-Inspired Memory Decay System

Research code for a graph-based memory simulator that separates short-term `activation` from long-term `stability`, compares exponential vs power-law forgetting, and supports reinforcement-aware reactivation plus LLM-driven parameter search.

The project is intentionally framed as a **human-memory-inspired AI memory model**, not a literal model of human cognition. Current experiments still use synthetic Korean memory datasets, and real-user validation remains future work.

## What Changed In The Reinforcement Redesign

- Memory nodes now track `activation_score`, `stability_score`, `retrieval_count`, and `last_reinforced_tick`
- Re-activation distinguishes `direct` recall from weaker `cascade` reinforcement
- Decay slows down as a function of both `impact` and accumulated `stability`
- Evaluation is no longer anchored to a single activation threshold
- The main summary now separates:
  - `retrieval_score`
  - `plausibility_score`
  - `overall_score`

## Architecture

```
SyntheticDataGenerator (OpenAI API)
  -> JSONL dataset
    -> MemoryGraph (NetworkX + sentence-transformers)
      -> Nodes: activation, stability, impact, retrieval metadata
      -> Edges: weighted associations
        -> DecayEngine
          -> Exponential / power-law forgetting
          -> Impact-aware slowing
          -> Stability-aware slowing
          -> Direct / cascade reinforcement
            -> Evaluator
              -> threshold_sweep()
              -> score_summary()
              -> recall / precision / plausibility diagnostics
                -> AutoImprover (OpenAI API)
```

## Key Parameters

- `lambda_fact`, `lambda_episode`: base exponential decay rates
- `beta_fact`, `beta_episode`: base power-law decay rates
- `alpha`: impact modifier strength
- `stability_weight`: how much accumulated stability slows future decay
- `stability_decay`: how quickly reinforcement fades
- `reinforcement_gain_direct`: stability increase for direct recall
- `reinforcement_gain_assoc`: weaker stability increase for associated recall
- `stability_cap`: upper bound on reinforcement strength

## Evaluation Model

`Evaluator.threshold_sweep()` measures recall and precision over the fixed grid `[0.2, 0.3, 0.4, 0.5]`.

`Evaluator.score_summary()` reports:

- `retrieval_score = 0.7 * recall_mean + 0.3 * precision_mean`
- `plausibility_score = 0.6 * corr_score + 0.4 * smoothness_score`
- `overall_score = 0.7 * retrieval_score + 0.3 * plausibility_score`

`composite_score()` remains as a backward-compatible alias to `overall_score`.

## Canonical Data And Scripts

- Canonical checked-in dataset: `data/memories_500.jsonl`
- Auto-improvement entrypoint: `scripts/run_auto_improve.py`
- Visualization entrypoint: `scripts/visualize.py`
- Current research draft: `docs/final-report.md`
- Embedding backend default: `auto` (`GEMINI_API_KEY`가 있으면 Gemini, 없으면 local sentence-transformers)

The visualization script now supports:

- `recall_curves.png`
- `precision_curves.png`
- `combined_comparison.png`
- `threshold_sensitivity.png`
- `reinforcement_ablation.png`
- `auto_improvement_rounds.png`

## Development

```bash
uv sync --extra dev
PYTHONPATH=src uv run pytest -q
```

Example simulation run:

```bash
PYTHONPATH=src uv run python -m memory_decay.main \
  --dataset data/memories_500.jsonl \
  --decay-type exponential \
  --embedding-backend auto \
  --reactivation-policy scheduled_query \
  --total-ticks 200 \
  --eval-interval 20
```

See [design spec](docs/superpowers/specs/2026-03-17-memory-decay-design.md) for the original architecture notes and [final report](docs/final-report.md) for the current research narrative.
