# Human-like Memory Decay System

A research project modeling human memory with decay functions, re-activation, and impact factors using graph-based associations and LLM-driven auto-improvement. The system simulates how memories fade over time following configurable decay curves (exponential and power law), how accessing related memories can re-activate fading ones, and how emotional significance affects retention — then uses an AI agent to iteratively optimize the model's parameters.

## Key Research Questions

1. **Exponential vs power law**: which decay function better matches human-like recall patterns?
2. **Re-activation effect**: how much does accessing associated memories affect preservation of related memories?
3. **Fact vs episode decay**: is there a meaningful difference in decay rates between factual knowledge and episodic memories?
4. **Impact and retention**: do high-impact (emotionally significant) memories stay recallable longer, and by how much?

## Architecture Overview

```
┌─────────────────────────────┐
│   SyntheticDataGenerator    │  (Anthropic API)
│   Generate memory items &   │
│   recall test queries       │
└─────────────┬───────────────┘
              │ JSONL dataset
              ▼
┌─────────────────────────────┐
│        MemoryGraph          │  (NetworkX + sentence-transformers)
│   Nodes: memory items       │
│   Edges: associations       │
│   Embeddings: MiniLM-L6-v2  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│        DecayEngine          │
│   Exponential / Power Law   │
│   Impact modifier           │
│   Re-activation cascades    │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│         Evaluator           │
│   5-metric composite score  │
│   Periodic recall tests     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│       AutoImprover          │  (Anthropic API)
│   Analyze results           │
│   Propose parameter changes │
│   Iterative improvement     │
└─────────────────────────────┘
```

## Tech Stack

- **Python** — core language
- **NetworkX** — graph-based memory representation
- **sentence-transformers** — embedding generation (all-MiniLM-L6-v2, 384-dim)
- **Anthropic API** — synthetic data generation and auto-improvement agent
- **NumPy** — numerical computation for decay functions and evaluation metrics

## Project Structure

```
src/memory_decay/
  graph.py          # MemoryGraph: NetworkX-based memory store
  decay.py          # DecayEngine: decay functions and re-activation
  evaluator.py      # Evaluator: multi-metric recall evaluation
  data_gen.py       # SyntheticDataGenerator: Anthropic-powered data creation
  auto_improver.py  # AutoImprover: LLM-driven parameter optimization
```

## Getting Started

```bash
pip install -e .
```

See [design spec](docs/superpowers/specs/2026-03-17-memory-decay-design.md) for full architecture and implementation details.
