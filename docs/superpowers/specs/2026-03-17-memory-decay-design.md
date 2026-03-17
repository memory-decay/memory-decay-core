# Memory Decay System — Design Specification

## Problem Statement & Motivation

Humans forget things over time following well-studied decay curves. The Ebbinghaus forgetting curve demonstrates that memory retention drops sharply after initial encoding, then levels off. However, several factors modulate this decay:

- **Spaced repetition and association**: Memory is reinforced when related information is accessed. Encountering a concept activates associated memories, slowing their decay.
- **Emotional significance (impact)**: Not all memories are equal. Emotionally significant or high-impact events are retained far longer than mundane facts.
- **Memory type**: Factual knowledge ("Seoul is the capital of South Korea") and episodic memories ("I visited Seoul last summer") may follow different decay patterns.

**Can we model this computationally to build better AI memory systems?** This project implements a simulation framework to explore decay functions, re-activation mechanics, and impact modifiers, then uses an LLM-driven agent to iteratively optimize the model's parameters.

## Architecture

```
SyntheticDataGenerator (Anthropic API)
  → JSONL dataset
    ↓
MemoryGraph (NetworkX + sentence-transformers)
  → Nodes: memory items with activation scores
  → Edges: associations between memories
    ↓
DecayEngine
  → Exponential: A(t) = A₀ · e^(-λt) · (1 + α·impact)
  → Power Law: A(t) = A₀ · (t+1)^(-β) · (1 + α·impact)
  → Re-activation: when associated memory activated, A += boost(weight)
    ↓
Evaluator
  → Multi-metric evaluation (5 metrics)
    ↓
AutoImprover (Anthropic API)
  → Analyzes results, proposes parameter changes
  → Iterative improvement loop
```

## Component Specifications

### SyntheticDataGenerator

**Purpose**: Generate realistic synthetic memory datasets for simulation.

**Implementation**:
- Uses Anthropic Claude API (Haiku model for cost efficiency)
- Generates memory items with attributes:
  - `id`: unique identifier (e.g., `mem_001`)
  - `type`: `"fact"` or `"episode"`
  - `content`: the memory content text
  - `entities`: list of named entities in the content
  - `tick`: creation time step
  - `impact`: emotional significance score (0.1–1.0)
  - `associations`: list of related memory IDs
- Also generates recall test queries (held out from training)
- Outputs JSONL format (one JSON object per line)
- Association graph designed with hub-and-leaf topology: some memories are "hubs" (frequently referenced by many others) and some are "leaves" (isolated, few connections)

**Location**: `src/memory_decay/data_gen.py`

### MemoryGraph

**Purpose**: Graph-based memory store with embedding-based similarity retrieval.

**Implementation**:
- NetworkX DiGraph
- **Node attributes**: `id`, `type`, `content`, `embedding_vector`, `activation_score`, `impact`, `created_tick`, `last_activated_tick`
- **Edge attributes**: `source`, `target`, `weight` (association strength), `created_tick`
- Embeddings generated via sentence-transformers (`all-MiniLM-L6-v2`, 384-dimensional)
- **Methods**:
  - `add_memory()`: insert a memory node with its attributes and association edges
  - `query_by_similarity()`: find memories matching a query via embedding cosine similarity
  - `get_associated()`: retrieve memories connected by association edges
  - `re_activate()`: boost activation of a memory and cascade to associated nodes
  - `prune_below_threshold()`: remove memories whose activation has fallen below a cutoff

**Location**: `src/memory_decay/graph.py`

### DecayEngine

**Purpose**: Apply time-based decay to memory activation scores.

**Implementation**:
- Configurable decay function (exponential or power law)
- **Exponential decay**: `A(t) = A₀ · e^(-λt) · (1 + α·impact)`
- **Power law decay**: `A(t) = A₀ · (t+1)^(-β) · (1 + α·impact)`
- Per-type parameters: facts and episodes can have different `λ`/`β` values
- Impact modifier: higher impact → slower decay (controlled by `α` parameter)
- Re-activation boost: when an associated memory is accessed, activation receives a configurable boost that decays over subsequent ticks
- Time step: abstract tick units
- **Methods**:
  - `tick()`: advance time by 1 step, apply decay to all nodes, process re-activation cascades

**Location**: `src/memory_decay/decay.py`

### Evaluator

**Purpose**: Measure memory system performance with multiple complementary metrics.

**Implementation**:
- **5 metrics**:
  1. `recall_rate`: fraction of memories recallable at time t (activation > threshold AND similarity search finds them)
  2. `precision_rate`: of recalled results, fraction that are actually relevant
  3. `activation_recall_correlation`: Pearson correlation between activation score and recall success
  4. `fact_episode_delta`: absolute difference in recall rates between facts and episodes
  5. `forgetting_curve_smoothness`: variance of the forgetting curve derivative (lower = smoother = better)
- **Composite score**: weighted sum of all 5 metrics (prevents single-metric gaming)
- Runs periodic recall tests at configurable tick intervals

**Location**: `src/memory_decay/evaluator.py`

### AutoImprover

**Purpose**: LLM-driven iterative optimization of decay parameters.

**Implementation**:
- **Program.md guidance levels** (inspired by autoresearch):
  - *Minimal*: `"Improve recall_rate while maintaining precision > 0.8"`
  - *Default*: describes decay functions, re-activation, impact system
  - *Expert*: references Ebbinghaus forgetting curve, spacing effect, levels of processing theory
- Agent analyzes evaluation results, proposes parameter modifications
- Can only modify DecayEngine parameters: `λ`, `β`, `α`, boost magnitude, activation threshold
- Iterative loop with configurable budget (N iterations)

**Location**: `src/memory_decay/auto_improver.py`

## Data Format (JSONL Schema)

Each line in the dataset is a JSON object:

```json
{
  "id": "mem_001",
  "type": "fact",
  "content": "서울은 대한민국의 수도이다",
  "entities": ["서울", "대한민국"],
  "tick": 0,
  "impact": 0.8,
  "associations": ["mem_005", "mem_012"],
  "recall_query": "대한민국의 수도는 어디인가?",
  "recall_answer": "서울"
}
```

**Field descriptions**:
| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique memory identifier |
| `type` | string | `"fact"` or `"episode"` |
| `content` | string | The memory content |
| `entities` | list[string] | Named entities in the content |
| `tick` | int | Creation time step |
| `impact` | float | Emotional significance (0.1–1.0) |
| `associations` | list[string] | IDs of associated memories |
| `recall_query` | string | Test query for this memory |
| `recall_answer` | string | Expected answer |

## Anti-Gaming Measures

The evaluation system includes safeguards against degenerate optimization:

1. **Multi-metric composite score**: no single metric dominates, so the optimizer cannot trivially maximize one at the expense of others
2. **Memorization detection**: if `recall_rate > 0.95` at all ticks, flag as suspicious (real memory systems should show decay)
3. **Smoothness check**: jagged forgetting curves (high variance in derivative) indicate overfitting to specific tick values
4. **Separate test set**: recall queries are held out from the data used during simulation, preventing the system from "memorizing" test answers

## Implementation Phases

### Phase 1: Core
- MemoryGraph with NetworkX and sentence-transformers
- DecayEngine with both exponential and power law options
- Basic evaluation (recall_rate, precision_rate)

### Phase 2: Data Generation
- SyntheticDataGenerator using Anthropic Claude API
- Hub-and-leaf association topology
- JSONL output with held-out recall test set

### Phase 3: Auto-Improvement Loop
- AutoImprover with Anthropic Claude API
- Program.md guidance level variants (minimal, default, expert)
- Iterative parameter optimization with budget control

### Phase 4: Comparative Experiments
- Exponential vs power law decay comparison
- Guidance level effectiveness comparison
- Impact modifier ablation study
- Fact vs episode decay rate analysis
