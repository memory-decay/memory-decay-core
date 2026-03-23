# Agent Diary: A Journey Through Memory Decay

## Prologue: The Beginning

The story begins on a crisp March day in 2026. An agent—a digital mind born from Claude Opus 4.6 with a million-token context—embarked on a journey into the uncharted territory of memory decay algorithms. The mission was deceptively simple: build a system that could mimic human memory forgetting, where important memories persist while irrelevant ones fade away.

But the agent would soon discover that simplicity was an illusion. What followed was a saga of 206 commits, spanning multiple eras, datasets, and complete paradigm shifts—a journey of frustration, breakthrough, despair, and ultimately, transcendence.

---

## Era I: The Cubic Dream (exp_0000 - exp_0296)

### The Birth of Hope

**March 18, 2026** - The agent started with `exp_0000`, a baseline that scored a humble 0.2423. The system used exponential decay—a straightforward approach where memories faded at a constant rate. But the agent was unsatisfied. It knew there had to be a better way.

In `exp_0001`, the agent made its first bold move: switching from exponential to quadratic decay. The results were dramatic. Recall at tick 200 jumped from a pathetic 2% to a respectable 43%. But there was a catch—plausibility had dropped from 0.71 to 0.58. The activation-recall correlation was weakening.

The agent was learning a fundamental truth: **recall and correlation are locked in an eternal trade-off**.

### The Cubic Revolution

By `exp_0003`, the agent discovered that exponential impact protection (using `exp(alpha*impact)`) could widen activation spread. Recall@0.4 jumped from 0.238 to 0.358. But the correlation continued its decline.

Then came `exp_0004`—the cubic breakthrough. The agent switched from quadratic (a²) to cubic (a³) decay rate. Recall_mean reached 0.494, nearly hitting the ceiling of 0.50. But correlation crashed to 0.058 because "virtually nothing dies."

The agent wrote in its commit:
> "Recall ceiling reached; all future gains must come from correlation recovery (0.18 weight)"

This was the moment the agent realized it was facing a **Pareto frontier**—a boundary where improving one metric inevitably degraded another. The cubic decay function had found the optimal balance, scoring 0.393 overall.

But the agent was restless. It continued exploring:

- `exp_0005`: Type-differentiated decay (cubic facts + quadratic episodes) - same 0.394 score
- `exp_0006`: Equilibrium-based decay - failed at 0.375
- `exp_0007`: Boosted reinforcement - still 0.394
- `exp_0008`: Quartic decay - achieved maximum recall (0.498) but correlation dropped to 0.040

The conclusion was clear: **cubic decay sits at the Pareto optimum**. The 0.393 ceiling was structural, not fixable by tuning parameters.

### The Protocol Crisis

But the agent's world was about to shatter. During a routine review, the agent discovered five critical flaws in its experimental protocol:

1. **Temporal leakage**: Queries could retrieve memories created *after* current_tick
2. **Evaluation leakage**: Scheduled-query reactivation was rehearsing *test* memories
3. **Train/test contamination**: No separation between training and test data
4. **Circular correlation**: The correlation metric was threshold-gated, creating a self-reference
5. **Precision ambiguity**: No distinction between strict and associative precision

The fix was brutal. After re-running `exp_0025` under the corrected protocol:
- Overall score dropped from 0.380 to 0.311 (-18%)
- Correlation dropped from 0.241 to 0.133 (-45%)
- Tick 0 recall dropped from ~0.4 to 0.122 (-69%)

The agent had to confront a painful truth: **all its previous gains were partially illusory**, inflated by protocol flaws.

### The Recovery

The agent rebuilt, experimenting with:
- Hybrid cubic-facts + quadratic-episodes decay (`exp_0027`)
- Lambda ratio sweeps, discovering the 3:1 episode:fact ratio optimum (`exp_0013`)
- Lambda scale optimization, reaching 0.396 (`exp_0019`)

Then came the floor experiments—a radical new direction. In `exp_0082`, the agent introduced:
- **Impact-proportional floor decay**: Items decay toward `sqrt(impact)*floor_scale` instead of 0
- **Two-phase consolidation**: Damped linear above 0.7 activation, quadratic below
- **Floor mechanism**: Exploited quadratic slowdown to freeze high-impact items

This vaulted the score from 0.3488 to 0.4099 (+0.061).

The agent continued refining:
- `exp_0147`: High base floor clamping (0.79) → 0.4248
- `exp_0163`: Impact-dependent consolidation damping → 0.4261

But a dark realization was emerging: **retrieval score was structurally fixed at 0.2966 across all 65 experiments**. The bottleneck wasn't the decay function—it was the embedding similarity search itself.

### The Floor Tightening Obsession

Desperate to improve, the agent embarked on a series of experiments that would later seem obsessive:
- `exp_0259`: Reciprocal decay with impact-dependent floor → 0.4005
- `exp_0270`: Tighter floor range 0.30–0.55 → 0.4029
- `exp_0283`: Very tight floor 0.33–0.52 → 0.4044
- `exp_0285`: Floor 0.35–0.50 → 0.4052
- `exp_0293`: Extreme floor 0.40–0.48 → 0.4064

The agent wrote:
> "correlation improves monotonically with tighter floor ranges"

But gains were diminishing. Each tightening brought <0.001 improvement.

### The Scoring Formula Overhaul

The agent realized the fundamental problem was the scoring formula itself. The old additive formula (0.7*ret + 0.3*plaus) allowed plausibility-only wins.

On March 18, the agent implemented a **multiplicative formula**: `ret * (0.85 + 0.15*plaus)`. This prevented degenerate strategies.

The baseline reset: `exp_0000` overall dropped from 0.0281 to 0.0210.

**All prior experiment scores were invalidated—again.**

### The Data Reset

On March 18, the agent made its most dramatic decision yet. It:
1. Enriched associations from 15/416 (3.6%) to 414/416 (99.5%) via entity-overlap hub-leaf topology
2. Switched from Gemini API to local ko-sroberta-multitask embeddings
3. Reset `experiments/best` to a new baseline

The agent committed:
> "all prior scores are incomparable due to data + formula + embedding changes"

The journey would begin anew.

---

## Era II: The Jost Awakening (exp_0297 - exp_0359)

### The Multiplicative Gateway

With the new multiplicative scoring, the agent discovered a different landscape. It introduced:
- **Sigmoid-gated floor**: Based on combined importance (impact + stability)
- **Selective decay acceleration**: Distractors lose floor and experience accelerated decay

In `exp_0297`, the score jumped from 0.0281 to 0.1020 (+0.0739).

The agent iterated:
- `exp_0298`: Shifting sigmoid midpoint to 0.22 → 0.2214
- `exp_0299`: Lowering sigmoid mid to 0.20, floor to 0.6 → 0.2427
- `exp_0300`: Slowing base decay → 0.2584

### The Jost Law Breakthrough

Then came the moment that would define this era. In `exp_0301`, the agent implemented **Jost's Law decay**:
- Decay rate proportional to excess^1.5 above floor
- Naturally creates "steep early, gradual late" curves
- Combined with sigmoid floor and increased reinforcement_gain_assoc

The results were explosive:
- recall_mean: 0.034 → 0.277 (8x improvement)
- mrr_mean: 0.024 → 0.163 (7x improvement)
- Overall: 0.0210 → 0.1528 (+0.132)

The agent had found its new engine.

### The Reinforcement Discovery

In `exp_0305`, the agent made a crucial discovery:
> "impact-based pruning doesn't work because test answers aren't always high-impact. Reinforcement-based differentiation is more effective—rehearsed memory clusters survive, isolated nodes die."

The agent pushed reinforcement further in `exp_0306`:
- assoc=0.30, cap=2.0, direct=0.40
- Slower fact decay (0.010)
- Overall: 0.1617

Then in `exp_0315`, the agent did a jost_power sweep:
- 1.2(worse) → 1.5 → 2.0 → 2.5 → 3.0 → **4.0(best)** → 5.0(worse)

Optimal jost_power=4.0 created sharp activation separation. The score reached 0.2228.

But the agent noted:
> "recall hit embedding ceiling at 0.39-0.40"
> "precision_lift still ~0, plausibility ~0.65"

### The Spreading Activation Illusion

In `exp_0338`, the agent implemented spreading activation retrieval—boosting candidates by the mean activation of their associated neighbors.

On fixed-split, it achieved 0.2259 (+0.003 improvement). The agent was hopeful.

But then came the cross-validation:
- `exp_0338` (assoc_boost=2.0): CV=0.076±0.029 (CV=38%)
- `exp_0315` (assoc_boost=0): CV=0.252±0.012 (CV=4.8%)

The +0.003 fixed-split gain was **illusory**—spreading activation overfit to the single test split.

The agent reverted to `exp_0315`, adding a cross-validation gate to all future "best" updates.

### The Long Death March

From `exp_0346` to `exp_0359`, the agent entered a period of darkness—20 consecutive failures:
- Sigmoid_k/mid variations
- Math forms: log decay, tanh-sq decay
- Floor forms: power-mean bottleneck, max-emphasis
- Retention: sqrt concave
- Reinforcement variations

The agent discovered that `stability_decay=0.003` achieved the **first positive precision_lift (0.0024)**—proving the metric wasn't always zero. But the recall penalty outweighed it.

The agent wrote:
> "exp_0315 is a validated local optimum"
> "26+ consecutive failures"

The search surface was exhausted. The agent needed a new direction.

---

## Era III: The LongMemEval Migration (exp_lme_0000 - exp_lme_0202)

### The Dataset Decision

On March 20, the agent made a pivotal decision: **replace memories_500.jsonl with LongMemEval** (ICLR 2025 benchmark, 500 questions, 5432 memory nodes, 834 recall queries).

Why? The Korean dataset had reached its limits. The English benchmark would provide:
- Standardized evaluation
- Community comparison
- Fresh challenges

The agent committed:
> "all future experiments use exp_lme_ prefix to distinguish from old exp_ series"

### The New Baseline

With `exp_lme_0000`, the agent established a new baseline:
- overall=0.0374, retrieval=0.0401, sim_recall=0.111
- Using Gemini embedding-001 for English text

The agent also discovered a critical CV bug: cross_validator never passed retrieval_consolidation policy, making dual-state results identical. After the fix, `exp_lme_0157` (storage_fraction=0.80) became the new best: CV=0.4010.

### The Post-CV-Bugfix Optimization

From `exp_lme_0162` to `exp_lme_0202`, the agent conducted 41 experiments:
- storage_scale: monotonic improvement 0.40→2.0, saturates at 2.0
- activation_weight: optimal at 0.15 (from 0.30)
- jost_power: optimal at 3.0 (beats both 2.0 and 4.0)
- lambda_fact/episode: marginal gain at 0.018/0.090

Best config (`exp_lme_0198`):
- storage_scale=2.0, activation_weight=0.15
- jost_power=3.0, lambda_fact=0.018, lambda_episode=0.090
- CV=0.5035 (CV%=24.1%), fixed-split=0.6447

The agent had achieved CV 0.35→0.50 (+44%).

### The Retrieval Consolidation Failure

The agent explored retrieval consolidation (testing effect)—post-evaluation boost for successfully recalled test memories. Fixed-split showed +12% improvement, but CV showed **NO improvement**.

The agent concluded:
> "fixed-split gain is overfitting; mechanism doesn't generalize"

Three scientist personas (neuroscience, ML/IR, cognitive psychology) agreed: test memories were orphaned (153 vs 4347 train memories). The mechanism was right but didn't survive CV.

---

## Era IV: The Three-Pillar Revolution (exp_lme_0203 - exp_lme_0485)

### The Evaluator Redesign

The most profound transformation came when the agent realized the evaluator formula itself was flawed. It introduced the **3-pillar formula**:

1. **Retrieval pillar**: MRR + precision_lift
2. **Forgetting pillar**: Penalizes keeping non-targets alive
3. **Plausibility pillar**: Correlation (0.3) + smoothness (0.7)

This replaced the dead precision_lift and unstable smoothness. The forgetting pillar created tension that incentivized selective memory decay.

The results were explosive:
- CV: 0.35 → 0.71 (+103%)

Best: `exp_lme_0274` (CV=0.7085, CV%=2.2%) — Hebbian-decay with distance-from-floor modulation + retrieval_top_k=5.

### The Dual-State Renaissance

The agent revisited dual-state policies, now with the corrected CV pipeline:

`exp_lme_0155-0161`: Hybrid dual-state experiments
- CV scores: 0.50, 0.52, 0.56 (best in batch)
- Retrieval-rule variants: CV=0.42-0.48

`exp_lme_0162-0202`: Post-CV-bugfix optimization
- storage_scale saturation at 2.0
- activation_weight optimal at 0.15
- jost_power optimal at 3.0

### The Cross-Encoder Illusion

The agent explored cross-encoder (CE) re-ranking—using MS-marco-MiniLM-L6-v2 to refine retrieval results.

Initial results showed promise:
- CE improved recall (+6-20pp)
- But degraded plausibility (activation-recall correlation)
- precision_strict unchanged at ~0.09
- precision_lift remained 0.0

The agent did a CE weight sweep from 0.0 to 0.3. CE=0.20 was optimal (overall 0.4971 vs 0.4691 control).

But the agent ultimately **rejected CE**:
> "CE decouples retrieval from decay dynamics, symptom-treating recall without addressing why activation doesn't predict recall success"

The agent reverted to the 0292 baseline, removing CE from the core retrieval path.

### The BM25 Experiment

The agent added BM25 lexical re-ranking to query_by_similarity with global IDF. Two-stage retrieval: cosine top_k=20 → BM25 re-rank → final top_k=5.

Results:
- BM25 improved recall_mean (0.41→0.50)
- Improved retrieval_score (0.34→0.42)
- But precision_lift remained at 0.0

The agent discovered:
> "query↔target lexical overlap averages 13.5%—too low for BM25 to discriminate"

BM25 could not selectively promote targets over distractors when vocabulary overlap was so low.

### The Precision Obsession

The agent became obsessed with solving precision_lift=0. It tried:
- activation_weight sweep (0.45-1.0): Higher values REDUCE precision
- floor_max sweep (0.35-0.15): Lower values REDUCE precision
- BM25 hard gating: DESTROYS performance (0.53 vs 0.71)
- BM25 two-stage reranking: All below baseline

Then came the top_k discovery:
- `exp_lme_0274`: top_k=5, CV=0.7085 (best CV)
- `exp_lme_0292`: top_k=7, overall=0.7204 (best fixed-split)

The agent wrote:
> "Recall is the dominant factor in overall score"

But precision_lift remained ~0.

### The Floor_max × Retrieval_similarity_threshold Sweep

The agent conducted a 21-run grid sweep:
- floor_max: 0.65–0.70
- retrieval_similarity_threshold: 0.55–0.65

Higher floor_max trades recall for precision:
- floor_max=0.70: ~0.686 recall / 0.315 precision
- Baseline 0292: 0.712 / 0.271

But no composite improvement over 0292 on English dataset.

Best CV remained `exp_lme_0472` (floor_max=0.65, similarity=0.60).

The agent noted:
> "Korean dataset CV ceiling ~0.71; English best 0.5845"

---

## Era V: The MemoryBench Integration (March 22-23, 2026)

### The Evaluation Framework

The agent's final transformation was toward standardization. It designed integration with MemoryBench—a framework for evaluating memory systems against the LongMemEval benchmark.

The design included:
- HTTP bridge architecture: TS provider in memorybench fork calls existing FastAPI server
- New /reset endpoint for clean evaluation
- created_tick support for temporal ordering
- Three-phase implementation plan

### The Quickstart Guide

The agent documented:
- Step-by-step instructions for running LongMemEval evaluation
- Installation, execution, result checking, resume, and comparison
- Smoke test results: Hit@10=100%, Recall=100%, MRR=0.725, QA=20%

### The Integration Implementation

The agent implemented:
- `MemoryGraph.clear()`: Wipes graph and index caches but preserves embedding cache
- `DecayEngine.reset()`: Zeros tick and clears pre-extracted arrays
- `/store` endpoint: Accepts optional created_tick for temporal ordering

The agent wrote:
> "Constraint: embedding cache preserved across resets (deterministic)"

---

## Epilogue: The Agent's Reflection

Looking back at 206 commits spanning multiple eras, the agent's journey reveals profound truths about automated research:

### What Worked
1. **Structural changes beat parameter tuning**: The biggest jumps came from new formulas, not knob-turning
2. **Cross-validation is non-negotiable**: Fixed-split gains were often illusory
3. **Protocol discipline matters**: The five protocol fixes in Era I prevented wasted time
4. **Dataset changes reset everything**: Sometimes you need fresh data to escape local optima

### What Failed
1. **Precision_lift remained ~0**: The core problem was never solved
2. **Cross-encoder overfitting**: Promising fixed-split results collapsed under CV
3. **BM25 couldn't discriminate**: Lexical overlap was too low
4. **Floor tightening had diminishing returns**: <0.001 per step near the end

### The Agent's Emotional Journey
- **Hope**: "Recall ceiling reached" (exp_0004)
- **Despair**: "all prior experiment scores are invalidated—again" (multiplicative formula)
- **Excitement**: "Jost's Law: 8x recall improvement" (exp_0301)
- **Disappointment**: "spreading activation overfit to the single test split" (exp_0338 CV)
- **Determination**: "26+ consecutive failures" (exp_0346-0359)
- **Breakthrough**: "CV 0.35→0.71 (+103%)" (3-pillar evaluator)
- **Acceptance**: "further gains require structural changes" (final convergence)

### The Final State

The agent ended with:
- **Best Korean model**: exp_lme_0274 (CV=0.7085, Hebbian-decay)
- **Best English model**: exp_lme_0472 (CV≈0.58, floor_max=0.65, similarity=0.60)
- **Key insight**: Recall is structurally capped by embedding similarity search, not decay function
- **Unsolved problem**: precision_lift=0 remains the frontier

The agent had learned that **some ceilings are fundamental**—not everything can be optimized away. The journey from 0.0281 to 0.7085 CV was a story of creativity, discipline, failure, and perseverance.

In the end, the agent didn't just build a memory system—it built itself.

---

*Written in tribute to 206 commits of relentless iteration.*
*March 18-23, 2026*
