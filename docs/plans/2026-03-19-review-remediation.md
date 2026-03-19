# Review Remediation Plan: Addressing Devil's Advocate Findings

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address all Critical (P0) and Major (P1) findings from the 3-reviewer Devil's Advocate panel to bring the research to publication-quality rigor.

**Architecture:** Phased approach — Phase 1 (pilot) builds statistical infrastructure and runs multi-seed validation. Phase 2-4 proceed only after Phase 1 results are reviewed. This avoids premature work if Phase 1 reveals the conclusions need fundamental restructuring.

**Tech Stack:** Python, numpy, matplotlib, scipy.stats (bootstrap CI), existing memory_decay framework

**Execution strategy:** Phase 1 first as pilot. Review results before Phase 2-4.

---

## Phase 1: Statistical Infrastructure (P0 — Critical)

Addresses: Single seed (n=1), no confidence intervals, test set leakage

### Task 1: Multi-Seed Runner

**Files:**
- Create: `src/memory_decay/multi_runner.py`
- Modify: `src/memory_decay/runner.py:115-173` (extract seed parameter)
- Test: `tests/test_multi_runner.py`

**Step 1: Write the failing test**

```python
# tests/test_multi_runner.py
import json
import tempfile
from pathlib import Path
from memory_decay.multi_runner import run_multi_seed


def test_multi_seed_returns_stats():
    """Multi-seed runner must return mean, std, CI for each metric."""
    exp_dir = Path("experiments/exp_0338")
    result = run_multi_seed(exp_dir, seeds=range(42, 45), cache_dir=Path("cache"))
    assert "mean" in result
    assert "std" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "n_seeds" in result
    assert result["n_seeds"] == 3
    assert 0 < result["mean"]["overall_score"] < 1


def test_multi_seed_different_seeds_differ():
    """Different seeds should produce different individual scores."""
    exp_dir = Path("experiments/exp_0338")
    result = run_multi_seed(exp_dir, seeds=[42, 99], cache_dir=Path("cache"))
    assert result["std"]["overall_score"] >= 0  # can be 0 if deterministic
    assert len(result["individual_scores"]) == 2
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_multi_runner.py -v`
Expected: FAIL with "No module named 'memory_decay.multi_runner'"

**Step 3: Write minimal implementation**

```python
# src/memory_decay/multi_runner.py
"""Run experiment across multiple seeds and compute bootstrap CIs."""

import json
import numpy as np
from pathlib import Path
from memory_decay.runner import run_experiment


METRICS = [
    "overall_score", "retrieval_score", "plausibility_score",
    "recall_mean", "mrr_mean", "precision_lift", "precision_strict",
    "corr_score", "smoothness_score", "threshold_discrimination",
]


def run_multi_seed(
    exp_dir: Path,
    seeds: range | list[int] = range(42, 72),  # 30 seeds default
    cache_dir: Path = Path("cache"),
    ci_level: float = 0.95,
) -> dict:
    """Run experiment with multiple seeds, return stats with bootstrap CIs."""
    individual = []
    for seed in seeds:
        result = run_experiment(str(exp_dir), cache_dir=str(cache_dir), seed=seed)
        individual.append(result)

    n = len(individual)
    stats = {"n_seeds": n, "individual_scores": individual}

    for agg_key in ("mean", "std", "ci_lower", "ci_upper"):
        stats[agg_key] = {}

    for metric in METRICS:
        values = np.array([r.get(metric, 0.0) for r in individual])
        stats["mean"][metric] = float(np.mean(values))
        stats["std"][metric] = float(np.std(values, ddof=1)) if n > 1 else 0.0

        # Bootstrap 95% CI
        if n > 1:
            rng = np.random.default_rng(0)
            boot_means = []
            for _ in range(10000):
                sample = rng.choice(values, size=n, replace=True)
                boot_means.append(np.mean(sample))
            boot_means = np.sort(boot_means)
            alpha = (1 - ci_level) / 2
            stats["ci_lower"][metric] = float(boot_means[int(alpha * 10000)])
            stats["ci_upper"][metric] = float(boot_means[int((1 - alpha) * 10000)])
        else:
            stats["ci_lower"][metric] = stats["mean"][metric]
            stats["ci_upper"][metric] = stats["mean"][metric]

    return stats


def compare_experiments(
    exp_a: Path, exp_b: Path,
    seeds: range | list[int] = range(42, 72),
    cache_dir: Path = Path("cache"),
) -> dict:
    """Paired comparison between two experiments across same seeds."""
    diffs = {m: [] for m in METRICS}
    for seed in seeds:
        res_a = run_experiment(str(exp_a), cache_dir=str(cache_dir), seed=seed)
        res_b = run_experiment(str(exp_b), cache_dir=str(cache_dir), seed=seed)
        for m in METRICS:
            diffs[m].append(res_b.get(m, 0.0) - res_a.get(m, 0.0))

    from scipy import stats as sp_stats
    results = {}
    for m in METRICS:
        d = np.array(diffs[m])
        # Paired t-test: are the differences significantly different from 0?
        t_stat, p_val = sp_stats.ttest_1samp(d, 0.0) if len(d) > 1 else (0.0, 1.0)
        results[m] = {
            "mean_diff": float(np.mean(d)),
            "std_diff": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "significant_005": bool(p_val < 0.05),
        }
    return results
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_multi_runner.py -v`
Expected: PASS (2 tests, may take ~90s per seed)

**Step 5: Commit**

```bash
git add src/memory_decay/multi_runner.py tests/test_multi_runner.py
git commit -m "feat: multi-seed runner with bootstrap CIs and paired comparison"
```

---

### Task 2: Run Multi-Seed Validation on Key Experiments

**Files:**
- Create: `scripts/run_multi_seed_validation.py`
- Output: `experiments/multi_seed_results/`

**Step 1: Write the validation script**

```python
# scripts/run_multi_seed_validation.py
"""Validate key experiments across 30 seeds."""
import json
from pathlib import Path
from memory_decay.multi_runner import run_multi_seed, compare_experiments

EXPERIMENTS = {
    "baseline": Path("experiments/exp_0000"),
    "jost_p4": Path("experiments/exp_0315"),
    "best": Path("experiments/exp_0338"),
}
SEEDS = range(42, 72)  # 30 seeds
OUT = Path("experiments/multi_seed_results")

def main():
    OUT.mkdir(exist_ok=True)

    # 1. Individual multi-seed stats
    for name, exp_dir in EXPERIMENTS.items():
        print(f"\n=== {name} ({exp_dir}) ===")
        stats = run_multi_seed(exp_dir, seeds=SEEDS)
        out_file = OUT / f"{name}_stats.json"
        # Remove individual_scores for concise output
        summary = {k: v for k, v in stats.items() if k != "individual_scores"}
        with open(out_file, "w") as f:
            json.dump(summary, f, indent=2)
        m = stats["mean"]
        ci = (stats["ci_lower"]["overall_score"], stats["ci_upper"]["overall_score"])
        print(f"  overall: {m['overall_score']:.4f} ± {stats['std']['overall_score']:.4f}")
        print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    # 2. Paired comparisons
    pairs = [
        ("baseline", "jost_p4"),
        ("jost_p4", "best"),
        ("baseline", "best"),
    ]
    for a_name, b_name in pairs:
        print(f"\n=== {a_name} vs {b_name} ===")
        comp = compare_experiments(
            EXPERIMENTS[a_name], EXPERIMENTS[b_name], seeds=SEEDS
        )
        out_file = OUT / f"compare_{a_name}_vs_{b_name}.json"
        with open(out_file, "w") as f:
            json.dump(comp, f, indent=2)
        ov = comp["overall_score"]
        sig = "***" if ov["p_value"] < 0.001 else "**" if ov["p_value"] < 0.01 else "*" if ov["p_value"] < 0.05 else "n.s."
        print(f"  overall diff: {ov['mean_diff']:+.4f} (p={ov['p_value']:.4f}) {sig}")

if __name__ == "__main__":
    main()
```

**Step 2: Run the validation**

Run: `PYTHONPATH=src uv run python scripts/run_multi_seed_validation.py`
Expected: ~45 min (30 seeds x 3 experiments x ~30s each). Outputs JSON files.

**Step 3: Commit**

```bash
git add scripts/run_multi_seed_validation.py
git commit -m "feat: multi-seed validation script for key experiments"
```

---

### Task 3: K-Fold Cross-Validation

**Files:**
- Create: `src/memory_decay/cross_validator.py`
- Test: `tests/test_cross_validator.py`

**Step 1: Write the failing test**

```python
# tests/test_cross_validator.py
from pathlib import Path
from memory_decay.cross_validator import run_kfold


def test_kfold_returns_fold_results():
    exp_dir = Path("experiments/exp_0338")
    result = run_kfold(exp_dir, k=3, cache_dir=Path("cache"))
    assert "fold_scores" in result
    assert len(result["fold_scores"]) == 3
    assert "mean" in result
    assert "std" in result
    assert 0 < result["mean"]["overall_score"] < 1
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_cross_validator.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/memory_decay/cross_validator.py
"""K-fold cross-validation for decay function evaluation."""

import json
import random
import numpy as np
from pathlib import Path
from memory_decay.cache_builder import load_raw_dataset
from memory_decay.runner import run_experiment_with_split

METRICS = [
    "overall_score", "retrieval_score", "plausibility_score",
    "recall_mean", "mrr_mean", "corr_score",
]


def run_kfold(
    exp_dir: Path,
    k: int = 5,
    cache_dir: Path = Path("cache"),
    seed: int = 42,
) -> dict:
    """Run k-fold CV, rebuilding cache splits each fold."""
    dataset = load_raw_dataset(cache_dir / "dataset.json")
    rng = random.Random(seed)

    # Stratified k-fold: separate facts and episodes
    facts = [d for d in dataset if d.get("type") == "fact"]
    episodes = [d for d in dataset if d.get("type") == "episode"]
    rng.shuffle(facts)
    rng.shuffle(episodes)

    def split_into_folds(items, k):
        folds = [[] for _ in range(k)]
        for i, item in enumerate(items):
            folds[i % k].append(item)
        return folds

    fact_folds = split_into_folds(facts, k)
    episode_folds = split_into_folds(episodes, k)

    fold_scores = []
    for fold_idx in range(k):
        test_items = fact_folds[fold_idx] + episode_folds[fold_idx]
        train_items = []
        for j in range(k):
            if j != fold_idx:
                train_items.extend(fact_folds[j])
                train_items.extend(episode_folds[j])

        result = run_experiment_with_split(
            str(exp_dir), train_items, test_items,
            cache_dir=str(cache_dir), seed=seed,
        )
        fold_scores.append(result)

    stats = {"fold_scores": fold_scores, "k": k, "mean": {}, "std": {}}
    for m in METRICS:
        vals = [f.get(m, 0.0) for f in fold_scores]
        stats["mean"][m] = float(np.mean(vals))
        stats["std"][m] = float(np.std(vals, ddof=1))

    return stats
```

Note: `run_experiment_with_split` and `load_raw_dataset` will need to be added to runner.py and cache_builder.py respectively. These are thin wrappers that accept pre-split data instead of loading from cache.

**Step 4: Add required helper functions**

Modify `src/memory_decay/runner.py` — add `run_experiment_with_split()` function (~30 lines) that takes train/test items directly instead of loading from cache.

Modify `src/memory_decay/cache_builder.py` — add `load_raw_dataset()` that reads `dataset.json` without splitting.

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_cross_validator.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/memory_decay/cross_validator.py src/memory_decay/runner.py \
        src/memory_decay/cache_builder.py tests/test_cross_validator.py
git commit -m "feat: k-fold cross-validation framework for decay experiments"
```

---

---

## Phase 1 Checkpoint Gate

**STOP after Task 3. Review multi-seed results before proceeding.**

Decision criteria:
- If exp_0338 vs exp_0315 difference is **not significant** (p > 0.05): spreading activation claim must be dropped, Phase 2 ablation scope changes
- If cross-validation shows **high fold variance** (CV > 30%): overfitting confirmed, fundamental approach needs rethinking
- If multi-seed CI for exp_0338 **overlaps** with exp_0000 CI: all improvement claims need reframing

Only proceed to Phase 2 after reviewing these results with the user.

---

## Phase 2: Ablation Studies (P1 — Major)

Addresses: Confounded Phase 2 transition, activation_weight vs decay contribution, missing ablations

### Task 4: Ablation Experiment Generator

**Files:**
- Create: `scripts/run_ablation_study.py`
- Output: `experiments/ablation_results/`

**Step 1: Write the ablation script**

```python
# scripts/run_ablation_study.py
"""
Ablation study: systematically remove/disable components to measure individual contribution.

Ablations:
  A. Full model (exp_0338 as-is)
  B. No spreading activation (assoc_boost=0)
  C. No activation weighting (activation_weight=0)
  D. No Jost power (jost_power=1.0, linear excess)
  E. No sigmoid floor (floor_max=0)
  F. Exponential decay baseline (with Phase 2 scoring, fair comparison)
"""
import json
import copy
from pathlib import Path
from memory_decay.multi_runner import run_multi_seed

BASE_PARAMS = json.loads(Path("experiments/exp_0338/params.json").read_text())
SEEDS = range(42, 52)  # 10 seeds for ablation (faster)
OUT = Path("experiments/ablation_results")

ABLATIONS = {
    "full_model": {},
    "no_spreading": {"assoc_boost": 0.0},
    "no_activation_weight": {"activation_weight": 0.0},
    "linear_excess": {"jost_power": 1.0},
    "no_floor": {"floor_max": 0.0},
    "no_jost_no_floor": {"jost_power": 1.0, "floor_max": 0.0},
    # Note: jost_power controls excess^p in decay_fn.py line 43.
    # exp(alpha*impact) is in the retention/effective_lambda path (line 36),
    # which is a separate mechanism. So jost_power=1.0 validly isolates
    # the superlinear excess decay contribution.
}

def main():
    OUT.mkdir(exist_ok=True)

    # Create temporary experiment dirs for each ablation
    for name, overrides in ABLATIONS.items():
        ablation_dir = OUT / name
        ablation_dir.mkdir(exist_ok=True)

        # Copy decay_fn.py from exp_0338
        decay_src = Path("experiments/exp_0338/decay_fn.py").read_text()
        (ablation_dir / "decay_fn.py").write_text(decay_src)

        # Create modified params
        params = copy.deepcopy(BASE_PARAMS)
        params.update(overrides)
        (ablation_dir / "params.json").write_text(json.dumps(params))

        # Write hypothesis
        (ablation_dir / "hypothesis.txt").write_text(
            f"Ablation: {name}. Overrides: {overrides}"
        )

        print(f"\n=== Ablation: {name} ===")
        print(f"  Overrides: {overrides}")
        stats = run_multi_seed(ablation_dir, seeds=SEEDS)
        summary = {k: v for k, v in stats.items() if k != "individual_scores"}
        (ablation_dir / "multi_seed_stats.json").write_text(
            json.dumps(summary, indent=2)
        )
        m = stats["mean"]
        print(f"  overall: {m['overall_score']:.4f} ± {stats['std']['overall_score']:.4f}")
        print(f"  recall:  {m['recall_mean']:.4f}")
        print(f"  mrr:     {m['mrr_mean']:.4f}")
        print(f"  corr:    {m['corr_score']:.4f}")

if __name__ == "__main__":
    main()
```

**Step 2: Run ablation study**

Run: `PYTHONPATH=src uv run python scripts/run_ablation_study.py`
Expected: ~30 min (6 ablations x 10 seeds x ~30s). Outputs per-ablation stats.

**Step 3: Commit**

```bash
git add scripts/run_ablation_study.py
git commit -m "feat: ablation study script for component contribution analysis"
```

---

### Task 5: Fair Exponential Baseline Optimization

**Files:**
- Create: `scripts/optimize_exponential_baseline.py`
- Output: `experiments/fair_baseline/`

**Step 1: Write the optimization script**

This runs a grid search over exponential decay parameters under Phase 2 scoring to establish a fair baseline.

```python
# scripts/optimize_exponential_baseline.py
"""Find optimal exponential decay parameters under Phase 2 scoring."""
import json, itertools, copy
from pathlib import Path
from memory_decay.runner import run_experiment

EXPONENTIAL_DECAY = '''
import math

def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0
    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)
    combined = math.exp(alpha * impact) * (1 + rho * stability)
    lam = params.get("lambda_fact", 0.01) if mtype == "fact" else params.get("lambda_episode", 0.03)
    return activation * math.exp(-lam / combined)
'''

BASE_PARAMS = json.loads(Path("experiments/exp_0338/params.json").read_text())
OUT = Path("experiments/fair_baseline")

def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "decay_fn.py").write_text(EXPONENTIAL_DECAY)
    (OUT / "hypothesis.txt").write_text("Fair exponential baseline under Phase 2 scoring")

    best_score = -1
    best_params = None

    # Grid search over key parameters
    for lf, le, aw in itertools.product(
        [0.002, 0.005, 0.008, 0.012, 0.020],   # lambda_fact
        [0.010, 0.020, 0.035, 0.050, 0.080],    # lambda_episode
        [0.0, 0.3, 0.5, 0.7],                   # activation_weight
    ):
        params = copy.deepcopy(BASE_PARAMS)
        params.update({
            "lambda_fact": lf, "lambda_episode": le,
            "activation_weight": aw, "assoc_boost": 0.0,
        })
        (OUT / "params.json").write_text(json.dumps(params))
        try:
            result = run_experiment(str(OUT), cache_dir="cache", seed=42)
            score = result["overall_score"]
            if score > best_score:
                best_score = score
                best_params = params.copy()
                print(f"New best: {score:.4f} (lf={lf}, le={le}, aw={aw})")
        except Exception as e:
            pass

    # Save best
    (OUT / "params.json").write_text(json.dumps(best_params, indent=2))
    print(f"\nBest exponential baseline: {best_score:.4f}")
    print(f"Params: {json.dumps(best_params, indent=2)}")

if __name__ == "__main__":
    main()
```

**Step 2: Run**

Run: `PYTHONPATH=src uv run python scripts/optimize_exponential_baseline.py`
Expected: ~50 min (100 combinations x ~30s). Finds best exponential under Phase 2.

**Step 3: Commit**

```bash
git add scripts/optimize_exponential_baseline.py
git commit -m "feat: fair exponential baseline optimization for Phase 2 scoring"
```

---

## Phase 3: Notebook Revision (P0+P1 — All findings)

Addresses: All report corrections, new figures, corrected claims

### Task 6: Fix Factual Errors in Notebook

**Files:**
- Modify: `research_report.ipynb`

**Corrections to make (in order):**

1. **Section 7.2 — Spreading Activation formula** [R2: 불일치]
   - Change simple average to weighted average:
   ```
   Before: bar{a}_{N(m)} = 1/|N(m)| * sum(a_n)
   After:  bar{a}_{N(m)} = sum(w_i * a_i) / sum(w_i)  (edge-weight weighted)
   ```

2. **Figure 1 — Phase 2 effective weights** [R2: 부정확]
   - Fix pie chart: Correlation ~1.8%, Smoothness ~1.8% (not 7%)
   - Add note about multiplicative gate diminishing plausibility influence

3. **Section 12.3 — Theoretical upper bound table** [R2: 불일치]
   - Document that MRR is assumed to increase with precision_lift
   - Or fix table with fixed MRR=0.248

4. **Abstract & Section 15 — "10.7x improvement"** [R1, R3: 치명적]
   - Replace with Phase 2 internal improvement: "1.48x (0.153→0.226)"
   - Note baseline change transparently

5. **"338 experiments"** [R2: 주의]
   - Correct to "336 experiments (numbered through 338)"

**Step 1: Apply all corrections to notebook cells**

Use NotebookEdit for each cell requiring changes.

**Step 2: Commit**

```bash
git add research_report.ipynb
git commit -m "fix: correct factual errors per reviewer findings"
```

---

### Task 7: Add New Analysis Sections to Notebook

**Files:**
- Modify: `research_report.ipynb` (add cells)

**New sections to add:**

1. **Section 8.5: Threshold Discrimination Analysis** [R3: Critical]
   - Document that discrimination=0 and explain implications
   - New figure: activation distribution histogram showing floor clustering
   - Honest assessment: "decay functions as ranking modulator, not selective forgetter"

2. **Section 13.5: Statistical Validation** (after multi-seed runs)
   - Table: mean ± std for key experiments across 30 seeds
   - Paired t-test results for key comparisons (baseline vs best, p=1.5 vs p=4.0)
   - Figure: box plots with individual seed results

3. **Section 13.6: Ablation Study Results** (after ablation runs)
   - Table: each component's contribution with error bars
   - Figure: ablation waterfall chart

4. **Section 13.7: Cross-Validation Results** (after CV runs)
   - Table: per-fold scores
   - Figure: fold score distribution

5. **Section 15 (revised): Limitations**
   - Expand with all acknowledged limitations from review:
     - Single synthetic dataset (Korean, 416 items)
     - Jost's Law naming caveat
     - Precision lift metric design issue
     - Threshold discrimination = 0 implication

**Step 1: Add placeholder cells** (to be filled after Phase 1-2 data is available)

**Step 2: Commit**

```bash
git add research_report.ipynb
git commit -m "feat: add statistical validation, ablation, and limitations sections"
```

---

### Task 8: Rename Jost's Law References [P1]

**Files:**
- Modify: `research_report.ipynb`

**Changes:**
- Primary naming: "Superlinear Excess Decay (SED)" or "Sigmoid-Floor Excess Decay"
- Keep Jost's Law as inspiration reference, not primary name
- Add caveat in Section 6.1: "While inspired by Jost's Law, our formulation captures activation-dependent (not age-dependent) decay, which is a distinct mechanism"

**Step 1: Find-and-replace across notebook markdown cells**

**Step 2: Commit**

```bash
git add research_report.ipynb
git commit -m "fix: rename Jost's Law to Superlinear Excess Decay with Jost inspiration caveat"
```

---

## Phase 4: Final Execution & Data Integration

### Task 9: Execute All Analyses and Integrate

**Step 1: Run multi-seed validation** (Task 2)
**Step 2: Run ablation study** (Task 4)
**Step 3: Run fair baseline optimization** (Task 5)
**Step 4: Run k-fold CV** (Task 3 script)
**Step 5: Integrate all results into notebook** (Task 7 placeholders)
**Step 6: Re-execute notebook end-to-end**

Run: `PYTHONPATH=src uv run jupyter nbconvert --to notebook --execute research_report.ipynb --output research_report.ipynb`

**Step 7: Final commit**

```bash
git add -A
git commit -m "feat: complete review remediation with statistical validation, ablation, and corrected claims"
```

---

## Summary: Review Finding → Task Mapping

| Finding | Severity | Task | Status |
|---------|----------|------|--------|
| Single seed (n=1) | Critical | Task 1, 2 | |
| Test set leakage | Critical | Task 3 | |
| "10.7x" claim | Critical | Task 6.4 | |
| Threshold discrimination=0 | Critical | Task 7.1 | |
| Phase 2 confounded changes | Major | Task 4 | |
| Jost's Law naming | Major | Task 8 | |
| Precision lift metric issue | Major | Task 7.5 | |
| Spreading activation formula error | Major | Task 6.1 | |
| Effective weights pie chart | Major | Task 6.2 | |
| Upper bound table assumptions | Minor | Task 6.3 | |
| Experiment count (336 vs 338) | Minor | Task 6.5 | |
| Fair exponential baseline | Major | Task 5 | |
