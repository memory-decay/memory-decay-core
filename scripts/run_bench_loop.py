"""MemoryBench-driven auto-improvement loop.

Uses MemoryBench (LongMemEval, LoCoMo, ConvoMem) as the optimization target.
Pre-built embedding cache (cache/openai/) eliminates embedding API costs.
Only answer/judge LLM calls cost money (~$0.60/iteration for Stage A).

Usage:
    export OPENAI_API_KEY=sk-...
    PYTHONPATH=src .venv/bin/python scripts/run_bench_loop.py [--budget 20]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memory_decay.auto_improver import AutoImprover
from memory_decay.bench_evaluator import evaluate, CompositeResult

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
HISTORY_PATH = EXPERIMENTS_DIR / "history.jsonl"


def _next_exp_num() -> int:
    """Find next experiment number."""
    existing = sorted(EXPERIMENTS_DIR.glob("exp_bench_*"))
    if not existing:
        return 1
    last = existing[-1].name.split("_")[-1]
    return int(last) + 1


def _load_best() -> tuple[Path, dict, float]:
    """Load current best experiment's params and score."""
    best_dir = EXPERIMENTS_DIR / "best"
    if not best_dir.exists():
        best_dir = EXPERIMENTS_DIR / "exp_lme_0255"  # fallback to previous best

    params_path = best_dir / "params.json"
    params = json.loads(params_path.read_text()) if params_path.exists() else {}

    # Try to load bench_results.json for MemoryBench score
    bench_results_path = best_dir / "bench_results.json"
    if bench_results_path.exists():
        bench_data = json.loads(bench_results_path.read_text())
        best_score = bench_data.get("bench_score", 0.0)
    else:
        best_score = 0.0  # No MemoryBench baseline yet

    return best_dir, params, best_score


def _save_experiment(
    exp_dir: Path,
    params: dict,
    result: CompositeResult,
    hypothesis: str,
    decay_fn_source: Path | None = None,
) -> None:
    """Save experiment artifacts."""
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save params
    (exp_dir / "params.json").write_text(json.dumps(params, indent=2))

    # Save bench results
    (exp_dir / "bench_results.json").write_text(
        json.dumps(result.to_dict(), indent=2)
    )

    # Save hypothesis
    (exp_dir / "hypothesis.txt").write_text(hypothesis)

    # Copy decay_fn.py from best if not provided
    if decay_fn_source and decay_fn_source.exists():
        shutil.copy2(decay_fn_source, exp_dir / "decay_fn.py")
    elif (EXPERIMENTS_DIR / "best" / "decay_fn.py").exists():
        shutil.copy2(EXPERIMENTS_DIR / "best" / "decay_fn.py", exp_dir / "decay_fn.py")


def _record_history(
    exp_name: str,
    bench_score: float,
    result: CompositeResult,
    status: str,
    hypothesis: str,
) -> None:
    """Append to history.jsonl."""
    entry = {
        "experiment": exp_name,
        "bench_score": round(bench_score, 4),
        "status": status,
        "hypothesis": hypothesis[:200],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for name, r in result.results.items():
        entry[f"{name}_acc"] = round(r.accuracy, 4)

    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_loop(budget: int = 20, stage_a_limit: int = 20, stage_b_limit: int = 50):
    """Run the auto-improvement loop."""
    print("=" * 60)
    print("MemoryBench Auto-Improvement Loop")
    print("=" * 60)

    # Load current best
    best_dir, best_params, best_score = _load_best()
    print(f"Current best: {best_dir.name}, bench_score={best_score:.4f}")
    print(f"Params: {json.dumps({k: best_params[k] for k in list(best_params)[:5]}, indent=2)}...")

    # If no baseline MemoryBench score, run baseline first
    if best_score == 0.0:
        print("\nNo MemoryBench baseline — running baseline evaluation...")
        baseline_result = evaluate(
            best_dir,
            run_prefix="baseline",
            limit=stage_a_limit,
        )
        best_score = baseline_result.bench_score
        print(f"Baseline bench_score: {best_score:.4f}")

        # Save baseline results
        (best_dir / "bench_results.json").write_text(
            json.dumps(baseline_result.to_dict(), indent=2)
        )
        _record_history(
            best_dir.name, best_score, baseline_result, "baseline", "Initial MemoryBench baseline"
        )

    # Initialize improver
    improver = AutoImprover(guidance_level="expert")
    eval_history: list[dict] = [{"bench_score": best_score, "overall_score": best_score}]
    no_improvement_count = 0
    current_params = dict(best_params)

    for i in range(budget):
        print(f"\n{'='*60}")
        print(f"Iteration {i+1}/{budget} | best_score={best_score:.4f} | no_improve={no_improvement_count}")
        print(f"{'='*60}")

        if no_improvement_count >= 8:
            print("Stopping: 8 consecutive iterations without improvement")
            break

        # 1. Propose new parameters
        print("Proposing parameters...")
        new_params = improver.propose_parameters(
            current_params, eval_history, i, budget,
        )

        # Check if params actually changed
        if new_params == current_params:
            print("  No parameter changes proposed, skipping")
            no_improvement_count += 1
            continue

        # Show changes
        changes = {k: new_params[k] for k in new_params if new_params.get(k) != current_params.get(k)}
        print(f"  Changes: {json.dumps(changes, indent=2)}")

        # 2. Create experiment
        exp_num = _next_exp_num()
        exp_name = f"exp_bench_{exp_num:04d}"
        exp_dir = EXPERIMENTS_DIR / exp_name
        print(f"  Experiment: {exp_name}")

        # Save experiment files (with decay_fn from best)
        reasoning = improver._history[-1].get("reasoning", "") if improver._history else ""
        _save_experiment(exp_dir, new_params, CompositeResult(bench_score=0.0), reasoning)

        # 3. Stage A: Quick screen
        print(f"\n  Stage A: {stage_a_limit} questions/benchmark...")
        try:
            stage_a_result = evaluate(
                exp_dir,
                run_prefix=exp_name,
                limit=stage_a_limit,
            )
        except Exception as e:
            print(f"  Stage A failed: {e}")
            _record_history(exp_name, 0.0, CompositeResult(bench_score=0.0), "error", str(e))
            no_improvement_count += 1
            continue

        print(f"  Stage A score: {stage_a_result.bench_score:.4f} (best: {best_score:.4f})")

        # 4. Judge
        if stage_a_result.bench_score <= best_score:
            print(f"  No improvement ({stage_a_result.bench_score:.4f} <= {best_score:.4f})")
            _record_history(exp_name, stage_a_result.bench_score, stage_a_result, "no_improvement", reasoning)
            _save_experiment(exp_dir, new_params, stage_a_result, reasoning)
            eval_history.append({"bench_score": stage_a_result.bench_score, "overall_score": stage_a_result.bench_score})
            no_improvement_count += 1
            continue

        # 5. Stage B: Confirm with more questions
        print(f"\n  Stage B: {stage_b_limit} questions/benchmark (confirming)...")
        try:
            stage_b_result = evaluate(
                exp_dir,
                run_prefix=f"{exp_name}-stageB",
                limit=stage_b_limit,
            )
        except Exception as e:
            print(f"  Stage B failed: {e}")
            _record_history(exp_name, stage_a_result.bench_score, stage_a_result, "stage_b_error", str(e))
            no_improvement_count += 1
            continue

        print(f"  Stage B score: {stage_b_result.bench_score:.4f}")

        if stage_b_result.bench_score > best_score:
            # Accept!
            print(f"  ✓ IMPROVEMENT CONFIRMED: {best_score:.4f} → {stage_b_result.bench_score:.4f}")
            best_score = stage_b_result.bench_score
            current_params = dict(new_params)
            no_improvement_count = 0

            # Update best symlink
            best_link = EXPERIMENTS_DIR / "best"
            if best_link.is_symlink():
                best_link.unlink()
            elif best_link.exists():
                shutil.rmtree(best_link)
            best_link.symlink_to(exp_dir.name)

            _save_experiment(exp_dir, new_params, stage_b_result, reasoning)
            _record_history(exp_name, stage_b_result.bench_score, stage_b_result, "improved", reasoning)
            eval_history.append({"bench_score": stage_b_result.bench_score, "overall_score": stage_b_result.bench_score})
        else:
            print(f"  Stage B did not confirm ({stage_b_result.bench_score:.4f} <= {best_score:.4f})")
            _save_experiment(exp_dir, new_params, stage_b_result, reasoning)
            _record_history(exp_name, stage_b_result.bench_score, stage_b_result, "stage_b_rejected", reasoning)
            eval_history.append({"bench_score": stage_b_result.bench_score, "overall_score": stage_b_result.bench_score})
            no_improvement_count += 1

    print(f"\n{'='*60}")
    print(f"Loop complete. Final best_score: {best_score:.4f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="MemoryBench auto-improvement loop")
    parser.add_argument("--budget", type=int, default=20, help="Max iterations")
    parser.add_argument("--stage-a-limit", type=int, default=20, help="Questions per benchmark in Stage A")
    parser.add_argument("--stage-b-limit", type=int, default=50, help="Questions per benchmark in Stage B")
    args = parser.parse_args()

    run_loop(
        budget=args.budget,
        stage_a_limit=args.stage_a_limit,
        stage_b_limit=args.stage_b_limit,
    )


if __name__ == "__main__":
    main()
