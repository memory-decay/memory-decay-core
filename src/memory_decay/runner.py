"""Single experiment runner for the auto-research loop.

Loads a custom decay function from an experiment directory,
runs a simulation using cached embeddings, and saves results.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Optional

from .cache_builder import load_cache, load_cached_embedder
from .main import build_graph_from_dataset, run_simulation
from .decay import DecayEngine
from .evaluator import Evaluator


def validate_decay_fn(fn_path: str, params: dict) -> tuple[bool, Optional[str]]:
    """Validate a decay function file before running a full experiment."""
    path = Path(fn_path)

    # 1. Syntax check
    try:
        with open(path, "r") as f:
            source = f.read()
        compile(source, str(path), "exec")
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    # 2. Load module
    try:
        spec = importlib.util.spec_from_file_location("decay_fn", str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        return False, f"Import error: {e}"

    if not hasattr(module, "compute_decay") or not callable(module.compute_decay):
        return False, "Module has no callable 'compute_decay'"

    fn = module.compute_decay

    # 3. Basic output range checks
    test_cases = [
        (1.0, 0.0, 0.0, "fact"),
        (1.0, 0.0, 0.0, "episode"),
        (0.5, 0.5, 0.5, "fact"),
        (0.8, 1.0, 1.0, "episode"),
        (0.0, 0.0, 0.0, "fact"),
    ]

    results = []
    for activation, impact, stability, mtype in test_cases:
        try:
            result = fn(activation, impact, stability, mtype, params)
        except Exception as e:
            return False, f"Runtime error with inputs ({activation}, {impact}, {stability}, {mtype}): {e}"

        if not isinstance(result, (int, float)):
            return False, f"Output is not numeric: {type(result)}"
        if result < -0.01 or result > 1.01:
            return False, f"Output {result} out of range [0, 1]"
        results.append(result)

    # 4. Must actually decay
    if results[0] >= 1.0 and results[1] >= 1.0:
        return False, "No decay detected: compute_decay(1.0, 0, 0, ...) returned 1.0 (constant, no decay)"

    # 5. Zero input should stay near zero
    if results[4] > 0.01:
        return False, f"Zero activation produced non-zero output: {results[4]}"

    # 6. Pure decay must not increase activation without reactivation.
    # Check a representative grid to reject self-exciting floor hacks.
    eps = 1e-6
    inflation_cases = [
        (activation, impact, stability, mtype)
        for activation in (0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9)
        for impact in (0.0, 0.5, 1.0)
        for stability in (0.0, 0.5, 1.0)
        for mtype in ("fact", "episode")
    ]

    for activation, impact, stability, mtype in inflation_cases:
        try:
            result = fn(activation, impact, stability, mtype, params)
        except Exception as e:
            return False, (
                f"Runtime error during monotonicity check "
                f"({activation}, {impact}, {stability}, {mtype}): {e}"
            )

        if result > activation + eps:
            return False, (
                "Activation increase detected without reactivation: "
                f"compute_decay({activation}, {impact}, {stability}, {mtype}) "
                f"returned {result:.6f} > {activation:.6f}. "
                "Decay functions must be monotone non-increasing."
            )

    return True, None


def _load_decay_fn(fn_path: str):
    """Dynamically import a decay function from a file."""
    spec = importlib.util.spec_from_file_location("decay_fn", fn_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_decay


def run_experiment(
    experiment_dir: str,
    cache_dir: str,
    total_ticks: int = 200,
    eval_interval: int = 20,
    reactivation_policy: str = "scheduled_query",
    seed: int = 42,
    force: bool = False,
) -> dict:
    """Run a single experiment from an experiment directory."""
    exp_path = Path(experiment_dir)
    start_time = time.time()
    results_path = exp_path / "results.json"

    if results_path.exists() and not force:
        raise FileExistsError(
            f"{results_path} already exists; refusing to overwrite prior experiment results. "
            "Use force=True or --force to rerun explicitly."
        )

    with open(exp_path / "params.json", "r") as f:
        params = json.load(f)

    fn_path = str(exp_path / "decay_fn.py")

    ok, error = validate_decay_fn(fn_path, params)
    if not ok:
        result = {
            "status": "validation_failed",
            "error": error,
            "duration_seconds": round(time.time() - start_time, 2),
        }
        with open(results_path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    cached_embedder, dataset, test_queries, rehearsal_targets = load_cache(cache_dir)
    graph = build_graph_from_dataset(dataset, embedder=cached_embedder)
    decay_fn = _load_decay_fn(fn_path)
    engine = DecayEngine(graph, custom_decay_fn=decay_fn, params=params)
    activation_weight = params.get("activation_weight", 0.5)
    assoc_boost = params.get("assoc_boost", 0.0)
    evaluator = Evaluator(graph, engine, activation_weight=activation_weight, assoc_boost=assoc_boost)

    snapshots = run_simulation(
        graph, engine, evaluator, test_queries,
        total_ticks=total_ticks,
        eval_interval=eval_interval,
        reactivation_policy=reactivation_policy,
        rehearsal_targets=rehearsal_targets,
        seed=seed,
    )

    final_summary = evaluator.score_summary(test_queries)
    duration = time.time() - start_time

    result = {
        "status": "completed",
        **final_summary,
        "snapshots": snapshots,
        "duration_seconds": round(duration, 2),
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def run_experiment_with_split(
    experiment_dir: str,
    train_items: list[dict],
    test_items: list[dict],
    cache_dir: str = "cache",
    total_ticks: int = 200,
    eval_interval: int = 20,
    reactivation_policy: str = "scheduled_query",
    seed: int = 42,
) -> dict:
    """Run an experiment with a pre-determined train/test split.

    Instead of loading the fixed split from cache, this accepts explicit
    train/test item lists.  Used by k-fold cross-validation.

    Returns the score_summary dict (same shape as the final summary from
    ``run_experiment``).
    """
    exp_path = Path(experiment_dir)

    with open(exp_path / "params.json", "r") as f:
        params = json.load(f)

    fn_path = str(exp_path / "decay_fn.py")

    ok, error = validate_decay_fn(fn_path, params)
    if not ok:
        return {"status": "validation_failed", "error": error, "overall_score": 0.0}

    # Load cached embedder (no dataset/split — we supply our own)
    cached_embedder = load_cached_embedder(cache_dir)

    # Build graph from ALL items (train + test) — the graph must contain
    # every memory so that associations and similarity search work correctly.
    all_items = train_items + test_items
    graph = build_graph_from_dataset(all_items, embedder=cached_embedder)

    # Derive test_queries and rehearsal_targets from the provided splits
    test_queries = [
        (item["recall_query"], item["id"])
        for item in test_items
        if "recall_query" in item
    ]
    rehearsal_targets = [item["id"] for item in train_items]

    decay_fn = _load_decay_fn(fn_path)
    engine = DecayEngine(graph, custom_decay_fn=decay_fn, params=params)
    activation_weight = params.get("activation_weight", 0.5)
    assoc_boost = params.get("assoc_boost", 0.0)
    evaluator = Evaluator(
        graph, engine,
        activation_weight=activation_weight,
        assoc_boost=assoc_boost,
    )

    run_simulation(
        graph, engine, evaluator, test_queries,
        total_ticks=total_ticks,
        eval_interval=eval_interval,
        reactivation_policy=reactivation_policy,
        rehearsal_targets=rehearsal_targets,
        seed=seed,
    )

    return evaluator.score_summary(test_queries)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a single decay experiment")
    parser.add_argument("experiment_dir", help="Path to experiment directory")
    parser.add_argument("--cache", default="cache", help="Cache directory")
    parser.add_argument("--ticks", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--policy", default="scheduled_query")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing results.json")
    args = parser.parse_args()

    result = run_experiment(
        args.experiment_dir, args.cache,
        total_ticks=args.ticks, eval_interval=args.eval_interval,
        reactivation_policy=args.policy, seed=args.seed, force=args.force,
    )

    status = result["status"]
    if status == "completed":
        print(f"Done: overall={result['overall_score']:.4f} "
              f"retrieval={result['retrieval_score']:.4f} "
              f"plausibility={result['plausibility_score']:.4f} "
              f"({result['duration_seconds']}s)")
    else:
        print(f"Failed: {result.get('error', 'unknown')}")
