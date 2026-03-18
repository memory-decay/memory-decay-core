"""Main simulation runner: end-to-end memory decay experiment.

Orchestrates the full pipeline:
1. Generate synthetic dataset (or load from JSONL)
2. Build MemoryGraph
3. Run simulation with DecayEngine
4. Evaluate at intervals
5. (Optional) Run auto-improvement loop
6. Output results
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

from .data_gen import SyntheticDataGenerator
from .graph import MemoryGraph
from .decay import DecayEngine
from .evaluator import Evaluator
from .auto_improver import AutoImprover


def merge_human_calibrated_params(base_params: dict, best_params_path: str) -> dict:
    """Apply only fact-side calibrated params from a human calibration artifact."""
    with open(best_params_path, "r", encoding="utf-8") as f:
        fitted = json.load(f)

    merged = dict(base_params)
    for key in (
        "lambda_fact",
        "stability_weight",
        "stability_decay",
        "reinforcement_gain_direct",
    ):
        if key in fitted:
            merged[key] = fitted[key]
    return merged


def build_graph_from_dataset(
    dataset: list[dict], embedder=None, embedding_backend: str = "auto"
) -> MemoryGraph:
    """Load a dataset into a MemoryGraph."""
    graph = MemoryGraph(embedder=embedder, embedding_backend=embedding_backend)

    for mem in dataset:
        # Resolve associations to (id, weight) tuples
        assocs = []
        for assoc in mem.get("associations", []):
            if isinstance(assoc, dict):
                assocs.append((assoc["id"], assoc.get("weight", 0.5)))
            elif isinstance(assoc, str):
                assocs.append((assoc, 0.5))

        graph.add_memory(
            memory_id=mem["id"],
            mtype=mem["type"],
            content=mem["content"],
            impact=mem.get("impact", 0.5),
            created_tick=mem.get("tick", 0),
            associations=assocs,
        )

    return graph


def run_simulation(
    graph: MemoryGraph,
    engine: DecayEngine,
    evaluator: Evaluator,
    test_queries: list[tuple[str, str]],
    total_ticks: int = 100,
    eval_interval: int = 5,
    reactivation_policy: str = "none",
    reactivation_interval: int = 10,
    reactivation_boost: float = 0.3,
    rehearsal_targets: list[str] | None = None,
    seed: Optional[int] = None,
) -> list[dict]:
    """Run a simulation and return evaluation snapshots.

    Args:
        graph: MemoryGraph with loaded memories.
        engine: DecayEngine configured with decay parameters.
        evaluator: Evaluator instance.
        test_queries: List of (query, expected_id) pairs.
        total_ticks: Total simulation ticks.
        eval_interval: Evaluate every N ticks.
        reactivation_policy: Re-activation policy: none, random, scheduled_query.
        reactivation_interval: Apply the selected policy every N ticks.
        reactivation_boost: Activation boost for direct re-activation.
        rehearsal_targets: Memory IDs eligible for scheduled_query reactivation.
            Required when reactivation_policy is "scheduled_query".
        seed: Random seed used by the random policy.

    Returns:
        List of evaluation summaries.
    """
    if reactivation_policy not in {"none", "random", "scheduled_query"}:
        raise ValueError(f"Unsupported reactivation_policy: {reactivation_policy}")

    if reactivation_policy == "scheduled_query" and not rehearsal_targets:
        raise ValueError(
            "rehearsal_targets must be provided when reactivation_policy is 'scheduled_query'"
        )

    rng = random.Random(seed)
    summaries = []
    params = engine.get_params()

    def collect_summary() -> dict:
        snap = evaluator.snapshot(test_queries)
        summary = evaluator.score_summary(test_queries)
        summary["tick"] = snap["tick"]
        return summary

    def apply_reactivation(tick: int) -> None:
        if reactivation_policy == "none" or tick % reactivation_interval != 0:
            return

        if reactivation_policy == "random":
            candidates = [
                nid
                for nid, attrs in graph._graph.nodes(data=True)
                if attrs.get("type") not in ("unknown", None)
                and attrs.get("created_tick", 0) <= engine.current_tick
            ]
            if not candidates:
                return
            target_id = rng.choice(candidates)
        else:  # scheduled_query — use rehearsal_targets, NOT test_queries
            assert rehearsal_targets is not None
            # Filter to only memories that exist at current tick
            eligible = [
                mid for mid in rehearsal_targets
                if graph._graph.nodes[mid].get("created_tick", 0) <= engine.current_tick
            ]
            if not eligible:
                return
            idx = ((tick // reactivation_interval) - 1) % len(eligible)
            target_id = eligible[idx]

        graph.re_activate(
            target_id,
            reactivation_boost,
            source="direct",
            reinforce=True,
            current_tick=engine.current_tick,
            reinforcement_gain_direct=params["reinforcement_gain_direct"],
            reinforcement_gain_assoc=params["reinforcement_gain_assoc"],
            stability_cap=params["stability_cap"],
        )

    # Initial evaluation (tick 0)
    summary = collect_summary()
    summaries.append(summary)
    print(
        f"  Tick {summary['tick']:>4d} | recall={summary['recall_rate']:.3f} | "
        f"precision={summary['precision_rate']:.3f} | retrieval={summary['retrieval_score']:.3f} | "
        f"overall={summary['overall_score']:.3f}"
    )

    for t in range(1, total_ticks + 1):
        apply_reactivation(t)

        engine.tick()

        if t % eval_interval == 0:
            summary = collect_summary()
            summaries.append(summary)
            print(
                f"  Tick {summary['tick']:>4d} | recall={summary['recall_rate']:.3f} | "
                f"precision={summary['precision_rate']:.3f} | retrieval={summary['retrieval_score']:.3f} | "
                f"overall={summary['overall_score']:.3f}"
            )

    return summaries


def run_experiment(
    decay_type: str = "exponential",
    num_memories: int = 50,
    total_ticks: int = 100,
    eval_interval: int = 5,
    reactivation_policy: str = "none",
    embedding_backend: str = "auto",
    guidance_level: str = "default",
    improvement_budget: int = 12,
    api_key: Optional[str] = None,
    dataset_path: Optional[str] = None,
    calibrated_params_path: Optional[str] = None,
    output_path: Optional[str] = None,
    seed: int = 42,
) -> dict:
    """Run a complete experiment (generation → simulation → evaluation → improvement)."""
    random.seed(seed)

    # Step 1: Get dataset
    if dataset_path:
        print(f"Loading dataset from {dataset_path}...")
        dataset = SyntheticDataGenerator.load_jsonl(dataset_path)
    else:
        print(f"Generating {num_memories} memories with Anthropic API...")
        gen = SyntheticDataGenerator(api_key=api_key)
        dataset = gen.generate_dataset(
            num_memories=num_memories,
            ticks_range=(0, total_ticks),
            seed=seed,
        )

    print(f"Dataset: {len(dataset)} memories")

    # Step 2: Split train/test
    gen = SyntheticDataGenerator(api_key=api_key or "dummy")
    train, test = gen.split_test_train(dataset, test_ratio=0.2, seed=seed)
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Build test queries
    test_queries = [(m["recall_query"], m["id"]) for m in test if "recall_query" in m]
    rehearsal_targets = [m["id"] for m in train]
    print(f"Test queries: {len(test_queries)}")

    # Step 3: Build graph
    print("Building memory graph...")
    graph = build_graph_from_dataset(dataset, embedding_backend=embedding_backend)

    # Step 4: Initial parameters
    params = {
        "lambda_fact": 0.05,
        "lambda_episode": 0.08,
        "beta_fact": 0.3,
        "beta_episode": 0.5,
        "alpha": 0.5,
        "stability_weight": 0.8,
        "stability_decay": 0.01,
        "reinforcement_gain_direct": 0.2,
        "reinforcement_gain_assoc": 0.05,
        "stability_cap": 1.0,
    }
    if calibrated_params_path:
        params = merge_human_calibrated_params(params, calibrated_params_path)

    # Step 5: Initial run
    engine = DecayEngine(graph, decay_type=decay_type, params=params)
    evaluator = Evaluator(graph, engine)

    print(f"\n=== Initial Run (decay_type={decay_type}) ===")
    initial_summaries = run_simulation(
        graph, engine, evaluator, test_queries,
        total_ticks=total_ticks, eval_interval=eval_interval,
        reactivation_policy=reactivation_policy,
        rehearsal_targets=rehearsal_targets,
        seed=seed,
    )
    initial_summary = initial_summaries[-1]
    initial_score = initial_summary["overall_score"]
    print(f"Initial overall score: {initial_score:.4f}")

    # Step 6: Auto-improvement loop
    improvement_history = []
    if improvement_budget > 0 and api_key:
        print(f"\n=== Auto-Improvement ({guidance_level}, budget={improvement_budget}) ===")
        improver = AutoImprover(
            api_key=api_key,
            guidance_level=guidance_level,
        )

        all_eval_history = initial_summaries[:]

        for i in range(improvement_budget):
            if improver.should_stop(all_eval_history, i, improvement_budget):
                print(f"  Early stop at iteration {i}")
                break

            print(f"\n  Iteration {i + 1}/{improvement_budget}:")
            new_params = improver.propose_parameters(
                params, all_eval_history, i, improvement_budget
            )
            print(f"  Params: {json.dumps(new_params, indent=4)}")

            # Reset graph and run with new params
            graph2 = build_graph_from_dataset(
                dataset,
                embedder=graph._get_embedder(),
                embedding_backend=embedding_backend,
            )
            engine2 = DecayEngine(graph2, decay_type=decay_type, params=new_params)
            evaluator2 = Evaluator(graph2, engine2)

            summaries = run_simulation(
                graph2, engine2, evaluator2, test_queries,
                total_ticks=total_ticks, eval_interval=eval_interval,
                reactivation_policy=reactivation_policy,
                rehearsal_targets=rehearsal_targets,
                seed=seed,
            )
            score_summary = summaries[-1]
            score = score_summary["overall_score"]
            print(f"  Overall score: {score:.4f}")

            improvement_history.append({
                "iteration": i,
                "params": new_params,
                "score_summary": score_summary,
                "overall_score": score,
                "snapshots": summaries,
            })

            all_eval_history.extend(summaries)
            params = new_params

            # Clean up to free memory
            del graph2, engine2, evaluator2

    # Step 7: Output results
    results = {
        "decay_type": decay_type,
        "num_memories": len(dataset),
        "total_ticks": total_ticks,
        "initial_params": engine.get_params(),
        "initial_snapshots": initial_summaries,
        "initial_score_summary": initial_summary,
        "initial_overall_score": initial_score,
        "initial_composite_score": initial_score,
        "improvement_history": improvement_history,
        "guidance_level": guidance_level,
        "reactivation_policy": reactivation_policy,
        "embedding_backend": embedding_backend,
    }

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Human-like Memory Decay Simulation")
    parser.add_argument("--decay-type", choices=["exponential", "power_law"], default="exponential")
    parser.add_argument("--num-memories", type=int, default=50)
    parser.add_argument("--total-ticks", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "local", "gemini"],
        default="auto",
    )
    parser.add_argument(
        "--reactivation-policy",
        choices=["none", "random", "scheduled_query"],
        default="none",
    )
    parser.add_argument("--guidance", choices=["minimal", "default", "expert"], default="default")
    parser.add_argument("--budget", type=int, default=12, help="Auto-improvement iterations")
    parser.add_argument("--dataset", type=str, default=None, help="Path to JSONL dataset")
    parser.add_argument(
        "--calibrated-params",
        type=str,
        default=None,
        help="Path to best_params.json produced by human calibration",
    )
    parser.add_argument("--output", type=str, default="data/results.json", help="Output path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiment(
        decay_type=args.decay_type,
        num_memories=args.num_memories,
        total_ticks=args.total_ticks,
        eval_interval=args.eval_interval,
        reactivation_policy=args.reactivation_policy,
        embedding_backend=args.embedding_backend,
        guidance_level=args.guidance,
        improvement_budget=args.budget,
        dataset_path=args.dataset,
        calibrated_params_path=args.calibrated_params,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
