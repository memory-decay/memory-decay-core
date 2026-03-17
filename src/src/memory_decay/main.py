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


def build_graph_from_dataset(
    dataset: list[dict], embedder=None
) -> MemoryGraph:
    """Load a dataset into a MemoryGraph."""
    graph = MemoryGraph(embedder=embedder)

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
    reactivation_interval: int = 10,
) -> list[dict]:
    """Run a simulation and return evaluation snapshots.

    Args:
        graph: MemoryGraph with loaded memories.
        engine: DecayEngine configured with decay parameters.
        evaluator: Evaluator instance.
        test_queries: List of (query, expected_id) pairs.
        total_ticks: Total simulation ticks.
        eval_interval: Evaluate every N ticks.
        reactivation_interval: Randomly re-activate a memory every N ticks.

    Returns:
        List of evaluation snapshots.
    """
    snapshots = []

    # Initial evaluation (tick 0)
    snap = evaluator.snapshot(test_queries)
    snapshots.append(snap)
    print(f"  Tick {snap['tick']:>4d} | recall={snap['recall_rate']:.3f} | "
          f"precision={snap['precision_rate']:.3f} | composite={snap.get('composite_score', 0):.3f}")

    for t in range(1, total_ticks + 1):
        # Random re-activation (simulates encountering related information)
        if t % reactivation_interval == 0:
            all_ids = [nid for nid, attrs in graph._graph.nodes(data=True)
                       if attrs.get("type") not in ("unknown", None)]
            if all_ids:
                target = random.choice(all_ids)
                graph.re_activate(target, 0.3)

        engine.tick()

        if t % eval_interval == 0:
            snap = evaluator.snapshot(test_queries)
            snapshots.append(snap)
            print(f"  Tick {snap['tick']:>4d} | recall={snap['recall_rate']:.3f} | "
                  f"precision={snap['precision_rate']:.3f} | composite={snap.get('composite_score', 0):.3f}")

    return snapshots


def run_experiment(
    decay_type: str = "exponential",
    num_memories: int = 50,
    total_ticks: int = 100,
    eval_interval: int = 5,
    guidance_level: str = "default",
    improvement_budget: int = 5,
    api_key: Optional[str] = None,
    dataset_path: Optional[str] = None,
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
    print(f"Test queries: {len(test_queries)}")

    # Step 3: Build graph
    print("Building memory graph...")
    graph = build_graph_from_dataset(dataset)

    # Step 4: Initial parameters
    params = {
        "lambda_fact": 0.05,
        "lambda_episode": 0.08,
        "beta_fact": 0.3,
        "beta_episode": 0.5,
        "alpha": 0.5,
    }

    # Step 5: Initial run
    engine = DecayEngine(graph, decay_type=decay_type, params=params)
    evaluator = Evaluator(graph, engine)

    print(f"\n=== Initial Run (decay_type={decay_type}) ===")
    initial_snapshots = run_simulation(
        graph, engine, evaluator, test_queries,
        total_ticks=total_ticks, eval_interval=eval_interval,
    )
    initial_score = evaluator.composite_score(test_queries)
    print(f"Initial composite score: {initial_score:.4f}")

    # Step 6: Auto-improvement loop
    improvement_history = []
    if improvement_budget > 0 and api_key:
        print(f"\n=== Auto-Improvement ({guidance_level}, budget={improvement_budget}) ===")
        improver = AutoImprover(
            api_key=api_key,
            guidance_level=guidance_level,
        )

        all_eval_history = evaluator.history[:]

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
            graph2 = build_graph_from_dataset(dataset, embedder=graph._get_embedder())
            engine2 = DecayEngine(graph2, decay_type=decay_type, params=new_params)
            evaluator2 = Evaluator(graph2, engine2)

            run_simulation(
                graph2, engine2, evaluator2, test_queries,
                total_ticks=total_ticks, eval_interval=eval_interval,
            )
            score = evaluator2.composite_score(test_queries)
            print(f"  Composite score: {score:.4f}")

            improvement_history.append({
                "iteration": i,
                "params": new_params,
                "composite_score": score,
                "snapshots": evaluator2.history,
            })

            all_eval_history.extend(evaluator2.history)
            params = new_params

            # Clean up to free memory
            del graph2, engine2, evaluator2

    # Step 7: Output results
    results = {
        "decay_type": decay_type,
        "num_memories": len(dataset),
        "total_ticks": total_ticks,
        "initial_params": params,
        "initial_snapshots": initial_snapshots,
        "initial_composite_score": initial_score,
        "improvement_history": improvement_history,
        "guidance_level": guidance_level,
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
    parser.add_argument("--guidance", choices=["minimal", "default", "expert"], default="default")
    parser.add_argument("--budget", type=int, default=5, help="Auto-improvement iterations")
    parser.add_argument("--dataset", type=str, default=None, help="Path to JSONL dataset")
    parser.add_argument("--output", type=str, default="data/results.json", help="Output path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiment(
        decay_type=args.decay_type,
        num_memories=args.num_memories,
        total_ticks=args.total_ticks,
        eval_interval=args.eval_interval,
        guidance_level=args.guidance,
        improvement_budget=args.budget,
        dataset_path=args.dataset,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
