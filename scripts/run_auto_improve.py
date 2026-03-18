"""Auto-improvement experiment script for Claude Code.

Run with:
  source .venv/bin/activate
  export $(cat .env | xargs)
  PYTHONPATH=src python3 scripts/run_auto_improve.py

Results saved to: data/auto_improvement_results.json
When done, this script writes a sentinel file: /tmp/auto_improve_DONE
"""

import json
import os
import sys
from memory_decay.main import build_graph_from_dataset, run_simulation
from memory_decay.decay import DecayEngine
from memory_decay.evaluator import Evaluator
from memory_decay.auto_improver import AutoImprover
from memory_decay.data_gen import SyntheticDataGenerator

DATASET_PATH = "data/memories_500.jsonl"
RESULTS_PATH = "data/auto_improvement_results.json"
SENTINEL = "/tmp/auto_improve_DONE"
TOTAL_TICKS = 200
EVAL_INTERVAL = 20
BUDGET = 12  # improvement rounds
REACTIVATION_POLICY = "scheduled_query"


def main():
    # Remove sentinel if exists
    if os.path.exists(SENTINEL):
        os.remove(SENTINEL)

    print("Loading dataset...")
    dataset = SyntheticDataGenerator.load_jsonl(DATASET_PATH)
    train, test = SyntheticDataGenerator(api_key="dummy").split_test_train(dataset, test_ratio=0.2, seed=42)
    test_queries = [(m["recall_query"], m["id"]) for m in test if "recall_query" in m]
    rehearsal_targets = [m["id"] for m in train]
    print(f"Dataset: {len(dataset)}, Test queries: {len(test_queries)}")

    guidance_levels = ["minimal", "default", "expert"]
    results = {}

    for level in guidance_levels:
        print(f"\n{'='*50}")
        print(f"Guidance level: {level}")
        print(f"{'='*50}")

        # --- Baseline run ---
        graph = build_graph_from_dataset(dataset)
        engine = DecayEngine(graph, decay_type="exponential")
        evaluator = Evaluator(graph, engine)

        baseline_summaries = run_simulation(
            graph,
            engine,
            evaluator,
            test_queries,
            total_ticks=TOTAL_TICKS,
            eval_interval=EVAL_INTERVAL,
            reactivation_policy=REACTIVATION_POLICY,
            rehearsal_targets=rehearsal_targets,
            seed=42,
        )
        baseline_summary = baseline_summaries[-1]
        baseline_score = baseline_summary["overall_score"]
        print(f"Baseline overall: {baseline_score:.4f}")

        # --- Improvement loop ---
        # Cache the embedder from the first graph so we reuse it
        embedder = graph._get_embedder()

        improver = AutoImprover(guidance_level=level)
        params = engine.get_params()
        all_eval_history = baseline_summaries[:]
        improvement_log = []

        for i in range(BUDGET):
            if improver.should_stop(all_eval_history, i, BUDGET):
                print(f"  Early stop at iteration {i}")
                break

            print(f"\n  Iteration {i + 1}/{BUDGET}:")
            new_params = improver.propose_parameters(params, all_eval_history, i, BUDGET)
            print(f"  Params: {json.dumps(new_params, indent=4)}")

            # Build fresh graph with new params
            graph2 = build_graph_from_dataset(dataset, embedder=embedder)
            engine2 = DecayEngine(graph2, decay_type="exponential", params=new_params)
            evaluator2 = Evaluator(graph2, engine2)

            summaries = run_simulation(
                graph2,
                engine2,
                evaluator2,
                test_queries,
                total_ticks=TOTAL_TICKS,
                eval_interval=EVAL_INTERVAL,
                reactivation_policy=REACTIVATION_POLICY,
                rehearsal_targets=rehearsal_targets,
                seed=42,
            )
            score_summary = summaries[-1]
            score = score_summary["overall_score"]
            print(f"  Overall score: {score:.4f}")

            improvement_log.append({
                "iteration": i,
                "params": new_params,
                "score_summary": score_summary,
                "overall_score": score,
                "threshold_metrics": score_summary["threshold_metrics"],
            })

            all_eval_history.extend(summaries)
            params = new_params
            del graph2, engine2, evaluator2

        # Final score is the last iteration's score (or baseline if no iterations ran)
        final_score = improvement_log[-1]["overall_score"] if improvement_log else baseline_score
        print(f"Final overall: {final_score:.4f} (delta: {final_score - baseline_score:+.4f})")

        results[level] = {
            "baseline_score": baseline_score,
            "baseline_summary": baseline_summary,
            "final_score": final_score,
            "improvement_delta": final_score - baseline_score,
            "rounds": len(improvement_log),
            "log": improvement_log,
            "reactivation_policy": REACTIVATION_POLICY,
        }

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Write sentinel
    with open(SENTINEL, "w") as f:
        f.write(json.dumps({k: v["improvement_delta"] for k, v in results.items()}))

    print(f"Sentinel written to {SENTINEL}")


if __name__ == "__main__":
    main()
