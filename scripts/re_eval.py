import sys
import os
import json
import importlib.util
from pathlib import Path
from memory_decay.main import build_graph_from_dataset
from memory_decay.decay import DecayEngine
from memory_decay.evaluator import Evaluator
from memory_decay.runner import SimulationRunner
from memory_decay.graph import MemoryGraph

def re_evaluate(exp_dir):
    fn_path = os.path.join(exp_dir, "decay_fn.py")
    spec = importlib.util.spec_from_file_location("decay_fn", fn_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    custom_decay = module.compute_decay

    params_path = os.path.join(exp_dir, "params.json")
    with open(params_path, "r") as f:
        params = json.load(f)

    dataset_path = "data/memories_500.jsonl"
    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))

    test_queries = []
    for item in dataset:
        if "recall_query" in item and "id" in item:
            test_queries.append((item["recall_query"], item["id"]))

    graph = MemoryGraph(embedding_backend="local")
    engine = DecayEngine(graph, decay_type="exponential", params=params, custom_decay_fn=custom_decay)
    
    runner = SimulationRunner(
        graph=graph,
        engine=engine,
        memories=dataset,
        test_queries=test_queries,
        reactivation_policy="scheduled_query",
        total_ticks=200,
        eval_interval=20
    )
    runner.run()
    
    # We can just get score_summary from runner._evaluator
    score = runner._evaluator.score_summary(test_queries)
    
    print(f"[{os.path.basename(exp_dir)}]")
    print(f"Overall: {score['overall_score']:.4f}")
    print(f"Retrieval: {score['retrieval_score']:.4f}")
    print(f"Plausibility: {score['plausibility_score']:.4f}")
    print(f"Recall: {score['recall_mean']:.4f}")
    print(f"Precision Lift: {score.get('precision_lift', 0):.4f}")
    print(f"Precision Strict: {score['precision_strict']:.4f}")
    print(f"MRR: {score['mrr_mean']:.4f}")
    print(f"Smoothness: {score['smoothness_score']:.4f}")
    print("-" * 40)

if __name__ == "__main__":
    for exp in ["experiments/exp_0004", "experiments/exp_0163", "experiments/exp_0259"]:
        if os.path.exists(exp):
            print(f"Re-evaluating {exp}...")
            re_evaluate(exp)
