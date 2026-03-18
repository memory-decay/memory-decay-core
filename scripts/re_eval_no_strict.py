import sys, os, json, importlib.util
from memory_decay.main import build_graph_from_dataset, run_simulation
from memory_decay.decay import DecayEngine
from memory_decay.evaluator import Evaluator
from memory_decay.graph import MemoryGraph

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def run_it(exp_dir):
    print(f"\n--- {exp_dir} ---")
    if not os.path.exists(exp_dir): return
    spec = importlib.util.spec_from_file_location("decay_fn", os.path.join(exp_dir, "decay_fn.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with open(os.path.join(exp_dir, "params.json")) as f: params = json.load(f)
    dataset = load_jsonl("data/memories_500.jsonl")
    queries = [(m["recall_query"], m["id"]) for m in dataset if "recall_query" in m]
    rehearsal_targets = [m["id"] for m in dataset] # just use all for testing
    graph = build_graph_from_dataset(dataset, embedding_backend="local")
    engine = DecayEngine(graph, decay_type="exponential", params=params, custom_decay_fn=mod.compute_decay)
    evaluator = Evaluator(graph, engine)
    run_simulation(graph, engine, evaluator, queries, reactivation_policy="scheduled_query", rehearsal_targets=rehearsal_targets, total_ticks=200, eval_interval=20)
    s = evaluator.score_summary(queries)
    print(f"Overall: {s['overall_score']:.4f} | Ret: {s['retrieval_score']:.4f} | Plaus: {s['plausibility_score']:.4f}")
    print(f"Rec: {s['recall_mean']:.4f} | MRR: {s['mrr_mean']:.4f} | PrecStrict: {s['precision_strict']:.4f} | PrecLift: {s.get('precision_lift', 0):.4f}")

for exp in ["experiments/exp_0000", "experiments/exp_0004", "experiments/exp_0259", "experiments/exp_0294", "experiments/exp_0295"]:
    run_it(exp)
