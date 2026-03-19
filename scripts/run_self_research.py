"""M27-style self-research loop for memory-decay.

Combines:
- Memory chain (persistent round-by-round learnings)
- Self-critique agent (failure trajectory analysis)
- Hypothesis generator (chain-aware hypothesis formation)
- Existing experiment runner (program.md protocol)

Each round:
1. Read memory chain summary
2. Self-critique analyzes recent failure patterns
3. Hypothesis generator proposes next experiment
4. Run experiment via existing runner
5. Write new memory chain round
6. Judge via Stage A / Stage B (from program.md)
"""

import json
import os
import subprocess
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEMORY_CHAIN = ROOT / "memory_chain"
EXP_DIR = ROOT / "experiments"
HISTORY = EXP_DIR / "history.jsonl"
BEST_DIR = EXP_DIR / "best"
CACHE_DIR = ROOT / "cache"

sys.path.insert(0, str(ROOT / "src"))
from memory_decay.memory_chain_indexer import MemoryChainIndexer, RoundData
from memory_decay.self_critic import SelfCritic
from memory_decay.hypothesis_generator import HypothesisGenerator


def run_experiment(exp_name: str, decay_fn_code: str, params: dict) -> dict | None:
    """Run a single experiment and return results."""
    exp_path = EXP_DIR / exp_name
    exp_path.mkdir(parents=True, exist_ok=True)

    (exp_path / "decay_fn.py").write_text(decay_fn_code)
    (exp_path / "params.json").write_text(json.dumps(params, indent=2))

    result = subprocess.run(
        ["uv", "run", "python", "-m", "memory_decay.runner", str(exp_path), "--cache", str(CACHE_DIR)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        timeout=120,
    )

    results_file = exp_path / "results.json"
    if results_file.exists():
        return json.loads(results_file.read_text())
    else:
        print(f"  FAILED: {result.stderr[-300:]}")
        return None


def judge_experiment(exp_path: Path, best_score: float) -> tuple[str, dict]:
    """Judge an experiment using Stage A (fixed-split) from program.md."""
    results = json.loads((exp_path / "results.json").read_text())
    overall = results.get("overall_score", 0)

    if overall > best_score:
        status = "improved"
    else:
        status = "no_gain"

    return status, results


def main():
    print("=" * 60)
    print("M27-Style Self-Research Loop — Memory Decay")
    print("=" * 60)

    idx = MemoryChainIndexer(MEMORY_CHAIN)
    critic = SelfCritic()
    generator = HypothesisGenerator()

    # Find next round number
    start_round = (idx.latest_round_number() or -1) + 1

    # Resolve best symlink target
    best_link_target = BEST_DIR.resolve() if BEST_DIR.is_symlink() else BEST_DIR
    try:
        best_results = json.loads((best_link_target / "results.json").read_text())
        best_score = best_results.get("overall_score", 0)
    except (json.JSONDecodeError, OSError):
        best_score = 0
        best_results = {"overall_score": 0}
    try:
        best_decay_fn = (best_link_target / "decay_fn.py").read_text()
    except OSError:
        best_decay_fn = ""
    try:
        best_params = json.loads((best_link_target / "params.json").read_text())
    except (json.JSONDecodeError, OSError):
        best_params = {}

    # Find next available experiment number (avoid collisions)
    existing_exp_nums = []
    for d in EXP_DIR.glob("exp_????"):
        try:
            existing_exp_nums.append(int(d.name.split("_")[1]))
        except ValueError:
            pass
    next_exp_num = max(existing_exp_nums) + 1 if existing_exp_nums else 0

    print(f"\nStarting at round {start_round}")
    print(f"Current best: {BEST_DIR.name} (overall={best_score:.4f})")

    max_rounds = 20
    improvements = 0

    for round_num in range(start_round, start_round + max_rounds):
        exp_name = f"exp_{next_exp_num:04d}"
        next_exp_num += 1
        print(f"\n{'=' * 60}")
        print(f"Round {round_num}: {exp_name}")

        # Step 1: Self-critique
        print("  [1/4] Running self-critique...")
        try:
            critique = critic.critique_from_chain(MEMORY_CHAIN, HISTORY, round_num)
            print(f"  Critique: {critique.get('diagnosis', 'N/A')[:100]}")
        except Exception as e:
            print(f"  Critique failed: {e}, using default direction")
            critique = {"next_direction": "Continue parameter refinement", "observations": []}

        # Step 2: Hypothesis generation
        print("  [2/4] Generating hypothesis...")
        try:
            chain_context = idx.read_chain_summary(from_round=max(0, round_num - 10))
            hyp_result = generator.generate_hypothesis(chain_context, best_decay_fn, best_params, round_num)
            hypothesis = hyp_result.get("hypothesis", critique.get("next_direction", ""))
            decay_fn_code = hyp_result.get("decay_fn_code", best_decay_fn)
            new_params = hyp_result.get("params", best_params)
        except Exception as e:
            print(f"  Hypothesis generation failed: {e}")
            hypothesis = critique.get("next_direction", "Continue refinement")
            decay_fn_code = best_decay_fn
            new_params = best_params

        # Step 3: Run experiment
        print("  [3/4] Running experiment...")
        results = run_experiment(exp_name, decay_fn_code, new_params)
        if results is None:
            record = {"experiment": exp_name, "status": "failed", "round": round_num}
            with open(HISTORY, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            continue

        overall = results.get("overall_score", 0)
        print(f"  Results: overall={overall:.4f}")

        # Step 4: Judge
        print("  [4/4] Judging...")
        status, _ = judge_experiment(EXP_DIR / exp_name, best_score)
        delta = overall - best_score
        print(f"  Status: {status} (delta={delta:+.4f})")

        # Record to history
        record = {
            "experiment": exp_name,
            "overall_score": round(overall, 4),
            "retrieval_score": round(results.get("retrieval_score", 0), 4),
            "plausibility_score": round(results.get("plausibility_score", 0), 4),
            "status": status,
            "round": round_num,
            "hypothesis": hypothesis,
        }
        with open(HISTORY, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Write memory chain round using RoundData dataclass
        data = RoundData(
            round_num=round_num,
            experiment=exp_name,
            scores={
                "overall_score": overall,
                "retrieval_score": results.get("retrieval_score", 0),
                "plausibility_score": results.get("plausibility_score", 0),
            },
            hypothesis=hypothesis,
            observations=critique.get("observations", []),
            decisions=[f"Status: {status}, delta={delta:+.4f}"],
            open_questions=["Is this the global optimum?"],
            next_direction=critique.get("next_direction", ""),
            parent_round=round_num - 1 if round_num > 0 else None,
        )
        idx.write_round(data)

        if status == "improved":
            best_score = overall
            best_decay_fn = decay_fn_code
            best_params = new_params
            improvements += 1
            # Update best symlink: remove whatever exists, create fresh symlink
            BEST_DIR.unlink(missing_ok=True)
            if BEST_DIR.is_dir():
                shutil.rmtree(BEST_DIR)
            BEST_DIR.symlink_to(exp_name)
            print(f"  *** NEW BEST: {exp_name} ({overall:.4f}) ***")

    print(f"\n{'=' * 60}")
    print(f"M27-style loop complete. {improvements} improvements found.")
    print(f"Final best: {best_score:.4f}")


if __name__ == "__main__":
    main()