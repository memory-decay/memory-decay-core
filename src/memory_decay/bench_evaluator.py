"""MemoryBench-based evaluator for the auto-improvement loop.

Replaces the internal evaluator with end-to-end MemoryBench evaluation
across 3 benchmarks (LongMemEval, LoCoMo, ConvoMem).

The embedding cache (cache/openai/) is pre-built — no embedding API
calls during evaluation. Only answer/judge calls cost money.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

BENCHMARKS = ["longmemeval", "locomo", "convomem"]
WEIGHTS = {"longmemeval": 0.50, "locomo": 0.30, "convomem": 0.20}

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MEMORYBENCH_DIR = Path(os.environ.get("MEMORYBENCH_DIR", str(Path.home() / "memorybench")))
SERVER_VENV_PYTHON = os.environ.get("SERVER_VENV_PYTHON", str(REPO_ROOT / ".venv" / "bin" / "python"))
CACHE_DIR = os.environ.get("MEMORYBENCH_CACHE_DIR", str(REPO_ROOT / "cache" / "openai"))


@dataclass
class BenchResult:
    """Result from a single benchmark run."""
    benchmark: str
    accuracy: float
    total: int
    correct: int
    mrr: float = 0.0
    hit_at_k: float = 0.0
    run_id: str = ""


@dataclass
class CompositeResult:
    """Composite result across all benchmarks."""
    bench_score: float
    results: dict[str, BenchResult] = field(default_factory=dict)
    run_prefix: str = ""

    def to_dict(self) -> dict:
        return {
            "bench_score": round(self.bench_score, 4),
            "benchmarks": {
                name: {
                    "accuracy": round(r.accuracy, 4),
                    "total": r.total,
                    "correct": r.correct,
                    "mrr": round(r.mrr, 4),
                    "run_id": r.run_id,
                }
                for name, r in self.results.items()
            },
        }


def _kill_server() -> None:
    """Kill any running memory-decay server on port 8100."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", ":8100"],
            capture_output=True, text=True, timeout=5,
        )
        for pid in result.stdout.strip().split("\n"):
            if pid.strip():
                os.kill(int(pid.strip()), signal.SIGTERM)
        time.sleep(1)
    except Exception:
        pass


def _start_server(experiment_dir: str | Path) -> subprocess.Popen:
    """Start memory-decay server with given experiment."""
    _kill_server()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    proc = subprocess.Popen(
        [
            SERVER_VENV_PYTHON, "-m", "memory_decay.server",
            "--port", "8100",
            "--cache-dir", CACHE_DIR,
            "--embedding-provider", "openai",
            "--embedding-api-key", api_key,
            "--experiment-dir", str(experiment_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(REPO_ROOT),
    )

    # Wait for server to be ready
    for _ in range(30):
        time.sleep(1)
        try:
            import urllib.request
            with urllib.request.urlopen("http://localhost:8100/health", timeout=2) as resp:
                if resp.status == 200:
                    return proc
        except Exception:
            pass

    proc.kill()
    raise RuntimeError("Server failed to start within 30 seconds")


def _run_benchmark(
    benchmark: str,
    run_id: str,
    sample_per_category: int,
    judge: str = "gpt-4o",
    answering_model: str = "gpt-4o",
) -> BenchResult:
    """Run a single MemoryBench benchmark and parse results."""
    cmd = [
        "bun", "run", "src/index.ts", "run",
        "-p", "memory-decay",
        "-b", benchmark,
        "-j", judge,
        "-m", answering_model,
        "-r", run_id,
        "-s", str(sample_per_category),
        "--sample-type", "random",
        "--force",
    ]

    env = {**os.environ}
    result = subprocess.run(
        cmd,
        capture_output=True, text=True,
        cwd=str(MEMORYBENCH_DIR),
        timeout=1800,  # 30 min — LongMemEval has ~48 sessions/question
        env=env,
    )

    if result.returncode != 0:
        print(f"  [WARN] {benchmark} run failed: {result.stderr[-500:]}")
        return BenchResult(benchmark=benchmark, accuracy=0.0, total=0, correct=0, run_id=run_id)

    # Parse report
    report_path = MEMORYBENCH_DIR / "data" / "runs" / run_id / "report.json"
    if not report_path.exists():
        print(f"  [WARN] No report at {report_path}")
        return BenchResult(benchmark=benchmark, accuracy=0.0, total=0, correct=0, run_id=run_id)

    report = json.loads(report_path.read_text())
    summary = report.get("summary", {})
    retrieval = report.get("retrieval", {})

    raw_acc = summary.get("accuracy", 0.0)
    # MemoryBench returns accuracy as 0-1 ratio (e.g., 0.55 = 55%)
    accuracy = raw_acc if raw_acc <= 1.0 else raw_acc / 100.0

    return BenchResult(
        benchmark=benchmark,
        accuracy=accuracy,
        total=summary.get("totalQuestions", 0),
        correct=summary.get("correctCount", 0),
        mrr=retrieval.get("mrr", 0.0),
        hit_at_k=retrieval.get("hitAtK", 0.0),
        run_id=run_id,
    )


def evaluate(
    experiment_dir: str | Path,
    run_prefix: str,
    sample_per_category: int = 5,
    benchmarks: list[str] | None = None,
    judge: str = "gpt-4o",
    answering_model: str = "gpt-4o",
) -> CompositeResult:
    """Run MemoryBench evaluation for an experiment.

    Args:
        experiment_dir: Path to experiment directory (with decay_fn.py + params.json)
        run_prefix: Prefix for run IDs (e.g., "exp_bench_0001")
        sample_per_category: Number of questions to randomly sample per category
        benchmarks: Which benchmarks to run (default: all 3)
        judge: Judge model
        answering_model: Answering model

    Returns:
        CompositeResult with weighted composite score
    """
    if benchmarks is None:
        benchmarks = BENCHMARKS

    print(f"Starting server with {experiment_dir}...")
    server_proc = _start_server(experiment_dir)

    try:
        results: dict[str, BenchResult] = {}
        for bench in benchmarks:
            run_id = f"{run_prefix}-{bench}"
            print(f"  Running {bench} (sample_per_category={sample_per_category})...")
            t0 = time.time()
            results[bench] = _run_benchmark(
                bench, run_id, sample_per_category, judge, answering_model,
            )
            elapsed = time.time() - t0
            r = results[bench]
            print(f"  {bench}: accuracy={r.accuracy*100:.1f}% ({r.correct}/{r.total}) [{elapsed:.0f}s]")

        # Compute weighted score
        bench_score = sum(
            WEIGHTS.get(bench, 0.0) * results[bench].accuracy
            for bench in benchmarks
        )

        return CompositeResult(
            bench_score=bench_score,
            results=results,
            run_prefix=run_prefix,
        )
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait(timeout=3)


def evaluate_cached(
    experiment_dir: str | Path,
    run_prefix: str,
    cached_run_ids: dict[str, str] | None = None,
    sample_per_category: int = 5,
    judge: str = "gpt-4o",
    answering_model: str = "gpt-4o",
) -> CompositeResult:
    """Evaluate using cached run IDs where available.

    For benchmarks with cached run IDs, reads existing report.json
    instead of re-running. Used for Stage A screening with reusable samples.
    """
    results: dict[str, BenchResult] = {}
    exp_dir = Path(experiment_dir)

    if cached_run_ids:
        for bench, run_id in cached_run_ids.items():
            report_path = MEMORYBENCH_DIR / "data" / "runs" / run_id / "report.json"
            if report_path.exists():
                report = json.loads(report_path.read_text())
                summary = report.get("summary", {})
                retrieval = report.get("retrieval", {})
                raw_acc = summary.get("accuracy", 0.0)
                results[bench] = BenchResult(
                    benchmark=bench,
                    accuracy=raw_acc if raw_acc <= 1.0 else raw_acc / 100.0,
                    total=summary.get("totalQuestions", 0),
                    correct=summary.get("correctCount", 0),
                    mrr=retrieval.get("mrr", 0.0),
                    run_id=run_id,
                )

    for bench in BENCHMARKS:
        if bench in results:
            continue
        result_file = exp_dir / f"{run_prefix}_{bench}_result.json"
        if result_file.exists():
            cached = json.loads(result_file.read_text())
            summary = cached.get("summary", {})
            retrieval = cached.get("retrieval", {})
            raw_acc = summary.get("accuracy", 0.0)
            results[bench] = BenchResult(
                benchmark=bench,
                accuracy=raw_acc if raw_acc <= 1.0 else raw_acc / 100.0,
                total=summary.get("totalQuestions", 0),
                correct=summary.get("correctCount", 0),
                mrr=retrieval.get("mrr", 0.0),
                run_id=f"{run_prefix}-{bench}",
            )

    benchmarks_to_run = [b for b in BENCHMARKS if b not in results]

    if benchmarks_to_run:
        fresh = evaluate(
            experiment_dir, run_prefix, sample_per_category,
            benchmarks=benchmarks_to_run,
            judge=judge,
            answering_model=answering_model,
        )
        results.update(fresh.results)

    bench_score = sum(
        WEIGHTS.get(bench, 0.0) * results[bench].accuracy
        for bench in BENCHMARKS if bench in results
    )

    return CompositeResult(
        bench_score=bench_score,
        results=results,
        run_prefix=run_prefix,
    )
