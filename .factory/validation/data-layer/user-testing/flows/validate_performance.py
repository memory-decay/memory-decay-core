"""Validation script for VAL-DATA-018 (cold load perf) and VAL-DATA-019 (memory efficiency)."""
import sys
import os
import time
import tracemalloc
import json
from datetime import datetime, timezone

# Add project root to path so we can import dashboard modules
sys.path.insert(0, "/home/roach/.openclaw/workspace/memory-decay")

from dashboard.data_loader import load_all_experiments

EXPERIMENTS_DIR = "/home/roach/.openclaw/workspace/memory-decay/experiments"
REPORT_PATH = "/home/roach/.openclaw/workspace/memory-decay/.factory/validation/data-layer/user-testing/flows/group-06-performance.json"

results = {
    "groupId": "group-06-performance",
    "testedAt": datetime.now(timezone.utc).isoformat(),
    "isolation": {
        "workingDirectory": "/home/roach/.openclaw/workspace/memory-decay",
        "pythonVenv": "dashboard/.venv/bin/python",
        "experimentsDir": "experiments/",
        "importModule": "dashboard.data_loader.load_all_experiments",
    },
    "toolsUsed": ["python"],
    "assertions": [],
    "frictions": [],
    "blockers": [],
    "summary": "",
}

# ---- Force a fresh import to avoid cached module state ----
# We want a genuine cold-load measurement.
import importlib
import dashboard.data_loader as dl_module
importlib.reload(dl_module)

# =========================================================
# VAL-DATA-018: Cold Load Performance (< 5 seconds)
# =========================================================
print("=== VAL-DATA-018: Cold Load Performance ===")

# Start tracemalloc before loading (also used for VAL-DATA-019)
tracemalloc.start()

# Cold start timing
start_time = time.perf_counter()
experiments = dl_module.load_all_experiments(EXPERIMENTS_DIR)
end_time = time.perf_counter()

elapsed_seconds = end_time - start_time
exp_count = len(experiments)
avg_per_exp_ms = (elapsed_seconds / exp_count) * 1000 if exp_count > 0 else 0

print(f"  Loaded {exp_count} experiments")
print(f"  Wall-clock time: {elapsed_seconds:.4f}s")
print(f"  Per-experiment average: {avg_per_exp_ms:.4f}ms")

val018_status = "pass" if elapsed_seconds < 5.0 else "fail"
val018_evidence = (
    f"Loaded {exp_count} experiments in {elapsed_seconds:.4f}s "
    f"(avg {avg_per_exp_ms:.4f}ms/experiment). "
    f"Threshold: < 5.0s. Result: {'PASS' if val018_status == 'pass' else 'FAIL'}."
)

print(f"  VAL-DATA-018: {val018_status.upper()}")

results["assertions"].append({
    "id": "VAL-DATA-018",
    "title": "Cold Load Performance",
    "status": val018_status,
    "evidence": val018_evidence,
    "steps": [
        {
            "action": "Reload data_loader module to ensure cold state",
            "expected": "Module reloaded successfully",
            "observed": "Module reloaded successfully"
        },
        {
            "action": "Start tracemalloc",
            "expected": "Memory tracking started",
            "observed": "Memory tracking started"
        },
        {
            "action": "Call load_all_experiments() with time.perf_counter()",
            "expected": "All experiments loaded",
            "observed": f"Loaded {exp_count} experiments in {elapsed_seconds:.4f}s"
        },
        {
            "action": "Verify elapsed time < 5.0s",
            "expected": "Time under 5s threshold",
            "observed": f"{elapsed_seconds:.4f}s {'<' if elapsed_seconds < 5.0 else '>='} 5.0s"
        }
    ],
})

# =========================================================
# VAL-DATA-019: Memory Efficiency (< 200 MB peak)
# =========================================================
print("\n=== VAL-DATA-019: Memory Efficiency ===")

# Get peak memory from tracemalloc (started before loading)
current_mem, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

peak_mb = peak_mem / (1024 * 1024)
current_mb = current_mem / (1024 * 1024)

print(f"  Peak memory: {peak_mb:.2f} MB ({peak_mem:,} bytes)")
print(f"  Current memory after load: {current_mb:.2f} MB ({current_mem:,} bytes)")

# Also get RSS via resource for a more realistic measure
import resource
rusage = resource.getrusage(resource.RUSAGE_SELF)
rss_mb = rusage.ru_maxrss / 1024  # Linux: ru_maxrss is in KB
print(f"  Process max RSS: {rss_mb:.2f} MB")

# Check for no redundant raw JSON copies - inspect dataclass instances
# Look at the size of a sample experiment to verify parsing
sample_sizes = []
for exp in experiments[:10]:
    import sys as _sys
    sample_sizes.append(_sys.getsizeof(exp))
avg_obj_size = sum(sample_sizes) / len(sample_sizes) if sample_sizes else 0
print(f"  Average Experiment object size (first 10): {avg_obj_size:.0f} bytes")

val019_status = "pass" if peak_mb < 200 else "fail"
val019_evidence = (
    f"Peak tracemalloc: {peak_mb:.2f} MB. Process max RSS: {rss_mb:.2f} MB. "
    f"Current after load: {current_mb:.2f} MB. "
    f"Threshold: < 200 MB. Result: {'PASS' if val019_status == 'pass' else 'FAIL'}."
)

print(f"  VAL-DATA-019: {val019_status.upper()}")

results["assertions"].append({
    "id": "VAL-DATA-019",
    "title": "Memory Efficiency",
    "status": val019_status,
    "evidence": val019_evidence,
    "steps": [
        {
            "action": "Start tracemalloc before loading",
            "expected": "Memory tracking active",
            "observed": "tracemalloc started before load_all_experiments"
        },
        {
            "action": "Load all experiments",
            "expected": "Data loaded into memory",
            "observed": f"{exp_count} experiments loaded"
        },
        {
            "action": "Get peak memory from tracemalloc",
            "expected": "Peak memory recorded",
            "observed": f"Peak: {peak_mb:.2f} MB, RSS: {rss_mb:.2f} MB"
        },
        {
            "action": "Verify peak < 200 MB",
            "expected": "Under 200 MB threshold",
            "observed": f"{peak_mb:.2f} MB {'<' if peak_mb < 200 else '>='} 200 MB"
        }
    ],
})

# Summary
pass_count = sum(1 for a in results["assertions"] if a["status"] == "pass")
fail_count = sum(1 for a in results["assertions"] if a["status"] == "fail")
total = len(results["assertions"])
results["summary"] = (
    f"Tested {total} assertions: {pass_count} passed, {fail_count} failed. "
    f"VAL-DATA-018: {elapsed_seconds:.4f}s cold load. "
    f"VAL-DATA-019: {peak_mb:.2f} MB peak memory."
)

print(f"\n=== SUMMARY: {results['summary']} ===")

# Write report
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
with open(REPORT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nReport written to: {REPORT_PATH}")
