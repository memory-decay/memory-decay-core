"""Validation script for schema evolution assertions VAL-DATA-006 through VAL-DATA-009."""
import sys
import json
import os

# Add project root to path
sys.path.insert(0, "/home/roach/.openclaw/workspace/memory-decay")
sys.path.insert(0, "/home/roach/.openclaw/workspace/memory-decay/dashboard")

from dashboard.data_loader import load_all_experiments

EXP_DIR = "/home/roach/.openclaw/workspace/memory-decay/experiments"

results = {}

# ===========================================================================
# VAL-DATA-006: Schema Evolution — Missing Fields Default to Null
# ===========================================================================
print("=" * 70)
print("VAL-DATA-006: Schema Evolution — Missing Fields Default to Null")
print("=" * 70)

try:
    experiments = load_all_experiments(EXP_DIR)
    exp_map = {e.id: e for e in experiments}

    exp_0000 = exp_map["exp_0000"]
    exp_lme_0059 = exp_map["exp_lme_0059"]

    fields_to_check = [
        "strict_score", "forgetting_depth", "forgetting_score",
        "eval_v2_score", "retention_curve", "selectivity_score",
        "robustness_score", "threshold_summary"
    ]

    print(f"\nexp_0000 fields check (should all be None):")
    old_nulls = {}
    for f in fields_to_check:
        val = getattr(exp_0000, f)
        old_nulls[f] = val
        print(f"  {f}: {val} (type: {type(val).__name__})")

    print(f"\nexp_lme_0059 fields check (should all be non-None):")
    new_non_nulls = {}
    for f in fields_to_check:
        val = getattr(exp_lme_0059, f)
        new_non_nulls[f] = {"value": val, "type": type(val).__name__}
        print(f"  {f}: {val} (type: {type(val).__name__})")

    # Verify
    old_all_null = all(v is None for v in old_nulls.values())
    new_all_present = all(v["value"] is not None for v in new_non_nulls.values())

    if old_all_null and new_all_present:
        print("\n[PASS] exp_0000 has all null, exp_lme_0059 has all non-null")
        results["VAL-DATA-006"] = {
            "status": "pass",
            "evidence": {
                "exp_0000_nulls": old_nulls,
                "exp_lme_0059_present": {k: v["value"] for k, v in new_non_nulls.items()},
                "old_all_null": old_all_null,
                "new_all_present": new_all_present,
            }
        }
    else:
        failures = []
        if not old_all_null:
            failures.append(f"exp_0000 fields not all null: {old_nulls}")
        if not new_all_present:
            failures.append(f"exp_lme_0059 fields not all present: {new_non_nulls}")
        print(f"\n[FAIL] {failures}")
        results["VAL-DATA-006"] = {
            "status": "fail",
            "evidence": {"failures": failures, "exp_0000_nulls": old_nulls, "exp_lme_0059_present": new_non_nulls}
        }
except Exception as e:
    print(f"\n[ERROR] {e}")
    results["VAL-DATA-006"] = {"status": "fail", "evidence": {"error": str(e)}}

# ===========================================================================
# VAL-DATA-007: Schema Evolution — Threshold Metrics Key Variation
# ===========================================================================
print("\n" + "=" * 70)
print("VAL-DATA-007: Schema Evolution — Threshold Metrics Key Variation")
print("=" * 70)

try:
    experiments = load_all_experiments(EXP_DIR)
    exp_map = {e.id: e for e in experiments}

    exp_0000 = exp_map["exp_0000"]
    exp_lme_0059 = exp_map["exp_lme_0059"]

    # Check old-era thresholds
    old_keys = sorted(exp_0000.threshold_metrics.keys())
    print(f"\nexp_0000 threshold_metrics keys: {old_keys}")
    print(f"  Count: {len(old_keys)}")
    print(f"  Expected: ['0.2', '0.3', '0.4', '0.5'] (4 keys)")

    old_expected = {"0.2", "0.3", "0.4", "0.5"}
    old_keys_set = set(old_keys)

    # Access missing thresholds on old experiment
    threshold_01_old = exp_0000.threshold_metrics.get("0.1")
    threshold_08_old = exp_0000.threshold_metrics.get("0.8")
    print(f"  exp_0000 threshold '0.1': {threshold_01_old}")
    print(f"  exp_0000 threshold '0.8': {threshold_08_old}")

    # Check new-era thresholds
    new_keys = sorted(exp_lme_0059.threshold_metrics.keys())
    print(f"\nexp_lme_0059 threshold_metrics keys: {new_keys}")
    print(f"  Count: {len(new_keys)}")
    print(f"  Expected: 9 thresholds (0.1-0.9)")

    new_expected = {f"0.{i}" for i in range(1, 10)}
    new_keys_set = set(new_keys)

    # Verify
    old_correct = old_keys_set == old_expected
    old_missing_null = threshold_01_old is None and threshold_08_old is None
    new_correct = new_keys_set == new_expected
    new_count_9 = len(new_keys) == 9

    all_pass = old_correct and old_missing_null and new_correct and new_count_9

    if all_pass:
        print("\n[PASS] Old has 4 thresholds (0.2-0.5), missing return None, new has 9 thresholds (0.1-0.9)")
        results["VAL-DATA-007"] = {
            "status": "pass",
            "evidence": {
                "exp_0000_keys": old_keys,
                "exp_0000_count": len(old_keys),
                "exp_0000_threshold_01": threshold_01_old,
                "exp_0000_threshold_08": threshold_08_old,
                "exp_lme_0059_keys": new_keys,
                "exp_lme_0059_count": len(new_keys),
            }
        }
    else:
        failures = []
        if not old_correct:
            failures.append(f"exp_0000 keys mismatch: got {old_keys}, expected {sorted(old_expected)}")
        if not old_missing_null:
            failures.append(f"exp_0000 missing thresholds not None: '0.1'={threshold_01_old}, '0.8'={threshold_08_old}")
        if not new_correct:
            failures.append(f"exp_lme_0059 keys mismatch: got {new_keys}, expected {sorted(new_expected)}")
        if not new_count_9:
            failures.append(f"exp_lme_0059 has {len(new_keys)} thresholds, expected 9")
        print(f"\n[FAIL] {failures}")
        results["VAL-DATA-007"] = {
            "status": "fail",
            "evidence": {"failures": failures}
        }
except Exception as e:
    print(f"\n[ERROR] {e}")
    results["VAL-DATA-007"] = {"status": "fail", "evidence": {"error": str(e)}}

# ===========================================================================
# VAL-DATA-008: Schema Evolution — params.json Field Variation
# ===========================================================================
print("\n" + "=" * 70)
print("VAL-DATA-008: Schema Evolution — params.json Field Variation")
print("=" * 70)

try:
    experiments = load_all_experiments(EXP_DIR)
    exp_map = {e.id: e for e in experiments}

    exp_0000 = exp_map["exp_0000"]
    exp_lme_0059 = exp_map["exp_lme_0059"]

    # Check old-era params
    print(f"\nexp_0000 params keys: {sorted(exp_0000.params.keys())}")
    has_beta_fact = "beta_fact" in exp_0000.params and exp_0000.params["beta_fact"] is not None
    no_floor_max = "floor_max" not in exp_0000.params or exp_0000.params.get("floor_max") is None
    print(f"  has beta_fact (not null): {has_beta_fact} (value: {exp_0000.params.get('beta_fact')})")
    print(f"  no floor_max (null): {no_floor_max} (value: {exp_0000.params.get('floor_max')})")

    # Check new-era params
    print(f"\nexp_lme_0059 params keys: {sorted(exp_lme_0059.params.keys())}")
    has_floor_max = "floor_max" in exp_lme_0059.params and exp_lme_0059.params["floor_max"] is not None
    no_beta_fact = "beta_fact" not in exp_lme_0059.params or exp_lme_0059.params.get("beta_fact") is None
    has_assoc_boost = "assoc_boost" in exp_lme_0059.params
    print(f"  has floor_max: {has_floor_max} (value: {exp_lme_0059.params.get('floor_max')})")
    print(f"  no beta_fact: {no_beta_fact} (value: {exp_lme_0059.params.get('beta_fact')})")
    print(f"  has assoc_boost: {has_assoc_boost} (value: {exp_lme_0059.params.get('assoc_boost')})")

    # Also check exp_lme_0000 per the assertion
    exp_lme_0000 = exp_map["exp_lme_0000"]
    print(f"\nexp_lme_0000 params keys: {sorted(exp_lme_0000.params.keys())}")
    lme0000_has_assoc = "assoc_boost" in exp_lme_0000.params
    lme0000_no_beta = "beta_fact" not in exp_lme_0000.params
    print(f"  has assoc_boost: {lme0000_has_assoc} (value: {exp_lme_0000.params.get('assoc_boost')})")
    print(f"  no beta_fact: {lme0000_no_beta}")

    old_ok = has_beta_fact and no_floor_max
    new_0059_ok = has_floor_max and no_beta_fact
    new_0000_ok = lme0000_has_assoc and lme0000_no_beta

    # Pass if old-era has beta_fact/no floor_max AND either new-era exp has new-style params
    new_ok = new_0059_ok or new_0000_ok

    if old_ok and new_ok:
        print("\n[PASS] Old-era has beta_fact (no floor_max), new-era has new params (no beta_fact)")
        results["VAL-DATA-008"] = {
            "status": "pass",
            "evidence": {
                "exp_0000_params": exp_0000.params,
                "exp_0000_has_beta_fact": has_beta_fact,
                "exp_0000_no_floor_max": no_floor_max,
                "exp_lme_0059_params": exp_lme_0059.params,
                "exp_lme_0059_has_floor_max": has_floor_max,
                "exp_lme_0059_no_beta_fact": no_beta_fact,
                "exp_lme_0000_params": exp_lme_0000.params,
                "exp_lme_0000_has_assoc_boost": lme0000_has_assoc,
                "exp_lme_0000_no_beta_fact": lme0000_no_beta,
            }
        }
    else:
        failures = []
        if not old_ok:
            failures.append(f"exp_0000 params: beta_fact={has_beta_fact}, no_floor_max={no_floor_max}")
        if not new_ok:
            failures.append(f"new-era params not correct: 0059={new_0059_ok}, 0000={new_0000_ok}")
        print(f"\n[FAIL] {failures}")
        results["VAL-DATA-008"] = {
            "status": "fail",
            "evidence": {"failures": failures}
        }
except Exception as e:
    print(f"\n[ERROR] {e}")
    results["VAL-DATA-008"] = {"status": "fail", "evidence": {"error": str(e)}}

# ===========================================================================
# VAL-DATA-009: Snapshots Array Structure
# ===========================================================================
print("\n" + "=" * 70)
print("VAL-DATA-009: Snapshots Array Structure")
print("=" * 70)

try:
    experiments = load_all_experiments(EXP_DIR)
    exp_map = {e.id: e for e in experiments}

    exp_0000 = exp_map["exp_0000"]
    exp_lme_0059 = exp_map["exp_lme_0059"]

    expected_ticks = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    # Check exp_0000 snapshots
    old_snaps = exp_0000.snapshots
    old_tick_vals = [s["tick"] for s in old_snaps]
    print(f"\nexp_0000 snapshots:")
    print(f"  Count: {len(old_snaps)} (expected 11)")
    print(f"  Tick values: {old_tick_vals}")
    print(f"  Expected: {expected_ticks}")
    old_tick_match = old_tick_vals == expected_ticks
    print(f"  Tick match: {old_tick_match}")

    # Old-era snapshot keys
    if old_snaps:
        old_snap_keys = set()
        for s in old_snaps:
            old_snap_keys.update(s.keys())
        print(f"  Snapshot keys ({len(old_snap_keys)}): {sorted(old_snap_keys)}")

        old_no_retention_curve = "retention_curve" not in old_snap_keys
        old_no_threshold_summary = "threshold_summary" not in old_snap_keys
        old_no_eval_v2 = "eval_v2_score" not in old_snap_keys
        old_no_selectivity = "selectivity_score" not in old_snap_keys
        old_no_robustness = "robustness_score" not in old_snap_keys
        print(f"  No retention_curve in snapshots: {old_no_retention_curve}")
        print(f"  No threshold_summary in snapshots: {old_no_threshold_summary}")
        print(f"  No eval_v2_score in snapshots: {old_no_eval_v2}")
        print(f"  No selectivity_score in snapshots: {old_no_selectivity}")
        print(f"  No robustness_score in snapshots: {old_no_robustness}")
    else:
        old_no_retention_curve = False
        old_no_threshold_summary = False
        old_no_eval_v2 = False
        old_no_selectivity = False
        old_no_robustness = False
        print("  WARNING: No snapshots found!")

    # Check exp_lme_0059 snapshots
    new_snaps = exp_lme_0059.snapshots
    new_tick_vals = [s["tick"] for s in new_snaps]
    print(f"\nexp_lme_0059 snapshots:")
    print(f"  Count: {len(new_snaps)} (expected 11)")
    print(f"  Tick values: {new_tick_vals}")
    new_tick_match = new_tick_vals == expected_ticks
    print(f"  Tick match: {new_tick_match}")

    # New-era snapshot keys
    if new_snaps:
        new_snap_keys = set()
        for s in new_snaps:
            new_snap_keys.update(s.keys())
        print(f"  Snapshot keys ({len(new_snap_keys)}): {sorted(new_snap_keys)}")

        new_has_retention_curve = "retention_curve" in new_snap_keys
        new_has_selectivity = "selectivity_score" in new_snap_keys
        print(f"  Has retention_curve in snapshots: {new_has_retention_curve}")
        print(f"  Has selectivity_score in snapshots: {new_has_selectivity}")
    else:
        new_has_retention_curve = False
        new_has_selectivity = False
        print("  WARNING: No snapshots found!")

    # Check experiments with no snapshots -> default to empty list
    no_snapshot_exps = []
    empty_snapshot_exps = []
    for exp in experiments:
        results_path = os.path.join(exp.dir_path, "results.json")
        if os.path.exists(results_path):
            import json as json_mod
            with open(results_path) as f:
                raw = json_mod.load(f)
            if "snapshots" not in raw:
                no_snapshot_exps.append(exp.id)
            if not exp.snapshots:
                empty_snapshot_exps.append(exp.id)

    print(f"\nExperiments with no 'snapshots' field in results.json: {no_snapshot_exps}")
    print(f"Experiments with empty snapshots list after loading: {empty_snapshot_exps}")

    # Check validation_failed experiments default to empty snapshots
    val_failed = [e for e in experiments if e.status == "validation_failed"]
    val_failed_snapshots = {e.id: e.snapshots for e in val_failed}
    print(f"\nvalidation_failed experiments snapshots: {val_failed_snapshots}")

    # Verify assertions
    checks = {
        "old_tick_match": old_tick_match and len(old_snaps) == 11,
        "new_tick_match": new_tick_match and len(new_snaps) == 11,
        "old_era_fewer_keys": old_no_retention_curve and old_no_threshold_summary and old_no_eval_v2 and old_no_selectivity and old_no_robustness,
        "new_era_has_more_keys": new_has_retention_curve or new_has_selectivity,
        "no_snapshot_defaults_empty": len(no_snapshot_exps) == 0 or all(exp_map[eid].snapshots == [] for eid in no_snapshot_exps),
    }

    all_pass = all(checks.values())
    if all_pass:
        print("\n[PASS] All snapshot structure checks passed")
        results["VAL-DATA-009"] = {
            "status": "pass",
            "evidence": {
                "exp_0000_snapshot_count": len(old_snaps),
                "exp_0000_ticks": old_tick_vals,
                "exp_0000_key_count": len(old_snap_keys) if old_snaps else 0,
                "exp_0000_keys": sorted(old_snap_keys) if old_snaps else [],
                "exp_0000_missing_new_keys": {
                    "retention_curve": not old_no_retention_curve,
                    "threshold_summary": not old_no_threshold_summary,
                    "eval_v2_score": not old_no_eval_v2,
                    "selectivity_score": not old_no_selectivity,
                    "robustness_score": not old_no_robustness,
                },
                "exp_lme_0059_snapshot_count": len(new_snaps),
                "exp_lme_0059_ticks": new_tick_vals,
                "exp_lme_0059_key_count": len(new_snap_keys) if new_snaps else 0,
                "exp_lme_0059_keys": sorted(new_snap_keys) if new_snaps else [],
                "exp_lme_0059_has_new_keys": {
                    "retention_curve": new_has_retention_curve,
                    "selectivity_score": new_has_selectivity,
                },
                "no_field_experiments_default_empty": no_snapshot_exps,
                "validation_failed_snapshots": val_failed_snapshots,
                "checks": checks,
            }
        }
    else:
        failures = [f"{k}={v}" for k, v in checks.items() if not v]
        print(f"\n[FAIL] Failed checks: {failures}")
        results["VAL-DATA-009"] = {
            "status": "fail",
            "evidence": {"failures": failures, "checks": checks}
        }
except Exception as e:
    import traceback
    print(f"\n[ERROR] {e}")
    traceback.print_exc()
    results["VAL-DATA-009"] = {"status": "fail", "evidence": {"error": str(e)}}

# ===========================================================================
# Output summary
# ===========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for aid, aresult in results.items():
    print(f"  {aid}: {aresult['status']}")

# Write results to stdout for capture
print("\n---JSON_RESULTS---")
print(json.dumps(results, indent=2, default=str))
