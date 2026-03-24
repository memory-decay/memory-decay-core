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

import json
import math
import random
import re
from collections import Counter
from typing import Optional

from .graph import MemoryGraph
from .decay import DecayEngine
from .evaluator import Evaluator


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
    fast_eval: bool = False,
) -> list[dict]:
    """Run a simulation and return evaluation snapshots.

    Args:
        graph: MemoryGraph with loaded memories.
        engine: DecayEngine configured with decay parameters.
        evaluator: Evaluator instance.
        test_queries: List of (query, expected_id) pairs.
        total_ticks: Total simulation ticks.
        eval_interval: Evaluate every N ticks.
        reactivation_policy: Re-activation policy: none, random, scheduled_query,
            or scheduled_query_all (includes test memories in reactivation).
        reactivation_interval: Apply the selected policy every N ticks.
        reactivation_boost: Activation boost for direct re-activation.
        rehearsal_targets: Memory IDs eligible for scheduled_query reactivation.
            Required when reactivation_policy is "scheduled_query" or
            "scheduled_query_all".
        seed: Random seed used by the random policy.

    Returns:
        List of evaluation summaries.
    """
    if reactivation_policy not in {"none", "random", "scheduled_query", "scheduled_query_all", "scheduled_query_plus_test", "retrieval_consolidation"}:
        raise ValueError(f"Unsupported reactivation_policy: {reactivation_policy}")

    if reactivation_policy in ("scheduled_query", "scheduled_query_all", "scheduled_query_plus_test", "retrieval_consolidation") and not rehearsal_targets:
        raise ValueError(
            "rehearsal_targets must be provided when reactivation_policy is "
            "'scheduled_query', 'scheduled_query_all', 'scheduled_query_plus_test', "
            "or 'retrieval_consolidation'"
        )

    # For policies that involve scheduled_query, extract test memory IDs
    test_memory_ids: set[str] = {q[1] for q in test_queries}
    train_memory_ids: set[str] = set(rehearsal_targets) if rehearsal_targets else set()

    rng = random.Random(seed)
    summaries = []
    params = engine.get_params()

    def collect_summary() -> dict:
        return evaluator.score_summary(test_queries, record_snapshot=True)

    def collect_snapshot() -> dict:
        """Lightweight snapshot: only recall/precision for history tracking."""
        return evaluator.snapshot(test_queries, record=True)

    def _get_scheduled_target(rehearsal_list: list[str], tick: int, interval: int) -> str | None:
        """Return the scheduled target for a given tick and interval."""
        eligible = [
            mid for mid in rehearsal_list
            if graph._graph.nodes[mid].get("created_tick", 0) <= engine.current_tick
        ]
        if not eligible:
            return None
        idx = ((tick // interval) - 1) % len(eligible)
        return eligible[idx]

    def _importance_for(node_id: str) -> float:
        node = graph._graph.nodes.get(node_id)
        if not node or node.get("type") in ("unknown", None):
            return 0.0

        alpha = float(params.get("alpha", 1.0))
        rho = float(params.get("stability_weight", 0.0))
        denom = alpha + rho
        if denom <= 0.0:
            return 0.0

        impact = max(float(node.get("impact", 0.0)), 0.0)
        stability = max(float(node.get("stability_score", 0.0)), 0.0)
        importance = (impact * alpha + stability * rho) / denom
        return min(max(importance, 0.0), 1.0)

    def _scaled_boost(node_id: str, base_boost: float) -> float:
        boost = float(base_boost)
        if not (
            params.get("importance_scaled_boost", False)
            or params.get("importance_scaled_retrieval_boost", False)
        ):
            return boost

        min_scale = max(float(params.get("importance_boost_min_scale", 0.5)), 0.0)
        max_scale = max(float(params.get("importance_boost_max_scale", 1.0)), min_scale)
        importance = _importance_for(node_id)
        scale = min_scale + (max_scale - min_scale) * importance
        return boost * scale

    def _lexical_tokens(text: str) -> list[str]:
        return re.findall(r"[0-9A-Za-z가-힣]+", text.lower())

    def _bm25_scores(query_text: str, candidate_ids: list[str]) -> dict[str, float]:
        query_terms = list(dict.fromkeys(_lexical_tokens(query_text)))
        if not query_terms or not candidate_ids:
            return {}

        tokenized_docs: dict[str, list[str]] = {}
        for cid in candidate_ids:
            node = graph.get_node(cid)
            content = node.get("content", "") if node else ""
            tokenized_docs[cid] = _lexical_tokens(content)

        avgdl = sum(len(tokens) for tokens in tokenized_docs.values()) / max(len(tokenized_docs), 1)
        avgdl = max(avgdl, 1.0)

        doc_freq: Counter[str] = Counter()
        for tokens in tokenized_docs.values():
            doc_freq.update(set(tokens))

        scores: dict[str, float] = {}
        k1 = 1.2
        b = 0.75
        n_docs = len(tokenized_docs)
        for cid, tokens in tokenized_docs.items():
            tf = Counter(tokens)
            dl = max(len(tokens), 1)
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                df = doc_freq.get(term, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = freq + k1 * (1.0 - b + b * dl / avgdl)
                score += idf * (freq * (k1 + 1.0)) / max(denom, 1e-9)
            scores[cid] = score
        return scores

    def _apply_fractional_hybrid_reinforcement(
        expected_id: str,
        *,
        retrieval_amount: float,
        storage_scale: float,
        current_tick: int,
    ) -> None:
        graph.re_activate(
            expected_id,
            retrieval_amount,
            source="retrieval_consolidation",
            reinforce=False,
            current_tick=current_tick,
            reinforcement_gain_direct=params["reinforcement_gain_direct"],
            reinforcement_gain_assoc=params["reinforcement_gain_assoc"],
            stability_cap=params["stability_cap"],
            score_mode="retrieval_only",
        )
        storage_amount = retrieval_amount * max(storage_scale, 0.0)
        if storage_amount > 0.0:
            graph.re_activate(
                expected_id,
                storage_amount,
                source="retrieval_consolidation",
                reinforce=False,
                current_tick=current_tick,
                reinforcement_gain_direct=params["reinforcement_gain_direct"],
                reinforcement_gain_assoc=params["reinforcement_gain_assoc"],
                stability_cap=params["stability_cap"],
                score_mode="storage_only",
            )
        graph.reinforce_memory(
            expected_id,
            reinforcement_gain=float(params["reinforcement_gain_direct"]),
            stability_cap=float(params["stability_cap"]),
            current_tick=current_tick,
            count_as_retrieval=True,
        )

    def apply_retrieval_consolidation(current_tick: int) -> None:
        """Boost test memories that were successfully recalled at this tick.

        Implements the testing effect / retrieval-induced facilitation:
        successful retrieval strengthens the memory trace through reconsolidation.
        """
        retrieval_boost = params.get("retrieval_boost", 0.10)
        activation_weight = params.get("activation_weight", 0.5)
        retrieval_mode = params.get(
            "retrieval_consolidation_mode",
            "activation_and_stability",
        )
        for query_text, expected_id in test_queries:
            # Query with same parameters as evaluator
            # Two-stage BM25 re-ranking: get larger initial set, then re-rank with BM25
            if retrieval_mode == "retrieval_bm25_rerank":
                initial_k = params.get("bm25_rerank_initial_k", 20)
                final_k = params.get("bm25_rerank_final_k", 5)
                bm25_weight = params.get("bm25_rerank_weight", 0.5)
                results = graph.query_by_similarity(
                    query_text,
                    top_k=initial_k,
                    current_tick=current_tick,
                    activation_weight=activation_weight,
                    assoc_boost=params.get("assoc_boost", 0.0),
                )
                if results:
                    cand_ids = [rid for rid, _ in results]
                    bm25 = _bm25_scores(query_text, cand_ids)
                    if bm25:
                        cos_scores = {rid: score for rid, score in results}
                        max_cos = max(cos_scores.values())
                        min_cos = min(cos_scores.values())
                        max_bm = max(bm25.values())
                        # Combine: normalize both to [0,1], then weighted sum
                        combined = {}
                        for rid in cand_ids:
                            norm_cos = (cos_scores[rid] - min_cos) / max(max_cos - min_cos, 1e-8)
                            norm_bm = bm25.get(rid, 0.0) / max(max_bm, 1e-8)
                            combined[rid] = (1 - bm25_weight) * norm_cos + bm25_weight * norm_bm
                        results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:final_k]
                else:
                    results = []
            else:
                results = graph.query_by_similarity(
                    query_text,
                    top_k=params.get("retrieval_top_k", 5),
                    current_tick=current_tick,
                    activation_weight=activation_weight,
                    assoc_boost=params.get("assoc_boost", 0.0),
                )
            result_ids = [rid for rid, _ in results]
            if expected_id in result_ids:
                rank = result_ids.index(expected_id) + 1
                if retrieval_mode == "stability_only_direct":
                    graph.reinforce_memory(
                        expected_id,
                        reinforcement_gain=float(params["reinforcement_gain_direct"]),
                        stability_cap=float(params["stability_cap"]),
                        current_tick=current_tick,
                        count_as_retrieval=True,
                    )
                elif retrieval_mode == "retrieval_only":
                    graph.re_activate(
                        expected_id,
                        _scaled_boost(expected_id, retrieval_boost),
                        source="retrieval_consolidation",
                        reinforce=True,
                        current_tick=current_tick,
                        reinforcement_gain_direct=params["reinforcement_gain_direct"],
                        reinforcement_gain_assoc=params["reinforcement_gain_assoc"],
                        stability_cap=params["stability_cap"],
                        score_mode="retrieval_only",
                    )
                elif retrieval_mode == "retrieval_top1_fraction":
                    if rank != 1:
                        continue
                    retrieval_amount = _scaled_boost(expected_id, retrieval_boost)
                    _apply_fractional_hybrid_reinforcement(
                        expected_id,
                        retrieval_amount=retrieval_amount,
                        storage_scale=float(params.get("retrieval_storage_boost_scale", 0.25)),
                        current_tick=current_tick,
                    )
                elif retrieval_mode == "retrieval_rank_scaled_fraction":
                    retrieval_amount = _scaled_boost(expected_id, retrieval_boost)
                    rank_power = max(float(params.get("retrieval_rank_power", 1.0)), 0.0)
                    min_rank_scale = max(float(params.get("retrieval_rank_min_scale", 0.25)), 0.0)
                    rank_scale = max(min_rank_scale, 1.0 / (rank ** rank_power))
                    retrieval_amount *= rank_scale
                    _apply_fractional_hybrid_reinforcement(
                        expected_id,
                        retrieval_amount=retrieval_amount,
                        storage_scale=float(params.get("retrieval_storage_boost_scale", 0.25)),
                        current_tick=current_tick,
                    )
                elif retrieval_mode == "retrieval_capped_fraction":
                    cap = min(max(float(params.get("retrieval_state_cap", 0.85)), 0.0), 1.0)
                    node = graph.get_node(expected_id)
                    current_retrieval = float(
                        node.get("retrieval_score", node.get("activation_score", 0.0))
                    ) if node else 0.0
                    retrieval_amount = min(
                        _scaled_boost(expected_id, retrieval_boost),
                        max(cap - current_retrieval, 0.0),
                    )
                    _apply_fractional_hybrid_reinforcement(
                        expected_id,
                        retrieval_amount=retrieval_amount,
                        storage_scale=float(params.get("retrieval_storage_boost_scale", 0.25)),
                        current_tick=current_tick,
                    )
                elif retrieval_mode == "retrieval_with_storage_fraction":
                    retrieval_amount = _scaled_boost(expected_id, retrieval_boost)
                    _apply_fractional_hybrid_reinforcement(
                        expected_id,
                        retrieval_amount=retrieval_amount,
                        storage_scale=float(params.get("retrieval_storage_boost_scale", 0.25)),
                        current_tick=current_tick,
                    )
                elif retrieval_mode == "retrieval_margin_bm25_fraction":
                    retrieval_amount = _scaled_boost(expected_id, retrieval_boost)
                    graph.re_activate(
                        expected_id,
                        retrieval_amount,
                        source="retrieval_consolidation",
                        reinforce=False,
                        current_tick=current_tick,
                        reinforcement_gain_direct=params["reinforcement_gain_direct"],
                        reinforcement_gain_assoc=params["reinforcement_gain_assoc"],
                        stability_cap=params["stability_cap"],
                        score_mode="retrieval_only",
                    )
                    graph.reinforce_memory(
                        expected_id,
                        reinforcement_gain=float(params["reinforcement_gain_direct"]),
                        stability_cap=float(params["stability_cap"]),
                        current_tick=current_tick,
                        count_as_retrieval=True,
                    )
                    if rank != 1:
                        continue
                    top_score = results[0][1]
                    second_score = results[1][1] if len(results) > 1 else 0.0
                    margin = top_score - second_score
                    margin_threshold = float(params.get("retrieval_margin_threshold", 0.2))
                    if margin < margin_threshold:
                        continue
                    bm25_scores = _bm25_scores(query_text, result_ids)
                    target_bm25 = bm25_scores.get(expected_id, 0.0)
                    bm25_threshold = float(params.get("retrieval_bm25_min_score", 0.01))
                    if target_bm25 < bm25_threshold:
                        continue
                    if bm25_scores:
                        best_lexical = max(bm25_scores, key=bm25_scores.get)
                        if best_lexical != expected_id:
                            continue
                    storage_amount = retrieval_amount * max(
                        float(params.get("retrieval_storage_boost_scale", 0.25)),
                        0.0,
                    )
                    if storage_amount > 0.0:
                        graph.re_activate(
                            expected_id,
                            storage_amount,
                            source="retrieval_consolidation",
                            reinforce=False,
                            current_tick=current_tick,
                            reinforcement_gain_direct=params["reinforcement_gain_direct"],
                            reinforcement_gain_assoc=params["reinforcement_gain_assoc"],
                            stability_cap=params["stability_cap"],
                            score_mode="storage_only",
                        )
                else:
                    # Successful recall — boost this memory (testing effect)
                    graph.re_activate(
                        expected_id,
                        _scaled_boost(expected_id, retrieval_boost),
                        source="retrieval_consolidation",
                        reinforce=True,
                        current_tick=current_tick,
                        reinforcement_gain_direct=params["reinforcement_gain_direct"],
                        reinforcement_gain_assoc=params["reinforcement_gain_assoc"],
                        stability_cap=params["stability_cap"],
                    )

    def apply_reactivation(tick: int) -> None:
        if reactivation_policy == "none":
            return

        if reactivation_policy == "random":
            if tick % reactivation_interval != 0:
                return
            candidates = [
                nid
                for nid, attrs in graph._graph.nodes(data=True)
                if attrs.get("type") not in ("unknown", None)
                and attrs.get("created_tick", 0) <= engine.current_tick
            ]
            if not candidates:
                return
            target_id = rng.choice(candidates)
            graph.re_activate(
                target_id,
                _scaled_boost(target_id, reactivation_boost),
                source="direct",
                reinforce=True,
                current_tick=engine.current_tick,
                reinforcement_gain_direct=params["reinforcement_gain_direct"],
                reinforcement_gain_assoc=params["reinforcement_gain_assoc"],
                stability_cap=params["stability_cap"],
            )
            return

        # scheduled_query_plus_test: separate train + test schedules
        if reactivation_policy == "scheduled_query_plus_test":
            if rehearsal_targets is None:
                raise ValueError("rehearsal_targets required for this policy")
            # Train reactivation: same schedule as scheduled_query
            if tick % reactivation_interval == 0:
                train_target = _get_scheduled_target(rehearsal_targets, tick, reactivation_interval)
                if train_target:
                    graph.re_activate(
                        train_target,
                        _scaled_boost(train_target, reactivation_boost),
                        source="direct",
                        reinforce=True,
                        current_tick=engine.current_tick,
                        reinforcement_gain_direct=params["reinforcement_gain_direct"],
                        reinforcement_gain_assoc=params["reinforcement_gain_assoc"],
                        stability_cap=params["stability_cap"],
                    )
            # Test reactivation: 2x frequency (every 5 ticks instead of 10)
            test_interval = 5
            if tick % test_interval == 0:
                test_eligible = [
                    tid for tid in test_memory_ids
                    if graph._graph.nodes[tid].get("created_tick", 0) <= engine.current_tick
                ]
                if test_eligible:
                    idx = ((tick // test_interval) - 1) % len(test_eligible)
                    test_target = test_eligible[idx]
                    graph.re_activate(
                        test_target,
                        _scaled_boost(test_target, reactivation_boost),
                        source="direct",
                        reinforce=True,
                        current_tick=engine.current_tick,
                        reinforcement_gain_direct=params["reinforcement_gain_direct"],
                        reinforcement_gain_assoc=params["reinforcement_gain_assoc"],
                        stability_cap=params["stability_cap"],
                    )
            return

        # retrieval_consolidation: train reactivation (every 10) + test reactivation (every 20, starting tick 80)
        if reactivation_policy == "retrieval_consolidation":
            if rehearsal_targets is None:
                raise ValueError("rehearsal_targets required for this policy")
            # Train: every reactivation_interval ticks (10)
            if tick % reactivation_interval == 0:
                train_target = _get_scheduled_target(rehearsal_targets, tick, reactivation_interval)
                if train_target:
                    graph.re_activate(
                        train_target,
                        _scaled_boost(train_target, reactivation_boost),
                        source="direct",
                        reinforce=True,
                        current_tick=engine.current_tick,
                        reinforcement_gain_direct=params["reinforcement_gain_direct"],
                        reinforcement_gain_assoc=params["reinforcement_gain_assoc"],
                        stability_cap=params["stability_cap"],
                    )
            # Test: configurable interval, starting at configurable tick
            test_start = params.get("test_reactivation_start_tick", 80)
            test_interval_rc = params.get("test_reactivation_interval", 20)
            if tick >= test_start and tick % test_interval_rc == 0:
                test_eligible = [
                    tid for tid in test_memory_ids
                    if graph._graph.nodes[tid].get("created_tick", 0) <= engine.current_tick
                ]
                if test_eligible:
                    idx = ((tick // test_interval_rc) - 1) % len(test_eligible)
                    test_target = test_eligible[idx]
                    graph.re_activate(
                        test_target,
                        _scaled_boost(test_target, reactivation_boost),
                        source="direct",
                        reinforce=True,
                        current_tick=engine.current_tick,
                        reinforcement_gain_direct=params["reinforcement_gain_direct"],
                        reinforcement_gain_assoc=params["reinforcement_gain_assoc"],
                        stability_cap=params["stability_cap"],
                    )
            return

        # Standard scheduled_query (train only) and scheduled_query_all
        if tick % reactivation_interval != 0:
            return

        if reactivation_policy == "scheduled_query":
            if rehearsal_targets is None:
                raise ValueError("rehearsal_targets required for this policy")
            target_id = _get_scheduled_target(rehearsal_targets, tick, reactivation_interval)
        else:  # scheduled_query_all
            if rehearsal_targets is None:
                raise ValueError("rehearsal_targets required for this policy")
            all_targets = list(rehearsal_targets) + [
                tid for tid in test_memory_ids
                if graph._graph.nodes[tid].get("created_tick", 0) <= engine.current_tick
                and tid not in set(rehearsal_targets)
            ]
            target_id = _get_scheduled_target(all_targets, tick, reactivation_interval)

        if target_id:
            graph.re_activate(
                target_id,
                _scaled_boost(target_id, reactivation_boost),
                source="direct",
                reinforce=True,
                current_tick=engine.current_tick,
                reinforcement_gain_direct=params["reinforcement_gain_direct"],
                reinforcement_gain_assoc=params["reinforcement_gain_assoc"],
                stability_cap=params["stability_cap"],
            )

    # Initial evaluation (tick 0)
    if fast_eval:
        summary = collect_snapshot()
        summaries.append(summary)
        print(
            f"  Tick {summary['tick']:>4d} | recall={summary['recall_rate']:.3f} | "
            f"precision={summary['precision_rate']:.3f}"
        )
    else:
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
            if fast_eval:
                summary = collect_snapshot()
                summaries.append(summary)
                print(
                    f"  Tick {summary['tick']:>4d} | recall={summary['recall_rate']:.3f} | "
                    f"precision={summary['precision_rate']:.3f}"
                )
            else:
                summary = collect_summary()
                summaries.append(summary)
                print(
                    f"  Tick {summary['tick']:>4d} | recall={summary['recall_rate']:.3f} | "
                    f"precision={summary['precision_rate']:.3f} | retrieval={summary['retrieval_score']:.3f} | "
                    f"overall={summary['overall_score']:.3f}"
                )
            # Retrieval consolidation: boost successfully recalled test memories
            if reactivation_policy == "retrieval_consolidation":
                apply_retrieval_consolidation(t)

    return summaries


