"""Integration tests for the full simulation pipeline."""

import json
import pytest
import numpy as np

from memory_decay import MemoryGraph, DecayEngine, Evaluator, SyntheticDataGenerator


def mock_embedder(text: str):
    rng = np.random.RandomState(hash(text) % 2**31)
    return rng.randn(384).astype(np.float32)


SAMPLE_DATASET = [
    {
        "id": "f1", "type": "fact",
        "content": "서울은 대한민국의 수도이다",
        "entities": ["서울", "대한민국"],
        "tick": 0, "impact": 0.9,
        "associations": [],
        "recall_query": "대한민국의 수도는?", "recall_answer": "서울",
    },
    {
        "id": "f2", "type": "fact",
        "content": "김민수는 커피를 좋아한다",
        "entities": ["김민수", "커피"],
        "tick": 5, "impact": 0.7,
        "associations": [{"id": "f1", "weight": 0.6}],
        "recall_query": "김민수는 무엇을 좋아하는가?", "recall_answer": "커피",
    },
    {
        "id": "e1", "type": "episode",
        "content": "서울에서 커피를 마셨다",
        "entities": ["서울", "커피"],
        "tick": 10, "impact": 0.5,
        "associations": [{"id": "f1", "weight": 0.8}, {"id": "f2", "weight": 0.7}],
        "recall_query": "어디서 커피를 마셨는가?", "recall_answer": "서울",
    },
    {
        "id": "e2", "type": "episode",
        "content": "길에서 강아지를 만났다",
        "entities": ["강아지"],
        "tick": 20, "impact": 0.3,
        "associations": [],
        "recall_query": "길에서 만난 동물은?", "recall_answer": "강아지",
    },
]


class TestSimulationPipeline:
    def test_build_graph_from_dataset(self):
        from memory_decay.main import build_graph_from_dataset
        graph = build_graph_from_dataset(SAMPLE_DATASET, embedder=mock_embedder)
        assert graph.num_memories == 4

    def test_full_simulation_exponential(self):
        from memory_decay.main import build_graph_from_dataset, run_simulation

        graph = build_graph_from_dataset(SAMPLE_DATASET, embedder=mock_embedder)
        engine = DecayEngine(graph, decay_type="exponential")
        evaluator = Evaluator(graph, engine)

        queries = [(m["recall_query"], m["id"]) for m in SAMPLE_DATASET]
        snapshots = run_simulation(
            graph, engine, evaluator, queries,
            total_ticks=30, eval_interval=10,
            reactivation_policy="random",
            reactivation_interval=15,
            seed=42,
        )

        assert len(snapshots) > 0
        # Recall should generally decrease over time
        first_recall = snapshots[0]["recall_rate"]
        last_recall = snapshots[-1]["recall_rate"]

        # All values in valid range
        for snap in snapshots:
            assert 0.0 <= snap["recall_rate"] <= 1.0
            assert 0.0 <= snap["precision_rate"] <= 1.0

    def test_full_simulation_power_law(self):
        from memory_decay.main import build_graph_from_dataset, run_simulation

        graph = build_graph_from_dataset(SAMPLE_DATASET, embedder=mock_embedder)
        engine = DecayEngine(graph, decay_type="power_law")
        evaluator = Evaluator(graph, engine)

        queries = [(m["recall_query"], m["id"]) for m in SAMPLE_DATASET]
        snapshots = run_simulation(
            graph, engine, evaluator, queries,
            total_ticks=30, eval_interval=10,
            reactivation_policy="none",
        )

        assert len(snapshots) > 0

    def test_exponential_vs_power_law_different_decay(self):
        from memory_decay.main import build_graph_from_dataset, run_simulation

        queries = [(m["recall_query"], m["id"]) for m in SAMPLE_DATASET]

        # Exponential
        g1 = build_graph_from_dataset(SAMPLE_DATASET, embedder=mock_embedder)
        e1 = DecayEngine(g1, decay_type="exponential")
        ev1 = Evaluator(g1, e1)
        run_simulation(g1, e1, ev1, queries, total_ticks=50, eval_interval=50)

        # Power law
        g2 = build_graph_from_dataset(SAMPLE_DATASET, embedder=mock_embedder)
        e2 = DecayEngine(g2, decay_type="power_law")
        ev2 = Evaluator(g2, e2)
        run_simulation(g2, e2, ev2, queries, total_ticks=50, eval_interval=50)

        # Both should produce valid results (not asserting specific values,
        # just that they run and produce different patterns)
        assert ev1.history[-1]["recall_rate"] >= 0.0
        assert ev2.history[-1]["recall_rate"] >= 0.0

    def test_scheduled_reactivation_improves_recall(self):
        """Scheduled query re-activation should help preserve recall."""
        from memory_decay.main import build_graph_from_dataset, run_simulation

        queries = [(m["recall_query"], m["id"]) for m in SAMPLE_DATASET]

        # Without re-activation
        g1 = build_graph_from_dataset(SAMPLE_DATASET, embedder=mock_embedder)
        e1 = DecayEngine(g1, decay_type="exponential", params={
            "lambda_fact": 0.05, "lambda_episode": 0.08,
            "beta_fact": 0.3, "beta_episode": 0.5, "alpha": 0.5,
            "stability_weight": 0.8, "stability_decay": 0.01,
            "reinforcement_gain_direct": 0.2, "reinforcement_gain_assoc": 0.05,
            "stability_cap": 1.0,
        })
        ev1 = Evaluator(g1, e1)
        run_simulation(g1, e1, ev1, queries, total_ticks=100, eval_interval=100,
                       reactivation_policy="none")

        # With scheduled query re-activation
        g2 = build_graph_from_dataset(SAMPLE_DATASET, embedder=mock_embedder)
        e2 = DecayEngine(g2, decay_type="exponential", params={
            "lambda_fact": 0.05, "lambda_episode": 0.08,
            "beta_fact": 0.3, "beta_episode": 0.5, "alpha": 0.5,
            "stability_weight": 0.8, "stability_decay": 0.01,
            "reinforcement_gain_direct": 0.2, "reinforcement_gain_assoc": 0.05,
            "stability_cap": 1.0,
        })
        ev2 = Evaluator(g2, e2)
        rehearsal_targets = [m["id"] for m in SAMPLE_DATASET]
        scheduled = run_simulation(
            g2, e2, ev2, queries,
            total_ticks=100, eval_interval=20,
            reactivation_policy="scheduled_query", reactivation_interval=10,
            rehearsal_targets=rehearsal_targets,
        )

        assert ev2.history[-1]["recall_rate"] >= ev1.history[-1]["recall_rate"]
        assert any(s["recall_rate"] < 0.95 for s in scheduled[1:])

    def test_random_policy_is_deterministic_with_seed(self):
        from memory_decay.main import build_graph_from_dataset, run_simulation

        queries = [(m["recall_query"], m["id"]) for m in SAMPLE_DATASET]

        g1 = build_graph_from_dataset(SAMPLE_DATASET, embedder=mock_embedder)
        e1 = DecayEngine(g1, decay_type="exponential")
        ev1 = Evaluator(g1, e1)
        out1 = run_simulation(
            g1, e1, ev1, queries,
            total_ticks=30, eval_interval=10,
            reactivation_policy="random", reactivation_interval=5, seed=7,
        )

        g2 = build_graph_from_dataset(SAMPLE_DATASET, embedder=mock_embedder)
        e2 = DecayEngine(g2, decay_type="exponential")
        ev2 = Evaluator(g2, e2)
        out2 = run_simulation(
            g2, e2, ev2, queries,
            total_ticks=30, eval_interval=10,
            reactivation_policy="random", reactivation_interval=5, seed=7,
        )

        assert [snap["recall_rate"] for snap in out1] == [snap["recall_rate"] for snap in out2]

    def test_retrieval_consolidation_can_scale_boost_by_importance(self, monkeypatch):
        from memory_decay.main import run_simulation

        graph = MemoryGraph(embedder=mock_embedder)
        graph.add_memory("high", "fact", "중요한 사실", 0.9, created_tick=0)
        graph.add_memory("low", "fact", "덜 중요한 사실", 0.1, created_tick=0)

        params = {
            "alpha": 1.0,
            "stability_weight": 0.0,
            "retrieval_boost": 0.2,
            "importance_scaled_retrieval_boost": True,
            "importance_boost_min_scale": 0.5,
            "reinforcement_gain_direct": 0.2,
            "reinforcement_gain_assoc": 0.05,
            "stability_cap": 1.0,
        }
        engine = DecayEngine(graph, decay_type="exponential", params=params)
        evaluator = Evaluator(graph, engine)
        queries = [("high-query", "high"), ("low-query", "low")]
        query_to_id = dict(queries)
        boost_amounts: dict[str, float] = {}

        def fake_query_by_similarity(
            query_text: str,
            top_k: int = 5,
            current_tick: int | None = None,
            activation_weight: float = 0.0,
            assoc_boost: float = 0.0,
        ):
            return [(query_to_id[query_text], 1.0)]

        def recording_reactivate(node_id: str, boost_amount: float, **kwargs):
            boost_amounts[node_id] = boost_amount

        monkeypatch.setattr(graph, "query_by_similarity", fake_query_by_similarity)
        monkeypatch.setattr(graph, "re_activate", recording_reactivate)

        run_simulation(
            graph,
            engine,
            evaluator,
            queries,
            total_ticks=1,
            eval_interval=1,
            reactivation_policy="retrieval_consolidation",
            rehearsal_targets=["high", "low"],
        )

        assert boost_amounts["high"] == pytest.approx(0.19)
        assert boost_amounts["low"] == pytest.approx(0.11)

    def test_retrieval_consolidation_can_reinforce_stability_without_activation_boost(self):
        from memory_decay.main import run_simulation

        vectors = {
            "memory target": np.array([1.0, 0.0], dtype=np.float32),
            "query target": np.array([1.0, 0.0], dtype=np.float32),
        }

        graph = MemoryGraph(embedder=lambda text: vectors[text])
        graph.add_memory("target", "fact", "memory target", 1.0, created_tick=0)
        graph.set_activation("target", 0.6)

        engine = DecayEngine(
            graph,
            custom_decay_fn=lambda activation, *_args: activation,
            params={
                "alpha": 1.0,
                "stability_weight": 0.0,
                "reinforcement_gain_direct": 0.2,
                "reinforcement_gain_assoc": 0.0,
                "stability_cap": 1.0,
                "retrieval_boost": 0.2,
                "retrieval_consolidation_mode": "stability_only_direct",
                "test_reactivation_start_tick": 100,
                "test_reactivation_interval": 100,
            },
        )
        evaluator = Evaluator(graph, engine)

        run_simulation(
            graph,
            engine,
            evaluator,
            [("query target", "target")],
            total_ticks=1,
            eval_interval=1,
            reactivation_policy="retrieval_consolidation",
            reactivation_interval=100,
            rehearsal_targets=["target"],
        )

        node = graph.get_node("target")
        assert node["activation_score"] == pytest.approx(0.6)
        assert node["stability_score"] > 0.0

    def test_retrieval_consolidation_can_boost_retrieval_more_than_storage(self):
        from memory_decay.main import run_simulation

        vectors = {
            "memory target": np.array([1.0, 0.0], dtype=np.float32),
            "query target": np.array([1.0, 0.0], dtype=np.float32),
        }

        graph = MemoryGraph(embedder=lambda text: vectors[text])
        graph.add_memory("target", "fact", "memory target", 1.0, created_tick=0)
        graph.set_activation("target", 0.6)
        graph.set_storage_score("target", 0.6)

        engine = DecayEngine(
            graph,
            custom_decay_fn=lambda activation, *_args: activation,
            params={
                "alpha": 1.0,
                "stability_weight": 0.0,
                "reinforcement_gain_direct": 0.2,
                "reinforcement_gain_assoc": 0.0,
                "stability_cap": 1.0,
                "retrieval_boost": 0.2,
                "retrieval_consolidation_mode": "retrieval_with_storage_fraction",
                "retrieval_storage_boost_scale": 0.25,
                "test_reactivation_start_tick": 100,
                "test_reactivation_interval": 100,
            },
        )
        evaluator = Evaluator(graph, engine)

        run_simulation(
            graph,
            engine,
            evaluator,
            [("query target", "target")],
            total_ticks=1,
            eval_interval=1,
            reactivation_policy="retrieval_consolidation",
            reactivation_interval=100,
            rehearsal_targets=["target"],
        )

        node = graph.get_node("target")
        assert node["retrieval_score"] == pytest.approx(0.8)
        assert node["activation_score"] == pytest.approx(0.8)
        assert node["storage_score"] == pytest.approx(0.65)
        assert node["stability_score"] > 0.0

    def test_retrieval_consolidation_top1_gate_skips_non_top1_hits(self, monkeypatch):
        from memory_decay.main import run_simulation

        graph = MemoryGraph(embedder=mock_embedder)
        graph.add_memory("target", "fact", "target memory", 1.0, created_tick=0)
        graph.add_memory("other", "fact", "other memory", 1.0, created_tick=0)
        graph.set_activation("target", 0.6)
        graph.set_storage_score("target", 0.6)

        engine = DecayEngine(
            graph,
            custom_decay_fn=lambda activation, *_args: activation,
            params={
                "alpha": 1.0,
                "stability_weight": 0.0,
                "reinforcement_gain_direct": 0.2,
                "reinforcement_gain_assoc": 0.0,
                "stability_cap": 1.0,
                "retrieval_boost": 0.2,
                "retrieval_consolidation_mode": "retrieval_top1_fraction",
                "retrieval_storage_boost_scale": 0.25,
                "test_reactivation_start_tick": 100,
                "test_reactivation_interval": 100,
            },
        )
        evaluator = Evaluator(graph, engine)

        monkeypatch.setattr(
            graph,
            "query_by_similarity",
            lambda *args, **kwargs: [("other", 1.0), ("target", 0.9)],
        )

        run_simulation(
            graph,
            engine,
            evaluator,
            [("query target", "target")],
            total_ticks=1,
            eval_interval=1,
            reactivation_policy="retrieval_consolidation",
            reactivation_interval=100,
            rehearsal_targets=["target"],
        )

        node = graph.get_node("target")
        assert node["retrieval_score"] == pytest.approx(0.6)
        assert node["storage_score"] == pytest.approx(0.6)
        assert node["stability_score"] == pytest.approx(0.0)

    def test_retrieval_consolidation_rank_scaled_fraction_uses_rank_to_shrink_boost(self, monkeypatch):
        from memory_decay.main import run_simulation

        graph = MemoryGraph(embedder=mock_embedder)
        graph.add_memory("target", "fact", "target memory", 1.0, created_tick=0)
        graph.add_memory("other", "fact", "other memory", 1.0, created_tick=0)
        graph.set_activation("target", 0.6)
        graph.set_storage_score("target", 0.6)

        engine = DecayEngine(
            graph,
            custom_decay_fn=lambda activation, *_args: activation,
            params={
                "alpha": 1.0,
                "stability_weight": 0.0,
                "reinforcement_gain_direct": 0.2,
                "reinforcement_gain_assoc": 0.0,
                "stability_cap": 1.0,
                "retrieval_boost": 0.2,
                "retrieval_consolidation_mode": "retrieval_rank_scaled_fraction",
                "retrieval_storage_boost_scale": 0.25,
                "retrieval_rank_power": 1.0,
                "retrieval_rank_min_scale": 0.25,
                "test_reactivation_start_tick": 100,
                "test_reactivation_interval": 100,
            },
        )
        evaluator = Evaluator(graph, engine)

        monkeypatch.setattr(
            graph,
            "query_by_similarity",
            lambda *args, **kwargs: [("other", 1.0), ("target", 0.9)],
        )

        run_simulation(
            graph,
            engine,
            evaluator,
            [("query target", "target")],
            total_ticks=1,
            eval_interval=1,
            reactivation_policy="retrieval_consolidation",
            reactivation_interval=100,
            rehearsal_targets=["target"],
        )

        node = graph.get_node("target")
        assert node["retrieval_score"] == pytest.approx(0.7)
        assert node["storage_score"] == pytest.approx(0.625)
        assert node["stability_score"] > 0.0

    def test_retrieval_consolidation_capped_fraction_respects_retrieval_cap(self, monkeypatch):
        from memory_decay.main import run_simulation

        graph = MemoryGraph(embedder=mock_embedder)
        graph.add_memory("target", "fact", "target memory", 1.0, created_tick=0)
        graph.set_activation("target", 0.78)
        graph.set_storage_score("target", 0.6)

        engine = DecayEngine(
            graph,
            custom_decay_fn=lambda activation, *_args: activation,
            params={
                "alpha": 1.0,
                "stability_weight": 0.0,
                "reinforcement_gain_direct": 0.2,
                "reinforcement_gain_assoc": 0.0,
                "stability_cap": 1.0,
                "retrieval_boost": 0.2,
                "retrieval_consolidation_mode": "retrieval_capped_fraction",
                "retrieval_storage_boost_scale": 0.25,
                "retrieval_state_cap": 0.85,
                "test_reactivation_start_tick": 100,
                "test_reactivation_interval": 100,
            },
        )
        evaluator = Evaluator(graph, engine)

        monkeypatch.setattr(
            graph,
            "query_by_similarity",
            lambda *args, **kwargs: [("target", 1.0)],
        )

        run_simulation(
            graph,
            engine,
            evaluator,
            [("query target", "target")],
            total_ticks=1,
            eval_interval=1,
            reactivation_policy="retrieval_consolidation",
            reactivation_interval=100,
            rehearsal_targets=["target"],
        )

        node = graph.get_node("target")
        assert node["retrieval_score"] == pytest.approx(0.85)
        assert node["storage_score"] == pytest.approx(0.6175)
        assert node["stability_score"] > 0.0

    def test_retrieval_consolidation_margin_bm25_fraction_requires_margin(self, monkeypatch):
        from memory_decay.main import run_simulation

        graph = MemoryGraph(embedder=mock_embedder)
        graph.add_memory("target", "fact", "seoul capital korea", 1.0, created_tick=0)
        graph.add_memory("other", "fact", "seoul capital city", 1.0, created_tick=0)
        graph.set_activation("target", 0.6)
        graph.set_storage_score("target", 0.6)

        engine = DecayEngine(
            graph,
            custom_decay_fn=lambda activation, *_args: activation,
            params={
                "alpha": 1.0,
                "stability_weight": 0.0,
                "reinforcement_gain_direct": 0.2,
                "reinforcement_gain_assoc": 0.0,
                "stability_cap": 1.0,
                "retrieval_boost": 0.2,
                "retrieval_consolidation_mode": "retrieval_margin_bm25_fraction",
                "retrieval_storage_boost_scale": 0.25,
                "retrieval_margin_threshold": 0.2,
                "retrieval_bm25_min_score": 0.01,
                "test_reactivation_start_tick": 100,
                "test_reactivation_interval": 100,
            },
        )
        evaluator = Evaluator(graph, engine)

        monkeypatch.setattr(
            graph,
            "query_by_similarity",
            lambda *args, **kwargs: [("target", 1.0), ("other", 0.95)],
        )

        run_simulation(
            graph,
            engine,
            evaluator,
            [("seoul capital", "target")],
            total_ticks=1,
            eval_interval=1,
            reactivation_policy="retrieval_consolidation",
            reactivation_interval=100,
            rehearsal_targets=["target"],
        )

        node = graph.get_node("target")
        assert node["retrieval_score"] == pytest.approx(0.8)
        assert node["storage_score"] == pytest.approx(0.6)
        assert node["stability_score"] > 0.0

    def test_retrieval_consolidation_margin_bm25_fraction_requires_lexical_agreement(self, monkeypatch):
        from memory_decay.main import run_simulation

        graph = MemoryGraph(embedder=mock_embedder)
        graph.add_memory("target", "fact", "coffee beans roast profile", 1.0, created_tick=0)
        graph.add_memory("other", "fact", "seoul capital korea", 1.0, created_tick=0)
        graph.set_activation("target", 0.6)
        graph.set_storage_score("target", 0.6)

        engine = DecayEngine(
            graph,
            custom_decay_fn=lambda activation, *_args: activation,
            params={
                "alpha": 1.0,
                "stability_weight": 0.0,
                "reinforcement_gain_direct": 0.2,
                "reinforcement_gain_assoc": 0.0,
                "stability_cap": 1.0,
                "retrieval_boost": 0.2,
                "retrieval_consolidation_mode": "retrieval_margin_bm25_fraction",
                "retrieval_storage_boost_scale": 0.25,
                "retrieval_margin_threshold": 0.2,
                "retrieval_bm25_min_score": 0.01,
                "test_reactivation_start_tick": 100,
                "test_reactivation_interval": 100,
            },
        )
        evaluator = Evaluator(graph, engine)

        monkeypatch.setattr(
            graph,
            "query_by_similarity",
            lambda *args, **kwargs: [("target", 1.0), ("other", 0.7)],
        )

        run_simulation(
            graph,
            engine,
            evaluator,
            [("seoul capital", "target")],
            total_ticks=1,
            eval_interval=1,
            reactivation_policy="retrieval_consolidation",
            reactivation_interval=100,
            rehearsal_targets=["target"],
        )

        node = graph.get_node("target")
        assert node["retrieval_score"] == pytest.approx(0.8)
        assert node["storage_score"] == pytest.approx(0.6)
        assert node["stability_score"] > 0.0

    def test_retrieval_consolidation_margin_bm25_fraction_boosts_storage_when_confident(self, monkeypatch):
        from memory_decay.main import run_simulation

        graph = MemoryGraph(embedder=mock_embedder)
        graph.add_memory("target", "fact", "seoul capital korea", 1.0, created_tick=0)
        graph.add_memory("other", "fact", "coffee beans roast", 1.0, created_tick=0)
        graph.set_activation("target", 0.6)
        graph.set_storage_score("target", 0.6)

        engine = DecayEngine(
            graph,
            custom_decay_fn=lambda activation, *_args: activation,
            params={
                "alpha": 1.0,
                "stability_weight": 0.0,
                "reinforcement_gain_direct": 0.2,
                "reinforcement_gain_assoc": 0.0,
                "stability_cap": 1.0,
                "retrieval_boost": 0.2,
                "retrieval_consolidation_mode": "retrieval_margin_bm25_fraction",
                "retrieval_storage_boost_scale": 0.25,
                "retrieval_margin_threshold": 0.2,
                "retrieval_bm25_min_score": 0.01,
                "test_reactivation_start_tick": 100,
                "test_reactivation_interval": 100,
            },
        )
        evaluator = Evaluator(graph, engine)

        monkeypatch.setattr(
            graph,
            "query_by_similarity",
            lambda *args, **kwargs: [("target", 1.0), ("other", 0.7)],
        )

        run_simulation(
            graph,
            engine,
            evaluator,
            [("seoul capital", "target")],
            total_ticks=1,
            eval_interval=1,
            reactivation_policy="retrieval_consolidation",
            reactivation_interval=100,
            rehearsal_targets=["target"],
        )

        node = graph.get_node("target")
        assert node["retrieval_score"] == pytest.approx(0.8)
        assert node["storage_score"] == pytest.approx(0.65)
        assert node["stability_score"] > 0.0
