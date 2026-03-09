"""Tests for the query module."""

import time

import numpy as np
import pytest

from convmap import Engine, Chunk, EmbeddedConversation, MicroCluster, MapState
from convmap import query


def _make_conversation(vector, conv_id, timestamp=None):
    meta = {"id": conv_id}
    if timestamp is not None:
        meta["timestamp"] = timestamp
    return EmbeddedConversation(
        id=conv_id,
        chunks=[Chunk(text="", index=0, embedding=vector)],
        metadata=meta,
    )


def _cluster_vectors(center, n, noise=0.05):
    vecs = []
    for _ in range(n):
        v = center + np.random.randn(len(center)) * noise
        v = (v / np.linalg.norm(v)).astype(np.float32)
        vecs.append(v)
    return vecs


def _build_engine(n=50, dims=64, seed=42):
    np.random.seed(seed)
    engine = Engine(dimensions=dims, epsilon=0.3, mu=3.0, maintenance_interval=20)

    # Create 3 real clusters so core clusters form
    centers = [np.random.randn(dims).astype(np.float32) for _ in range(3)]
    per_cluster = n // 3

    base_time = 1000.0
    i = 0
    for center in centers:
        vecs = _cluster_vectors(center, per_cluster, noise=0.05)
        for v in vecs:
            conv = _make_conversation(v, f"conv-{i}", timestamp=base_time + i)
            engine.ingest(conv)
            i += 1

    engine._maintain()
    return engine


class TestConcept:
    def test_finds_similar_vectors(self):
        np.random.seed(42)
        engine = _build_engine()

        # Use the first recent vector as the query
        query_vec = engine.recent_vectors[0][0]
        state = engine.state

        results = query.concept(state, query_vec, k=5)
        assert len(results) == 5
        assert results[0]["similarity"] >= results[1]["similarity"]
        # First result should be the vector itself
        assert results[0]["similarity"] > 0.99

    def test_time_filtered_concept(self):
        np.random.seed(42)
        engine = _build_engine()
        state = engine.state

        query_vec = np.random.randn(64).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)

        # Only vectors in first half of time range
        results = query.concept(state, query_vec, k=50, time_range=(1000.0, 1025.0))
        for r in results:
            ts = float(r["metadata"]["timestamp"])
            assert 1000.0 <= ts <= 1025.0


class TestTimeWindow:
    def test_filters_recent_vectors(self):
        engine = _build_engine()
        state = engine.state

        windowed = query.time_window(state, 1010.0, 1020.0)
        for _, meta in windowed.recent_vectors:
            ts = float(meta["timestamp"])
            assert 1010.0 <= ts <= 1020.0

    def test_preserves_clusters(self):
        engine = _build_engine()
        state = engine.state
        windowed = query.time_window(state, 1000.0, 1050.0)

        assert len(windowed.core_clusters) == len(state.core_clusters)


class TestFunnel:
    def test_returns_stages_by_size(self):
        engine = _build_engine()
        state = engine.state

        stages = query.funnel(state)
        assert len(stages) > 0
        counts = [s["count"] for s in stages]
        assert counts == sorted(counts, reverse=True)

    def test_shares_sum_to_one(self):
        engine = _build_engine()
        state = engine.state

        stages = query.funnel(state)
        total_share = sum(s["share"] for s in stages)
        assert total_share == pytest.approx(1.0, abs=1e-6)


class TestHistogram:
    def test_total_matches_vector_count(self):
        engine = _build_engine()
        state = engine.state

        hist = query.histogram(state)
        assigned = sum(c["count"] for c in hist["clusters"])
        assert assigned + hist["unassigned"] == hist["total"]

    def test_shares_sum_correctly(self):
        engine = _build_engine()
        state = engine.state

        hist = query.histogram(state)
        if hist["total"] > 0:
            total_share = sum(c["share"] for c in hist["clusters"])
            unassigned_share = hist["unassigned"] / hist["total"]
            assert total_share + unassigned_share == pytest.approx(1.0, abs=1e-6)


class TestCompareWindows:
    def test_compare_returns_shifts(self):
        engine = _build_engine()
        state = engine.state

        result = query.compare_windows(
            state,
            window_a=(1000.0, 1025.0),
            window_b=(1025.0, 1050.0),
        )
        assert result["window_a_count"] > 0
        assert result["window_b_count"] > 0
        assert len(result["cluster_shifts"]) > 0

    def test_deltas_sum_to_zero(self):
        engine = _build_engine()
        state = engine.state

        result = query.compare_windows(
            state,
            window_a=(1000.0, 1025.0),
            window_b=(1025.0, 1050.0),
        )
        total_delta = sum(s["delta"] for s in result["cluster_shifts"])
        assert total_delta == pytest.approx(0.0, abs=1e-6)


class TestSegment:
    def test_returns_members_of_cluster(self):
        engine = _build_engine()
        state = engine.state

        if state.core_clusters:
            members = query.segment(state, 0)
            assert len(members) > 0
            # All should be most similar to cluster 0
            for m in members:
                sims = [mc.similarity(m["vector"]) for mc in state.core_clusters]
                assert int(np.argmax(sims)) == 0

    def test_invalid_index_returns_empty(self):
        engine = _build_engine()
        state = engine.state
        assert query.segment(state, 999) == []


class TestAnomalies:
    def test_anomalies_below_threshold(self):
        engine = _build_engine()
        state = engine.state

        results = query.anomalies(state, threshold=0.99)
        for r in results:
            assert r["best_similarity"] < 0.99


class TestEmbedderMetadata:
    def test_metadata_folded_into_text(self):
        from convmap.embedder import Embedder

        # Test the static method directly
        meta = {
            "category": "billing dispute",
            "summary": "Customer called about overdue payment",
            "duration": "120",
            "id": "conv-123",
            "timestamp": "1700000000",
            "status": "ok",  # too short, skipped
        }
        result = Embedder._flatten_metadata(meta)
        assert "category: billing dispute" in result
        assert "summary: Customer called about overdue payment" in result
        assert "duration" not in result
        assert "id" not in result
        assert "timestamp" not in result
        assert "status" not in result

    def test_empty_metadata_returns_empty(self):
        from convmap.embedder import Embedder
        assert Embedder._flatten_metadata({}) == ""
        assert Embedder._flatten_metadata(None) == ""
