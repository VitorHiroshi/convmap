"""Engine tests using synthetic vectors — no model download needed."""

import numpy as np
import pytest

from convmap import Engine, Chunk, EmbeddedConversation
from convmap.lenses import density


def _make_cluster_vectors(center: np.ndarray, n: int, noise: float = 0.05) -> list[np.ndarray]:
    """Generate n vectors clustered around a center with small noise."""
    vectors = []
    for _ in range(n):
        v = center + np.random.randn(len(center)) * noise
        v = v / np.linalg.norm(v)
        vectors.append(v.astype(np.float32))
    return vectors


def _make_conversation(vector: np.ndarray, conv_id: str) -> EmbeddedConversation:
    """Wrap a vector as a single-chunk embedded conversation."""
    return EmbeddedConversation(
        id=conv_id,
        chunks=[Chunk(text="", index=0, embedding=vector)],
    )


class TestEngineIngestion:
    def test_single_vector_goes_to_outlier(self):
        engine = Engine(dimensions=64)
        vec = np.random.randn(64).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        engine.ingest(_make_conversation(vec, "conv-1"))

        # First vector with no clusters goes to outlier buffer
        assert len(engine.outlier_buffer) == 1
        assert len(engine.core_clusters) == 0

    def test_similar_vectors_merge(self):
        engine = Engine(dimensions=64, epsilon=0.3, mu=3.0, maintenance_interval=50)
        center = np.random.randn(64).astype(np.float32)
        center = center / np.linalg.norm(center)

        vectors = _make_cluster_vectors(center, 10, noise=0.02)
        for i, v in enumerate(vectors):
            engine.ingest(_make_conversation(v, f"conv-{i}"))

        # After maintenance, similar vectors should form clusters
        engine._maintain()

        total = len(engine.core_clusters) + len(engine.potential_clusters)
        assert total >= 1
        assert total < 10  # They shouldn't each be their own cluster

    def test_distinct_clusters_stay_separate(self):
        np.random.seed(42)
        engine = Engine(dimensions=64, epsilon=0.3, mu=3.0, maintenance_interval=100)

        # Two well-separated cluster centers
        center_a = np.zeros(64, dtype=np.float32)
        center_a[0] = 1.0
        center_b = np.zeros(64, dtype=np.float32)
        center_b[32] = 1.0

        vecs_a = _make_cluster_vectors(center_a, 20, noise=0.02)
        vecs_b = _make_cluster_vectors(center_b, 20, noise=0.02)

        for i, v in enumerate(vecs_a + vecs_b):
            engine.ingest(_make_conversation(v, f"conv-{i}"))

        engine._maintain()

        all_clusters = engine.core_clusters + engine.potential_clusters
        assert len(all_clusters) >= 2


class TestDensityLens:
    def test_clusters_sorted_by_weight(self):
        np.random.seed(42)
        engine = Engine(dimensions=64, epsilon=0.3, mu=3.0, maintenance_interval=50)

        center = np.random.randn(64).astype(np.float32)
        center = center / np.linalg.norm(center)
        vectors = _make_cluster_vectors(center, 20, noise=0.02)

        for i, v in enumerate(vectors):
            engine.ingest(_make_conversation(v, f"conv-{i}"))

        engine._maintain()

        state = engine.state
        result = density.clusters(state)

        if len(result) > 1:
            weights = [r["weight"] for r in result]
            assert weights == sorted(weights, reverse=True)

    def test_distribution_summary(self):
        engine = Engine(dimensions=64)
        state = engine.state

        dist = density.distribution(state)
        assert dist["core_count"] == 0
        assert dist["potential_count"] == 0
        assert dist["outlier_count"] == 0

    def test_nearest_returns_k_or_fewer(self):
        np.random.seed(42)
        engine = Engine(dimensions=64, epsilon=0.3, mu=3.0, maintenance_interval=10)

        center = np.random.randn(64).astype(np.float32)
        center = center / np.linalg.norm(center)
        vectors = _make_cluster_vectors(center, 15, noise=0.02)

        for i, v in enumerate(vectors):
            engine.ingest(_make_conversation(v, f"conv-{i}"))

        engine._maintain()

        state = engine.state
        query = np.random.randn(64).astype(np.float32)
        query = query / np.linalg.norm(query)

        result = density.nearest(state, query, k=3)
        assert len(result) <= 3


class TestRecentBuffer:
    def test_bounded_size(self):
        engine = Engine(dimensions=64, max_recent=5, maintenance_interval=100)

        for i in range(10):
            vec = np.random.randn(64).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            engine.ingest(_make_conversation(vec, f"conv-{i}"))

        assert len(engine.recent_vectors) == 5
