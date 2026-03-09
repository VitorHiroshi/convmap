"""Tests for topology, TDA, and DTW lenses."""

import numpy as np
import pytest

from convmap import Engine, Chunk, EmbeddedConversation, MicroCluster, MapState
from convmap.lenses import topology, tda, dtw


def _make_conversation(vectors, conv_id, metadata=None):
    chunks = [
        Chunk(text=f"chunk-{i}", index=i, embedding=v)
        for i, v in enumerate(vectors)
    ]
    return EmbeddedConversation(id=conv_id, chunks=chunks, metadata=metadata or {})


def _make_state_with_clusters(centers, weights=None, dims=64):
    """Build a MapState with explicit core clusters."""
    import time
    now = time.time()
    weights = weights or [10.0] * len(centers)
    clusters = []
    for i, (c, w) in enumerate(zip(centers, weights)):
        c_norm = c / (np.linalg.norm(c) + 1e-8)
        clusters.append(MicroCluster(
            centroid=c_norm.astype(np.float32),
            weight=w,
            radius=0.3,
            created_at=now,
            updated_at=now,
            count=int(w * 10),
        ))
    return MapState(
        core_clusters=clusters,
        potential_clusters=[],
        outlier_buffer=[],
        recent_vectors=[],
        snapshots=[],
        dimensions=dims,
        timestamp=now,
    )


# --- Topology ---


class TestTopology:
    def test_adjacency_finds_similar_clusters(self):
        np.random.seed(42)
        # Two similar clusters and one distant
        c1 = np.random.randn(64).astype(np.float32)
        c2 = c1 + np.random.randn(64) * 0.1  # close to c1
        c3 = np.random.randn(64).astype(np.float32)  # random, likely far

        state = _make_state_with_clusters([c1, c2, c3])
        edges = topology.adjacency(state, threshold=0.5)

        # c1 and c2 should be adjacent
        has_c1_c2 = any(
            (e["i"] == 0 and e["j"] == 1) or (e["i"] == 1 and e["j"] == 0)
            for e in edges
        )
        assert has_c1_c2

    def test_adjacency_empty_with_one_cluster(self):
        c = np.random.randn(64).astype(np.float32)
        state = _make_state_with_clusters([c])
        assert topology.adjacency(state) == []

    def test_isolated_finds_distant_clusters(self):
        np.random.seed(42)
        # Create orthogonal clusters — all isolated from each other
        centers = []
        for i in range(5):
            c = np.zeros(64, dtype=np.float32)
            c[i * 10] = 1.0
            centers.append(c)

        state = _make_state_with_clusters(centers)
        result = topology.isolated(state, threshold=0.3)

        # All should be isolated since they're orthogonal
        assert len(result) == 5

    def test_bridges_finds_multi_cluster_vectors(self):
        np.random.seed(42)
        c1 = np.zeros(64, dtype=np.float32)
        c1[0] = 1.0
        c2 = np.zeros(64, dtype=np.float32)
        c2[1] = 1.0

        state = _make_state_with_clusters([c1, c2])

        # Add a bridge vector that's similar to both
        bridge = (c1 + c2) / np.sqrt(2)
        state.recent_vectors = [(bridge.astype(np.float32), {"id": "bridge"})]

        result = topology.bridges(state, cluster_threshold=0.6)
        assert len(result) == 1
        assert result[0]["n_clusters"] == 2

    def test_density_map_returns_per_cluster(self):
        np.random.seed(42)
        centers = [np.random.randn(64).astype(np.float32) for _ in range(3)]
        state = _make_state_with_clusters(centers, weights=[20.0, 5.0, 1.0])

        result = topology.density_map(state)
        assert len(result["clusters"]) == 3
        # Sorted by density descending
        densities = [c["density"] for c in result["clusters"]]
        assert densities == sorted(densities, reverse=True)


# --- TDA ---


class TestTDA:
    def test_persistence_with_few_points(self):
        state = _make_state_with_clusters([np.random.randn(64) for _ in range(2)])
        result = tda.persistence(state, source="core")
        # Too few points, should return empty dimensions
        assert result["n_points"] == 2

    def test_persistence_fallback_with_enough_points(self):
        np.random.seed(42)
        centers = [np.random.randn(64).astype(np.float32) for _ in range(10)]
        state = _make_state_with_clusters(centers)

        result = tda.persistence(state, source="core")
        assert result["n_points"] == 10
        assert 0 in result["dimensions"]

    def test_features_returns_sorted_by_lifetime(self):
        np.random.seed(42)
        centers = [np.random.randn(64).astype(np.float32) for _ in range(10)]
        state = _make_state_with_clusters(centers)

        result = tda.features(state, source="core")
        if len(result) > 1:
            lifetimes = [f["lifetime"] for f in result]
            assert lifetimes == sorted(lifetimes, reverse=True)

    def test_landscape_distance_same_state_is_zero(self):
        np.random.seed(42)
        centers = [np.random.randn(64).astype(np.float32) for _ in range(5)]
        state = _make_state_with_clusters(centers)

        dist = tda.landscape_distance(state, state, source="core")
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_landscape_distance_different_states(self):
        np.random.seed(42)
        centers_a = [np.random.randn(64).astype(np.float32) for _ in range(5)]
        centers_b = [np.random.randn(64).astype(np.float32) for _ in range(5)]
        state_a = _make_state_with_clusters(centers_a)
        state_b = _make_state_with_clusters(centers_b)

        dist = tda.landscape_distance(state_a, state_b, source="core")
        assert dist > 0


# --- DTW ---


class TestDTW:
    def test_identical_sequences_have_zero_distance(self):
        np.random.seed(42)
        vecs = [np.random.randn(64).astype(np.float32) for _ in range(5)]
        for v in vecs:
            v /= np.linalg.norm(v)

        conv = _make_conversation(vecs, "a")
        assert dtw.distance(conv, conv) == pytest.approx(0.0, abs=1e-6)

    def test_different_sequences_have_positive_distance(self):
        np.random.seed(42)
        vecs_a = [np.random.randn(64).astype(np.float32) for _ in range(5)]
        vecs_b = [np.random.randn(64).astype(np.float32) for _ in range(5)]

        conv_a = _make_conversation(vecs_a, "a")
        conv_b = _make_conversation(vecs_b, "b")

        assert dtw.distance(conv_a, conv_b) > 0

    def test_alignment_returns_path(self):
        np.random.seed(42)
        vecs_a = [np.random.randn(64).astype(np.float32) for _ in range(4)]
        vecs_b = [np.random.randn(64).astype(np.float32) for _ in range(6)]

        conv_a = _make_conversation(vecs_a, "a")
        conv_b = _make_conversation(vecs_b, "b")

        result = dtw.alignment(conv_a, conv_b)
        assert result["len_a"] == 4
        assert result["len_b"] == 6
        assert len(result["path"]) >= max(4, 6)
        # Path should start at (0,0) and end at (3,5)
        assert result["path"][0]["i"] == 0
        assert result["path"][0]["j"] == 0
        assert result["path"][-1]["i"] == 3
        assert result["path"][-1]["j"] == 5

    def test_pairwise_is_symmetric(self):
        np.random.seed(42)
        convs = [
            _make_conversation(
                [np.random.randn(64).astype(np.float32) for _ in range(3)],
                f"conv-{i}",
            )
            for i in range(4)
        ]

        matrix = dtw.pairwise(convs)
        assert matrix.shape == (4, 4)
        np.testing.assert_array_almost_equal(matrix, matrix.T)
        np.testing.assert_array_almost_equal(np.diag(matrix), 0)

    def test_most_similar_finds_closest(self):
        np.random.seed(42)
        base_vecs = [np.random.randn(64).astype(np.float32) for _ in range(3)]
        for v in base_vecs:
            v /= np.linalg.norm(v)

        target = _make_conversation(base_vecs, "target")

        # Near duplicate
        near_vecs = [v + np.random.randn(64).astype(np.float32) * 0.01 for v in base_vecs]
        near = _make_conversation(near_vecs, "near")

        # Random
        far = _make_conversation(
            [np.random.randn(64).astype(np.float32) for _ in range(3)],
            "far",
        )

        result = dtw.most_similar(target, [far, near], k=2)
        assert result[0]["id"] == "near"

    def test_empty_conversation_returns_inf(self):
        empty = EmbeddedConversation(id="empty", chunks=[])
        nonempty = _make_conversation(
            [np.random.randn(64).astype(np.float32)], "nonempty"
        )
        assert dtw.distance(empty, nonempty) == float("inf")
