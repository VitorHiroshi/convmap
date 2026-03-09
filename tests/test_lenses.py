"""Tests for drift, HDC, and neighborhood lenses."""

import numpy as np
import pytest

from convmap import Engine, Chunk, EmbeddedConversation, Snapshot
from convmap.lenses import density, drift, hdc, neighborhood


def _make_cluster_vectors(center, n, noise=0.05):
    vectors = []
    for _ in range(n):
        v = center + np.random.randn(len(center)) * noise
        v = v / np.linalg.norm(v)
        vectors.append(v.astype(np.float32))
    return vectors


def _make_conversation(vector, conv_id):
    return EmbeddedConversation(
        id=conv_id,
        chunks=[Chunk(text="", index=0, embedding=vector)],
    )


# --- Drift ---


class TestDrift:
    def test_compare_identical_snapshots(self):
        np.random.seed(42)
        centroids = [np.random.randn(64).astype(np.float32) for _ in range(3)]
        for c in centroids:
            c /= np.linalg.norm(c)

        snap = Snapshot(
            centroids=centroids,
            weights=[10.0, 5.0, 3.0],
            counts=[100, 50, 30],
            n_core=3,
            n_potential=1,
            n_outliers=0,
            timestamp=1000.0,
        )

        diff = drift.compare(snap, snap)
        assert diff["appeared"] == 0
        assert diff["disappeared"] == 0
        assert len(diff["moved"]) == 0

    def test_compare_detects_new_cluster(self):
        np.random.seed(42)
        c1 = np.random.randn(64).astype(np.float32)
        c1 /= np.linalg.norm(c1)
        c2 = np.random.randn(64).astype(np.float32)
        c2 /= np.linalg.norm(c2)

        snap_a = Snapshot(
            centroids=[c1],
            weights=[10.0],
            counts=[100],
            n_core=1, n_potential=0, n_outliers=0,
            timestamp=1000.0,
        )
        snap_b = Snapshot(
            centroids=[c1, c2],
            weights=[10.0, 5.0],
            counts=[100, 50],
            n_core=2, n_potential=0, n_outliers=0,
            timestamp=2000.0,
        )

        diff = drift.compare(snap_a, snap_b)
        assert diff["appeared"] == 1

    def test_compare_detects_weight_growth(self):
        np.random.seed(42)
        c1 = np.random.randn(64).astype(np.float32)
        c1 /= np.linalg.norm(c1)

        snap_a = Snapshot(
            centroids=[c1], weights=[5.0], counts=[50],
            n_core=1, n_potential=0, n_outliers=0, timestamp=1000.0,
        )
        snap_b = Snapshot(
            centroids=[c1], weights=[15.0], counts=[150],
            n_core=1, n_potential=0, n_outliers=0, timestamp=2000.0,
        )

        diff = drift.compare(snap_a, snap_b)
        assert len(diff["grew"]) == 1
        assert diff["grew"][0]["delta"] == 10.0

    def test_detect_returns_events_sorted_by_magnitude(self):
        np.random.seed(42)
        c1 = np.random.randn(64).astype(np.float32)
        c1 /= np.linalg.norm(c1)
        c2 = np.random.randn(64).astype(np.float32)
        c2 /= np.linalg.norm(c2)
        c3 = np.random.randn(64).astype(np.float32)
        c3 /= np.linalg.norm(c3)

        snaps = [
            Snapshot(
                centroids=[c1], weights=[10.0], counts=[100],
                n_core=1, n_potential=0, n_outliers=0,
                timestamp=1000.0, label="t0",
            ),
            Snapshot(
                centroids=[c1], weights=[10.0], counts=[100],
                n_core=1, n_potential=0, n_outliers=0,
                timestamp=2000.0, label="t1",
            ),
            Snapshot(
                centroids=[c1, c2, c3], weights=[10.0, 5.0, 3.0], counts=[100, 50, 30],
                n_core=3, n_potential=0, n_outliers=0,
                timestamp=3000.0, label="t2",
            ),
        ]

        events = drift.detect(snaps)
        assert len(events) == 2
        # t1→t2 has higher magnitude (two new clusters appeared)
        assert events[0]["from_label"] == "t1"
        # t0→t1 has zero magnitude (nothing changed)
        assert events[1]["magnitude"] == 0

    def test_compare_empty_snapshots(self):
        snap = Snapshot(
            centroids=[], weights=[], counts=[],
            n_core=0, n_potential=0, n_outliers=0, timestamp=0.0,
        )
        diff = drift.compare(snap, snap)
        assert diff["appeared"] == 0
        assert diff["disappeared"] == 0


# --- HDC ---


class TestHDC:
    def setup_method(self):
        self.encoder = hdc.HDCEncoder(embed_dim=64, hdc_dim=4000, seed=42)

    def test_signature_bundle_is_deterministic(self):
        np.random.seed(42)
        emb = np.random.randn(64).astype(np.float32)
        chunks = [Chunk(text="hello", index=0, embedding=emb)]

        sig_a = self.encoder.signature_bundle(chunks)
        sig_b = self.encoder.signature_bundle(chunks)
        assert np.array_equal(sig_a, sig_b)

    def test_similar_chunks_produce_similar_signatures(self):
        np.random.seed(42)
        base = np.random.randn(64).astype(np.float32)
        base /= np.linalg.norm(base)

        chunks_a = [Chunk(text="", index=0, embedding=base + np.random.randn(64) * 0.01)]
        chunks_b = [Chunk(text="", index=0, embedding=base + np.random.randn(64) * 0.01)]

        sig_a = self.encoder.signature_bundle(chunks_a)
        sig_b = self.encoder.signature_bundle(chunks_b)

        sim = self.encoder.similarity(sig_a, sig_b)
        assert sim > 0.5

    def test_dissimilar_chunks_produce_dissimilar_signatures(self):
        np.random.seed(42)
        emb_a = np.random.randn(64).astype(np.float32)
        emb_b = np.random.randn(64).astype(np.float32)

        sig_a = self.encoder.signature_bundle([Chunk(text="", index=0, embedding=emb_a)])
        sig_b = self.encoder.signature_bundle([Chunk(text="", index=0, embedding=emb_b)])

        sim = self.encoder.similarity(sig_a, sig_b)
        assert sim < 0.5

    def test_phase_encoding_differs_from_bundle(self):
        np.random.seed(42)
        chunks = [
            Chunk(text="", index=i, embedding=np.random.randn(64).astype(np.float32))
            for i in range(5)
        ]

        sig_bundle = self.encoder.signature_bundle(chunks)
        sig_phase = self.encoder.signature_phase(chunks)

        # They should NOT be identical — phase encoding adds structure
        assert not np.array_equal(sig_bundle, sig_phase)

    def test_noise_resilience(self):
        """Corrupt 30% of chunks and the signature should still be similar."""
        np.random.seed(42)
        chunks = [
            Chunk(text="", index=i, embedding=np.random.randn(64).astype(np.float32))
            for i in range(10)
        ]

        full_sig = self.encoder.signature_bundle(chunks)

        # Keep only 70% of chunks
        partial = chunks[:7]
        partial_sig = self.encoder.signature_bundle(partial)

        sim = self.encoder.similarity(full_sig, partial_sig)
        assert sim > 0.4  # Should still be meaningfully similar

    def test_empty_chunks_returns_zero(self):
        sig = self.encoder.signature_bundle([])
        assert np.all(sig == 0)


# --- Neighborhood ---


class TestNeighborhood:
    def test_similar_finds_nearest(self):
        np.random.seed(42)
        engine = Engine(dimensions=64, epsilon=0.3, mu=3.0, maintenance_interval=100)

        target = np.random.randn(64).astype(np.float32)
        target /= np.linalg.norm(target)

        # Insert target + noise
        near = target + np.random.randn(64) * 0.01
        near = (near / np.linalg.norm(near)).astype(np.float32)
        engine.ingest(_make_conversation(near, "near"))

        # Insert distant vectors
        for i in range(10):
            far = np.random.randn(64).astype(np.float32)
            far /= np.linalg.norm(far)
            engine.ingest(_make_conversation(far, f"far-{i}"))

        state = engine.state
        results = neighborhood.similar(state, target, k=3)

        assert len(results) == 3
        assert results[0]["metadata"]["id"] == "near"

    def test_between_finds_midpoint(self):
        np.random.seed(42)
        engine = Engine(dimensions=64, epsilon=0.3, mu=3.0, maintenance_interval=100)

        a = np.zeros(64, dtype=np.float32)
        a[0] = 1.0
        b = np.zeros(64, dtype=np.float32)
        b[1] = 1.0

        # Insert a vector at the midpoint
        mid = (a + b) / np.sqrt(2)
        engine.ingest(_make_conversation(mid, "mid"))

        # Insert distant vectors
        for i in range(10):
            far = np.random.randn(64).astype(np.float32)
            far /= np.linalg.norm(far)
            engine.ingest(_make_conversation(far, f"far-{i}"))

        state = engine.state
        results = neighborhood.between(state, a, b, k=1)

        assert results[0]["metadata"]["id"] == "mid"

    def test_radius_filters_by_threshold(self):
        np.random.seed(42)
        engine = Engine(dimensions=64, epsilon=0.3, mu=3.0, maintenance_interval=100)

        center = np.random.randn(64).astype(np.float32)
        center /= np.linalg.norm(center)

        # Insert 5 near vectors and 5 far vectors
        for i in range(5):
            near = center + np.random.randn(64) * 0.01
            near = (near / np.linalg.norm(near)).astype(np.float32)
            engine.ingest(_make_conversation(near, f"near-{i}"))

        for i in range(5):
            far = np.random.randn(64).astype(np.float32)
            far /= np.linalg.norm(far)
            engine.ingest(_make_conversation(far, f"far-{i}"))

        state = engine.state
        results = neighborhood.radius(state, center, min_similarity=0.9)

        for r in results:
            assert r["similarity"] >= 0.9
            assert r["metadata"]["id"].startswith("near")

    def test_empty_state_returns_empty(self):
        engine = Engine(dimensions=64)
        state = engine.state

        query = np.random.randn(64).astype(np.float32)
        assert neighborhood.similar(state, query) == []
        assert neighborhood.radius(state, query) == []
