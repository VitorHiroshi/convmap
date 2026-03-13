"""Tests for the persistence module."""

import numpy as np
import pytest

from convmap import Engine, Chunk, EmbeddedConversation
from convmap import persistence


def _make_conversation(vector, conv_id, timestamp=None):
    meta = {"id": conv_id}
    if timestamp is not None:
        meta["timestamp"] = timestamp
    return EmbeddedConversation(
        id=conv_id,
        chunks=[Chunk(text="", index=0, embedding=vector)],
        metadata=meta,
    )


def _build_engine(n=50, dims=64, seed=42):
    np.random.seed(seed)
    engine = Engine(dimensions=dims, epsilon=0.3, mu=3.0, maintenance_interval=20)

    centers = [np.random.randn(dims).astype(np.float32) for _ in range(3)]
    per_cluster = n // 3

    base_time = 1000.0
    i = 0
    for center in centers:
        for _ in range(per_cluster):
            v = center + np.random.randn(dims) * 0.05
            v = (v / np.linalg.norm(v)).astype(np.float32)
            conv = _make_conversation(v, f"conv-{i}", timestamp=base_time + i)
            engine.ingest(conv)
            i += 1

    engine._maintain()
    return engine


class TestSaveLoad:
    def test_roundtrip_preserves_config(self, tmp_path):
        engine = _build_engine()
        path = tmp_path / "test.convmap"

        persistence.save(engine, path)
        loaded = persistence.load(path)

        assert loaded.dimensions == engine.dimensions
        assert loaded.epsilon == engine.epsilon
        assert loaded.mu == engine.mu
        assert loaded.beta == engine.beta
        assert loaded.decay == engine.decay
        assert loaded.max_recent == engine.max_recent
        assert loaded.maintenance_interval == engine.maintenance_interval
        assert loaded._step == engine._step

    def test_roundtrip_preserves_clusters(self, tmp_path):
        engine = _build_engine()
        path = tmp_path / "test.convmap"

        persistence.save(engine, path)
        loaded = persistence.load(path)

        assert len(loaded.core_clusters) == len(engine.core_clusters)
        assert len(loaded.potential_clusters) == len(engine.potential_clusters)

        for orig, restored in zip(engine.core_clusters, loaded.core_clusters):
            assert np.allclose(orig.centroid, restored.centroid, atol=1e-6)
            assert orig.weight == pytest.approx(restored.weight)
            assert orig.count == restored.count
            assert orig.label == restored.label

    def test_roundtrip_preserves_recent_vectors(self, tmp_path):
        engine = _build_engine()
        path = tmp_path / "test.convmap"

        persistence.save(engine, path)
        loaded = persistence.load(path)

        assert len(loaded.recent_vectors) == len(engine.recent_vectors)

        for (orig_v, orig_m), (loaded_v, loaded_m) in zip(
            engine.recent_vectors, loaded.recent_vectors
        ):
            assert np.allclose(orig_v, loaded_v, atol=1e-6)
            assert orig_m["id"] == loaded_m["id"]

    def test_roundtrip_preserves_snapshots(self, tmp_path):
        engine = _build_engine()
        engine.snapshot(label="before")
        engine.snapshot(label="after")

        path = tmp_path / "test.convmap"
        persistence.save(engine, path)
        loaded = persistence.load(path)

        assert len(loaded.snapshots) == 2
        assert loaded.snapshots[0].label == "before"
        assert loaded.snapshots[1].label == "after"
        assert loaded.snapshots[0].n_core == engine.snapshots[0].n_core

        for orig_c, loaded_c in zip(
            engine.snapshots[0].centroids, loaded.snapshots[0].centroids
        ):
            assert np.allclose(orig_c, loaded_c, atol=1e-6)

    def test_empty_engine_roundtrip(self, tmp_path):
        engine = Engine(dimensions=64)
        path = tmp_path / "empty.convmap"

        persistence.save(engine, path)
        loaded = persistence.load(path)

        assert loaded.dimensions == 64
        assert len(loaded.core_clusters) == 0
        assert len(loaded.recent_vectors) == 0
        assert len(loaded.snapshots) == 0

    def test_incremental_update(self, tmp_path):
        engine = _build_engine(n=30)
        path = tmp_path / "inc.convmap"
        persistence.save(engine, path)

        # Load, add more data, save again
        engine2 = persistence.load(path)
        np.random.seed(99)
        for j in range(10):
            v = np.random.randn(64).astype(np.float32)
            v /= np.linalg.norm(v)
            conv = _make_conversation(v, f"new-{j}", timestamp=2000.0 + j)
            engine2.ingest(conv)

        persistence.save(engine2, path)
        final = persistence.load(path)

        assert final._step == engine2._step
        assert len(final.recent_vectors) == len(engine2.recent_vectors)

    def test_loaded_engine_can_ingest(self, tmp_path):
        """Loaded engine should be fully functional."""
        engine = _build_engine()
        path = tmp_path / "func.convmap"
        persistence.save(engine, path)

        loaded = persistence.load(path)
        np.random.seed(123)
        v = np.random.randn(64).astype(np.float32)
        v /= np.linalg.norm(v)
        conv = _make_conversation(v, "post-load")
        loaded.ingest(conv)

        assert loaded._step == engine._step + 1
