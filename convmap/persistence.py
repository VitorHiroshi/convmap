"""Persistent map storage — save and load engine state.

Uses a directory with two files:
  - meta.json: engine config, cluster metadata, vector metadata, snapshots
  - arrays.npz: all numpy arrays (centroids, vectors) in compressed format
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .engine import Engine
from .types import MicroCluster, Snapshot

_VERSION = 1


def save(engine: Engine, path: str | Path) -> Path:
    """Save engine state to a .convmap directory."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    dims = engine.dimensions

    meta = {
        "version": _VERSION,
        "config": {
            "dimensions": dims,
            "epsilon": engine.epsilon,
            "mu": engine.mu,
            "beta": engine.beta,
            "decay": engine.decay,
            "max_recent": engine.max_recent,
            "maintenance_interval": engine.maintenance_interval,
            "step": engine._step,
            "max_snapshots": engine._max_snapshots,
        },
        "core_clusters": [_serialize_cluster(mc) for mc in engine.core_clusters],
        "potential_clusters": [_serialize_cluster(mc) for mc in engine.potential_clusters],
        "recent_metadata": [m for _, m in engine.recent_vectors],
        "outlier_metadata": [m for _, m in engine.outlier_buffer],
        "snapshots": _serialize_snapshots(engine.snapshots),
    }

    with (path / "meta.json").open("w") as f:
        json.dump(meta, f)

    arrays = {}

    if engine.core_clusters:
        arrays["core_centroids"] = np.array(
            [mc.centroid for mc in engine.core_clusters]
        )

    if engine.potential_clusters:
        arrays["potential_centroids"] = np.array(
            [mc.centroid for mc in engine.potential_clusters]
        )

    if engine.recent_vectors:
        arrays["recent_vectors"] = np.array([v for v, _ in engine.recent_vectors])

    if engine.outlier_buffer:
        arrays["outlier_vectors"] = np.array([v for v, _ in engine.outlier_buffer])

    all_snap_centroids = []
    for snap in engine.snapshots:
        all_snap_centroids.extend(snap.centroids)
    if all_snap_centroids:
        arrays["snapshot_centroids"] = np.array(all_snap_centroids)

    np.savez_compressed(path / "arrays.npz", **arrays)
    return path


def load(path: str | Path) -> Engine:
    """Load engine state from a .convmap directory."""
    path = Path(path)

    with (path / "meta.json").open() as f:
        meta = json.load(f)

    if meta.get("version", 0) > _VERSION:
        raise ValueError(
            f"Map version {meta['version']} is newer than supported ({_VERSION})"
        )

    config = meta["config"]
    dims = config["dimensions"]

    arrays_path = path / "arrays.npz"
    arrays = dict(np.load(arrays_path, allow_pickle=False)) if arrays_path.exists() else {}

    engine = Engine(
        dimensions=dims,
        epsilon=config["epsilon"],
        mu=config["mu"],
        beta=config["beta"],
        decay=config["decay"],
        max_recent=config["max_recent"],
        maintenance_interval=config["maintenance_interval"],
    )
    engine._step = config["step"]
    engine._max_snapshots = config["max_snapshots"]

    core_centroids = arrays.get("core_centroids", np.empty((0, dims)))
    for i, mc_data in enumerate(meta["core_clusters"]):
        engine.core_clusters.append(_deserialize_cluster(mc_data, core_centroids[i]))

    pot_centroids = arrays.get("potential_centroids", np.empty((0, dims)))
    for i, mc_data in enumerate(meta["potential_clusters"]):
        engine.potential_clusters.append(_deserialize_cluster(mc_data, pot_centroids[i]))

    recent_vecs = arrays.get("recent_vectors", np.empty((0, dims)))
    for i, m in enumerate(meta["recent_metadata"]):
        engine.recent_vectors.append((recent_vecs[i], m))

    outlier_vecs = arrays.get("outlier_vectors", np.empty((0, dims)))
    for i, m in enumerate(meta["outlier_metadata"]):
        engine.outlier_buffer.append((outlier_vecs[i], m))

    snap_centroids = arrays.get("snapshot_centroids", np.empty((0, dims)))
    engine.snapshots = _deserialize_snapshots(meta["snapshots"], snap_centroids)

    return engine


def _serialize_cluster(mc: MicroCluster) -> dict:
    return {
        "weight": mc.weight,
        "radius": mc.radius,
        "created_at": mc.created_at,
        "updated_at": mc.updated_at,
        "count": mc.count,
        "label": mc.label,
    }


def _deserialize_cluster(data: dict, centroid: np.ndarray) -> MicroCluster:
    return MicroCluster(
        centroid=centroid.astype(np.float32),
        weight=data["weight"],
        radius=data["radius"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        count=data["count"],
        label=data.get("label"),
    )


def _serialize_snapshots(snapshots: list[Snapshot]) -> list[dict]:
    result = []
    offset = 0
    for snap in snapshots:
        n = len(snap.centroids)
        result.append({
            "weights": snap.weights,
            "counts": snap.counts,
            "n_core": snap.n_core,
            "n_potential": snap.n_potential,
            "n_outliers": snap.n_outliers,
            "timestamp": snap.timestamp,
            "label": snap.label,
            "centroid_offset": offset,
            "centroid_count": n,
        })
        offset += n
    return result


def _deserialize_snapshots(
    data: list[dict], centroids: np.ndarray
) -> list[Snapshot]:
    snapshots = []
    for s in data:
        offset = s["centroid_offset"]
        count = s["centroid_count"]
        snap_centroids = [
            centroids[i].astype(np.float32) for i in range(offset, offset + count)
        ]
        snapshots.append(Snapshot(
            centroids=snap_centroids,
            weights=s["weights"],
            counts=s["counts"],
            n_core=s["n_core"],
            n_potential=s["n_potential"],
            n_outliers=s["n_outliers"],
            timestamp=s["timestamp"],
            label=s.get("label"),
        ))
    return snapshots
