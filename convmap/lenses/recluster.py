"""Recluster lens — query-time re-clustering with different parameters.

Re-clusters the raw vectors stored in recent_vectors without modifying
the persisted map. Lets you explore different granularity levels on the fly.
"""

from __future__ import annotations

import time

import numpy as np

from ..types import MapState, MicroCluster


def recluster(
    state: MapState,
    epsilon: float = 0.15,
    mu: float = 3.0,
    beta: float = 0.3,
) -> dict:
    """Re-cluster recent vectors with different parameters.

    Args:
        state: current map state (uses recent_vectors only).
        epsilon: similarity threshold for merging (1 - epsilon = min cosine sim).
            Lower = more clusters. Default 0.15 (min sim 0.85).
        mu: minimum weight to become a core cluster. Lower = more core clusters.
        beta: fraction of mu for potential cluster survival.

    Returns:
        Dict with core_clusters, potential_clusters, outlier_count,
        and per-cluster member lists with metadata.
    """
    if not state.recent_vectors:
        return {
            "params": {"epsilon": epsilon, "mu": mu, "beta": beta},
            "core_clusters": [],
            "potential_clusters": [],
            "outlier_count": 0,
            "total_vectors": 0,
        }

    core: list[MicroCluster] = []
    potential: list[MicroCluster] = []
    outliers: list[tuple[np.ndarray, dict]] = []
    now = time.time()

    # Replay all vectors through a fresh clustering pass
    for vec, meta in state.recent_vectors:
        if not _try_merge(vec, core, epsilon, now):
            if not _try_merge(vec, potential, epsilon, now):
                outliers.append((vec, meta))

    # Process outliers into potential clusters
    for vec, _meta in outliers:
        if not _try_merge(vec, potential, epsilon, now):
            potential.append(MicroCluster(
                centroid=vec.copy(),
                weight=1.0,
                radius=epsilon,
                created_at=now,
                updated_at=now,
                count=1,
            ))
    remaining_outliers = 0

    # Promote / demote
    new_core = []
    new_potential = []
    for mc in potential:
        if mc.weight >= mu:
            new_core.append(mc)
        elif mc.weight >= beta * mu:
            new_potential.append(mc)
        else:
            remaining_outliers += mc.count

    for mc in core:
        if mc.weight >= mu:
            new_core.append(mc)
        else:
            new_potential.append(mc)

    core = sorted(new_core, key=lambda mc: mc.weight, reverse=True)
    potential = sorted(new_potential, key=lambda mc: mc.weight, reverse=True)

    # Assign vectors to clusters for reporting
    cluster_members = _assign_members(state.recent_vectors, core)

    return {
        "params": {"epsilon": epsilon, "mu": mu, "beta": beta},
        "total_vectors": len(state.recent_vectors),
        "core_clusters": [
            {
                "index": i,
                "weight": mc.weight,
                "count": mc.count,
                "radius": mc.radius,
                "member_count": len(cluster_members[i]),
                "top_members": [
                    {"similarity": sim, "metadata": meta}
                    for sim, meta in cluster_members[i][:5]
                ],
            }
            for i, mc in enumerate(core)
        ],
        "potential_clusters": [
            {
                "weight": mc.weight,
                "count": mc.count,
            }
            for mc in potential[:20]
        ],
        "outlier_count": remaining_outliers,
    }


def sweep(
    state: MapState,
    epsilon_values: list[float] | None = None,
    mu: float = 3.0,
) -> list[dict]:
    """Sweep across epsilon values to find the right granularity.

    Returns a summary for each epsilon showing cluster counts.
    """
    if epsilon_values is None:
        epsilon_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    results = []
    for eps in epsilon_values:
        r = recluster(state, epsilon=eps, mu=mu)
        results.append({
            "epsilon": eps,
            "min_similarity": round(1 - eps, 2),
            "core_clusters": len(r["core_clusters"]),
            "potential_clusters": len(r["potential_clusters"]),
            "outlier_count": r["outlier_count"],
        })
    return results


def _try_merge(
    vector: np.ndarray,
    clusters: list[MicroCluster],
    epsilon: float,
    now: float,
) -> bool:
    if not clusters:
        return False

    best_sim = -1.0
    best_idx = -1
    for i, mc in enumerate(clusters):
        sim = mc.similarity(vector)
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    if best_sim < (1 - epsilon):
        return False

    mc = clusters[best_idx]
    mc.count += 1
    mc.weight += 1
    lr = 1.0 / mc.count
    mc.centroid = mc.centroid * (1 - lr) + vector * lr
    mc.centroid = mc.centroid / (np.linalg.norm(mc.centroid) + 1e-8)
    mc.updated_at = now
    return True


def _assign_members(
    vectors: list[tuple[np.ndarray, dict]] | object,
    clusters: list[MicroCluster],
) -> dict[int, list[tuple[float, dict]]]:
    """Assign each vector to its nearest core cluster."""
    members: dict[int, list[tuple[float, dict]]] = {i: [] for i in range(len(clusters))}

    if not clusters:
        return members

    for vec, meta in vectors:
        best_sim = -1.0
        best_idx = 0
        for i, mc in enumerate(clusters):
            sim = mc.similarity(vec)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        members[best_idx].append((best_sim, meta))

    # Sort each cluster's members by similarity descending
    for idx in members:
        members[idx].sort(key=lambda x: x[0], reverse=True)

    return members
