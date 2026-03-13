"""Density lens — cluster analysis over the current map state."""

from __future__ import annotations

import numpy as np

from ..types import MapState


def clusters(state: MapState, min_weight: float = 0.0) -> list[dict]:
    """Current core clusters, sorted by weight (strongest patterns first)."""
    results = []
    for mc in state.core_clusters:
        if mc.weight < min_weight:
            continue
        results.append({
            "centroid": mc.centroid,
            "weight": mc.weight,
            "count": mc.count,
            "label": mc.label,
            "radius": mc.radius,
            "age_seconds": state.timestamp - mc.created_at,
        })
    return sorted(results, key=lambda x: x["weight"], reverse=True)


def emerging(state: MapState) -> list[dict]:
    """Potential clusters gaining weight — patterns that are forming."""
    results = []
    for mc in state.potential_clusters:
        age = max(state.timestamp - mc.created_at, 1.0)
        results.append({
            "centroid": mc.centroid,
            "weight": mc.weight,
            "count": mc.count,
            "age_seconds": age,
            "momentum": mc.weight / age,
        })
    return sorted(results, key=lambda x: x["momentum"], reverse=True)


def nearest(state: MapState, point: np.ndarray, k: int = 5) -> list[dict]:
    """Find the k core clusters most similar to a point."""
    if not state.core_clusters:
        return []

    scored = []
    for mc in state.core_clusters:
        sim = mc.similarity(point)
        scored.append({
            "centroid": mc.centroid,
            "similarity": sim,
            "weight": mc.weight,
            "count": mc.count,
            "label": mc.label,
        })
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:k]


def outliers(state: MapState) -> list[dict]:
    """Current outlier buffer — points that don't fit any cluster."""
    return [{"vector": vec, "metadata": meta} for vec, meta in state.outlier_buffer]


def distribution(state: MapState) -> dict:
    """High-level distribution summary of the map."""
    core_weights = [mc.weight for mc in state.core_clusters]
    potential_weights = [mc.weight for mc in state.potential_clusters]

    return {
        "core_count": len(state.core_clusters),
        "potential_count": len(state.potential_clusters),
        "outlier_count": len(state.outlier_buffer),
        "core_total_weight": sum(core_weights),
        "core_weight_std": float(np.std(core_weights)) if core_weights else 0.0,
        "potential_total_weight": sum(potential_weights),
    }


def anomalies(
    state: MapState, threshold: float = 0.5
) -> list[dict]:
    """Find recent vectors that don't fit well into any cluster.

    Returns vectors whose best cluster similarity is below threshold,
    sorted by similarity ascending (worst fit first).
    """
    if not state.recent_vectors or not state.core_clusters:
        return []

    results = []
    for vec, meta in state.recent_vectors:
        sims = [mc.similarity(vec) for mc in state.core_clusters]
        best_sim = max(sims)
        if best_sim < threshold:
            results.append({
                "best_similarity": best_sim,
                "best_cluster": int(np.argmax(sims)),
                "metadata": meta,
                "vector": vec,
            })

    return sorted(results, key=lambda x: x["best_similarity"])


def histogram(state: MapState) -> dict:
    """Distribution of recent vectors across clusters.

    Assigns each vector to its nearest cluster (if similarity >= 0.5),
    returns per-cluster counts and shares.
    """
    if not state.recent_vectors or not state.core_clusters:
        return {"total": 0, "clusters": [], "unassigned": 0}

    assignments = {i: 0 for i in range(len(state.core_clusters))}
    unassigned = 0

    for vec, _ in state.recent_vectors:
        sims = [mc.similarity(vec) for mc in state.core_clusters]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        if best_sim >= 0.5:
            assignments[best_idx] += 1
        else:
            unassigned += 1

    total = len(state.recent_vectors)
    cluster_list = []
    for i, count in sorted(assignments.items(), key=lambda x: x[1], reverse=True):
        cluster_list.append({
            "index": i,
            "label": state.core_clusters[i].label,
            "count": count,
            "share": count / total if total > 0 else 0,
        })

    return {"total": total, "clusters": cluster_list, "unassigned": unassigned}


def segment(state: MapState, cluster_index: int) -> list[dict]:
    """Get all recent vectors assigned to a specific cluster.

    Returns vectors whose nearest cluster is the given index,
    sorted by similarity descending.
    """
    if cluster_index >= len(state.core_clusters):
        return []

    mc = state.core_clusters[cluster_index]
    if not state.recent_vectors:
        return []

    results = []
    for vec, meta in state.recent_vectors:
        sims = [c.similarity(vec) for c in state.core_clusters]
        best_idx = int(np.argmax(sims))
        if best_idx == cluster_index:
            results.append({
                "similarity": sims[cluster_index],
                "metadata": meta,
                "vector": vec,
            })

    return sorted(results, key=lambda x: x["similarity"], reverse=True)


def funnel(state: MapState) -> list[dict]:
    """Cluster funnel — clusters ranked by member count with cumulative shares.

    Returns clusters ordered by size (largest first) with their share
    of total volume and cumulative drop-off.
    """
    if not state.core_clusters:
        return []

    total = sum(mc.count for mc in state.core_clusters)
    if total == 0:
        return []

    sorted_clusters = sorted(state.core_clusters, key=lambda mc: mc.count, reverse=True)

    stages = []
    cumulative = 0
    for i, mc in enumerate(sorted_clusters):
        share = mc.count / total
        cumulative += mc.count
        stages.append({
            "rank": i,
            "count": mc.count,
            "share": share,
            "cumulative_share": cumulative / total,
            "weight": mc.weight,
            "label": mc.label,
            "centroid": mc.centroid,
        })

    return stages


def cluster_distribution(state: MapState, vectors: list) -> list[float]:
    """Compute share of vectors assigned to each cluster.

    Used by compare_windows and other composition queries.
    """
    if not state.core_clusters or not vectors:
        return [0.0] * len(state.core_clusters) if state.core_clusters else []

    counts = [0] * len(state.core_clusters)
    for vec, _ in vectors:
        sims = [mc.similarity(vec) for mc in state.core_clusters]
        best_idx = int(np.argmax(sims))
        counts[best_idx] += 1

    total = sum(counts)
    if total == 0:
        return [0.0] * len(state.core_clusters)
    return [c / total for c in counts]
