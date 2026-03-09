"""Deterministic query interface for the map.

All queries are read-only operations over the engine state.
Timestamps are a first-class filter dimension. Everything else
lives in the vector space.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .types import MapState, MicroCluster


def concept(
    state: MapState,
    query_embedding: np.ndarray,
    k: int = 10,
    time_range: tuple[float, float] | None = None,
) -> list[dict]:
    """Find recent vectors most similar to a concept embedding.

    The query_embedding should come from embedder.embed_text("some concept").
    Optionally filter by timestamp range.
    """
    vectors = _filter_by_time(state.recent_vectors, time_range)
    if not vectors:
        return []

    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    mat = np.array([v for v, _ in vectors])
    sims = mat @ query_norm

    top_k = min(k, len(sims))
    indices = np.argsort(sims)[-top_k:][::-1]

    return [
        {
            "similarity": float(sims[idx]),
            "metadata": vectors[idx][1],
            "vector": vectors[idx][0],
        }
        for idx in indices
    ]


def time_window(
    state: MapState,
    start: float,
    end: float,
) -> MapState:
    """Return a filtered MapState containing only vectors within a time range.

    The returned state can be passed to any lens for time-scoped analysis.
    """
    filtered = _filter_by_time(state.recent_vectors, (start, end))
    filtered_outliers = _filter_by_time(state.outlier_buffer, (start, end))

    return MapState(
        core_clusters=state.core_clusters,
        potential_clusters=state.potential_clusters,
        outlier_buffer=filtered_outliers,
        recent_vectors=filtered,
        snapshots=state.snapshots,
        dimensions=state.dimensions,
        timestamp=state.timestamp,
    )


def funnel(state: MapState) -> list[dict]:
    """Analyze the call funnel based on cluster sizes and relationships.

    Returns clusters ordered by size (largest first) with their share
    of total volume, cumulative drop-off, and inter-cluster distances.
    """
    if not state.core_clusters:
        return []

    total = sum(mc.count for mc in state.core_clusters)
    if total == 0:
        return []

    # Sort by count descending
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


def compare_windows(
    state: MapState,
    window_a: tuple[float, float],
    window_b: tuple[float, float],
) -> dict:
    """Compare two time windows within the recent vector buffer.

    Returns distribution differences: which clusters gained/lost share.
    """
    vecs_a = _filter_by_time(state.recent_vectors, window_a)
    vecs_b = _filter_by_time(state.recent_vectors, window_b)

    if not vecs_a or not vecs_b or not state.core_clusters:
        return {
            "window_a_count": len(vecs_a) if vecs_a else 0,
            "window_b_count": len(vecs_b) if vecs_b else 0,
            "cluster_shifts": [],
        }

    dist_a = _cluster_distribution(state.core_clusters, vecs_a)
    dist_b = _cluster_distribution(state.core_clusters, vecs_b)

    shifts = []
    for i in range(len(state.core_clusters)):
        share_a = dist_a[i]
        share_b = dist_b[i]
        delta = share_b - share_a
        shifts.append({
            "cluster_index": i,
            "label": state.core_clusters[i].label,
            "share_a": share_a,
            "share_b": share_b,
            "delta": delta,
        })

    shifts.sort(key=lambda x: abs(x["delta"]), reverse=True)

    return {
        "window_a_count": len(vecs_a),
        "window_b_count": len(vecs_b),
        "cluster_shifts": shifts,
    }


def segment(
    state: MapState,
    cluster_index: int,
    time_range: tuple[float, float] | None = None,
) -> list[dict]:
    """Get all recent vectors assigned to a specific cluster.

    Useful for drilling into a cluster to see its members.
    """
    if cluster_index >= len(state.core_clusters):
        return []

    mc = state.core_clusters[cluster_index]
    vectors = _filter_by_time(state.recent_vectors, time_range)
    if not vectors:
        return []

    results = []
    for vec, meta in vectors:
        sim = mc.similarity(vec)
        # Assign to nearest cluster — check if this cluster is the nearest
        best_sim = max(c.similarity(vec) for c in state.core_clusters)
        if abs(sim - best_sim) < 1e-6:  # this is the nearest cluster
            results.append({
                "similarity": sim,
                "metadata": meta,
                "vector": vec,
            })

    return sorted(results, key=lambda x: x["similarity"], reverse=True)


def histogram(
    state: MapState,
    time_range: tuple[float, float] | None = None,
) -> dict:
    """Distribution of recent vectors across clusters.

    Returns per-cluster counts and shares.
    """
    vectors = _filter_by_time(state.recent_vectors, time_range)
    if not vectors or not state.core_clusters:
        return {"total": 0, "clusters": [], "unassigned": 0}

    assignments = {i: 0 for i in range(len(state.core_clusters))}
    unassigned = 0

    for vec, _ in vectors:
        sims = [mc.similarity(vec) for mc in state.core_clusters]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        if best_sim >= 0.5:  # minimum threshold
            assignments[best_idx] += 1
        else:
            unassigned += 1

    total = len(vectors)
    clusters = []
    for i, count in sorted(assignments.items(), key=lambda x: x[1], reverse=True):
        clusters.append({
            "index": i,
            "label": state.core_clusters[i].label,
            "count": count,
            "share": count / total if total > 0 else 0,
        })

    return {"total": total, "clusters": clusters, "unassigned": unassigned}


def anomalies(
    state: MapState,
    threshold: float = 0.5,
    time_range: tuple[float, float] | None = None,
) -> list[dict]:
    """Find recent vectors that don't fit well into any cluster.

    These are potential emerging patterns or edge cases.
    """
    vectors = _filter_by_time(state.recent_vectors, time_range)
    if not vectors or not state.core_clusters:
        return []

    results = []
    for vec, meta in vectors:
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


# --- Internal helpers ---


def _filter_by_time(
    vectors: list[tuple[np.ndarray, dict]],
    time_range: tuple[float, float] | None,
) -> list[tuple[np.ndarray, dict]]:
    """Filter vectors by timestamp in metadata."""
    if time_range is None:
        return vectors

    start, end = time_range
    filtered = []
    for vec, meta in vectors:
        ts = meta.get("timestamp")
        if ts is None:
            continue
        try:
            t = float(ts)
        except (ValueError, TypeError):
            continue
        if start <= t <= end:
            filtered.append((vec, meta))
    return filtered


def _cluster_distribution(
    clusters: list[MicroCluster],
    vectors: list[tuple[np.ndarray, dict]],
) -> list[float]:
    """Compute share of vectors per cluster."""
    counts = [0] * len(clusters)
    for vec, _ in vectors:
        sims = [mc.similarity(vec) for mc in clusters]
        best_idx = int(np.argmax(sims))
        counts[best_idx] += 1

    total = sum(counts)
    if total == 0:
        return [0.0] * len(clusters)
    return [c / total for c in counts]
