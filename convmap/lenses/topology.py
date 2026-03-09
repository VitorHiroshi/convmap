"""Topology lens — how regions of the map relate to each other.

Builds a graph of relationships between core clusters based on
centroid similarity, shared recent vectors, and bridge detection.
"""

from __future__ import annotations

import numpy as np

from ..types import MapState, MicroCluster


def adjacency(state: MapState, threshold: float = 0.5) -> list[dict]:
    """Build an adjacency list between core clusters based on centroid similarity.

    Returns edges sorted by similarity (strongest connections first).
    """
    clusters = state.core_clusters
    if len(clusters) < 2:
        return []

    centroids = np.array([mc.centroid for mc in clusters])
    sim_matrix = centroids @ centroids.T

    edges = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            if sim_matrix[i, j] >= threshold:
                edges.append({
                    "i": i,
                    "j": j,
                    "similarity": float(sim_matrix[i, j]),
                    "label_i": clusters[i].label,
                    "label_j": clusters[j].label,
                })

    return sorted(edges, key=lambda x: x["similarity"], reverse=True)


def bridges(state: MapState, cluster_threshold: float = 0.6) -> list[dict]:
    """Find recent vectors that sit between two or more clusters.

    A bridge vector is one that has high similarity to multiple clusters,
    indicating it connects different regions of the space.
    """
    if not state.recent_vectors or len(state.core_clusters) < 2:
        return []

    centroids = np.array([mc.centroid for mc in state.core_clusters])
    results = []

    for vec, meta in state.recent_vectors:
        sims = centroids @ vec
        above_threshold = np.where(sims >= cluster_threshold)[0]

        if len(above_threshold) >= 2:
            results.append({
                "metadata": meta,
                "cluster_indices": above_threshold.tolist(),
                "similarities": sims[above_threshold].tolist(),
                "n_clusters": len(above_threshold),
            })

    return sorted(results, key=lambda x: x["n_clusters"], reverse=True)


def isolated(state: MapState, threshold: float = 0.3) -> list[dict]:
    """Find core clusters that are far from all other clusters.

    These represent unique, standalone patterns.
    """
    clusters = state.core_clusters
    if len(clusters) < 2:
        return [{"index": 0, "label": clusters[0].label, "max_similarity": 0.0}] if clusters else []

    centroids = np.array([mc.centroid for mc in clusters])
    sim_matrix = centroids @ centroids.T
    np.fill_diagonal(sim_matrix, -1)  # ignore self-similarity

    results = []
    for i in range(len(clusters)):
        max_sim = float(np.max(sim_matrix[i]))
        if max_sim < threshold:
            results.append({
                "index": i,
                "label": clusters[i].label,
                "max_similarity": max_sim,
                "weight": clusters[i].weight,
            })

    return sorted(results, key=lambda x: x["max_similarity"])


def density_map(state: MapState) -> dict:
    """Compute the density landscape across core clusters.

    Returns per-cluster density (weight / radius) and overall statistics.
    """
    if not state.core_clusters:
        return {"clusters": [], "mean_density": 0.0, "std_density": 0.0}

    densities = []
    for mc in state.core_clusters:
        d = mc.weight / max(mc.radius, 1e-8)
        densities.append({
            "index": len(densities),
            "label": mc.label,
            "density": d,
            "weight": mc.weight,
            "radius": mc.radius,
            "count": mc.count,
        })

    density_values = [d["density"] for d in densities]
    return {
        "clusters": sorted(densities, key=lambda x: x["density"], reverse=True),
        "mean_density": float(np.mean(density_values)),
        "std_density": float(np.std(density_values)),
    }
