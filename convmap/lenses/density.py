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
