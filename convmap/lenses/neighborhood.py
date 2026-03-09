"""Neighborhood lens — similarity search over recent vectors."""

from __future__ import annotations

import numpy as np

from ..types import MapState


def similar(state: MapState, query: np.ndarray, k: int = 10) -> list[dict]:
    """Find the k most similar recent vectors to a query point."""
    if not state.recent_vectors:
        return []

    vectors = np.array([v for v, _ in state.recent_vectors])
    query_norm = query / (np.linalg.norm(query) + 1e-8)

    # Batch cosine similarity
    similarities = vectors @ query_norm

    top_k = min(k, len(similarities))
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        _, meta = state.recent_vectors[idx]
        results.append({
            "similarity": float(similarities[idx]),
            "metadata": meta,
            "vector": vectors[idx],
        })
    return results


def between(
    state: MapState,
    anchor_a: np.ndarray,
    anchor_b: np.ndarray,
    k: int = 10,
    bias: float = 0.5,
) -> list[dict]:
    """Find recent vectors that sit between two anchor points.

    bias=0.5 means equidistant. bias=0.2 means closer to anchor_a.
    Useful for finding conversations that bridge two patterns.
    """
    if not state.recent_vectors:
        return []

    # Interpolated query point
    target = anchor_a * (1 - bias) + anchor_b * bias
    target = target / (np.linalg.norm(target) + 1e-8)

    return similar(state, target, k)


def radius(
    state: MapState,
    center: np.ndarray,
    min_similarity: float = 0.8,
) -> list[dict]:
    """Find all recent vectors within a similarity radius of a center point."""
    if not state.recent_vectors:
        return []

    vectors = np.array([v for v, _ in state.recent_vectors])
    center_norm = center / (np.linalg.norm(center) + 1e-8)

    similarities = vectors @ center_norm

    results = []
    for idx in range(len(similarities)):
        if similarities[idx] >= min_similarity:
            _, meta = state.recent_vectors[idx]
            results.append({
                "similarity": float(similarities[idx]),
                "metadata": meta,
                "vector": vectors[idx],
            })

    return sorted(results, key=lambda x: x["similarity"], reverse=True)
