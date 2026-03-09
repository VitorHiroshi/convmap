"""TDA lens — topological data analysis for shape discovery.

Uses persistent homology to find topological features (connected components,
loops, voids) in regions of the embedding space. These features reveal
structural patterns that clustering alone cannot see.

Requires giotto-tda (optional dependency). Functions gracefully degrade
if not installed.
"""

from __future__ import annotations

import numpy as np

from ..types import MapState


def persistence(
    state: MapState,
    source: str = "core",
    max_dim: int = 1,
    max_points: int = 500,
) -> dict:
    """Compute persistence diagram from map state.

    Args:
        state: current map state.
        source: "core" for cluster centroids, "recent" for recent vectors.
        max_dim: maximum homology dimension (0=components, 1=loops, 2=voids).
        max_points: subsample if more points than this.

    Returns:
        Dictionary with persistence pairs per dimension and summary statistics.
    """
    points = _get_points(state, source, max_points)
    if len(points) < 3:
        return {"dimensions": {}, "n_points": len(points), "source": source}

    try:
        from gtda.homology import VietorisRipsPersistence
    except ImportError:
        return _fallback_persistence(points, max_dim)

    vr = VietorisRipsPersistence(
        metric="cosine",
        homology_dimensions=list(range(max_dim + 1)),
        n_jobs=1,
    )
    diagrams = vr.fit_transform(points.reshape(1, *points.shape))
    diagram = diagrams[0]

    dimensions = {}
    for dim in range(max_dim + 1):
        mask = diagram[:, 2] == dim
        pairs = diagram[mask, :2]
        lifetimes = pairs[:, 1] - pairs[:, 0]
        lifetimes = lifetimes[np.isfinite(lifetimes)]

        dimensions[dim] = {
            "n_features": int(np.sum(mask)),
            "lifetimes": sorted(lifetimes.tolist(), reverse=True)[:20],
            "mean_lifetime": float(np.mean(lifetimes)) if len(lifetimes) > 0 else 0.0,
            "max_lifetime": float(np.max(lifetimes)) if len(lifetimes) > 0 else 0.0,
        }

    return {"dimensions": dimensions, "n_points": len(points), "source": source}


def features(state: MapState, source: str = "core", max_points: int = 500) -> list[dict]:
    """Extract the most persistent topological features.

    Returns the longest-lived features across all dimensions,
    sorted by lifetime. Long-lived features represent robust
    topological structure (stable clusters, persistent loops).
    """
    result = persistence(state, source=source, max_dim=1, max_points=max_points)

    all_features = []
    for dim, info in result.get("dimensions", {}).items():
        for i, lifetime in enumerate(info.get("lifetimes", [])):
            all_features.append({
                "dimension": dim,
                "rank": i,
                "lifetime": lifetime,
                "type": _feature_type(dim),
            })

    return sorted(all_features, key=lambda x: x["lifetime"], reverse=True)


def landscape_distance(
    state_a: MapState,
    state_b: MapState,
    source: str = "core",
    max_points: int = 500,
) -> float:
    """Compute topological distance between two map states.

    Uses persistence landscape L2 distance. A high distance means the
    topological structure changed significantly between states.
    """
    feat_a = persistence(state_a, source=source, max_points=max_points)
    feat_b = persistence(state_b, source=source, max_points=max_points)

    # Simple approximation: compare lifetime distributions per dimension
    total_dist = 0.0
    all_dims = set(feat_a.get("dimensions", {}).keys()) | set(feat_b.get("dimensions", {}).keys())

    for dim in all_dims:
        lifetimes_a = feat_a.get("dimensions", {}).get(dim, {}).get("lifetimes", [])
        lifetimes_b = feat_b.get("dimensions", {}).get(dim, {}).get("lifetimes", [])

        # Pad to same length
        max_len = max(len(lifetimes_a), len(lifetimes_b))
        a = np.zeros(max_len)
        b = np.zeros(max_len)
        a[:len(lifetimes_a)] = lifetimes_a
        b[:len(lifetimes_b)] = lifetimes_b

        total_dist += float(np.linalg.norm(a - b))

    return total_dist


def _get_points(state: MapState, source: str, max_points: int) -> np.ndarray:
    """Extract point cloud from map state."""
    if source == "core":
        vecs = [mc.centroid for mc in state.core_clusters]
    elif source == "recent":
        vecs = [v for v, _ in state.recent_vectors]
    elif source == "all":
        vecs = [mc.centroid for mc in state.core_clusters]
        vecs += [mc.centroid for mc in state.potential_clusters]
        vecs += [v for v, _ in state.outlier_buffer]
    else:
        vecs = []

    if not vecs:
        return np.array([]).reshape(0, state.dimensions)

    points = np.array(vecs)

    # Subsample if too many points
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]

    return points


def _fallback_persistence(points: np.ndarray, max_dim: int) -> dict:
    """Simplified persistence without giotto-tda.

    Only computes dimension 0 (connected components) using pairwise distances.
    """
    from scipy.spatial.distance import pdist, squareform

    try:
        dists = squareform(pdist(points, metric="cosine"))
    except ImportError:
        # No scipy either — compute manually
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        normalized = points / (norms + 1e-8)
        sim_matrix = normalized @ normalized.T
        dists = 1.0 - sim_matrix

    # Simple single-linkage clustering to get merge distances
    n = len(points)
    alive = list(range(n))
    merge_dists = []

    cluster_map = {i: {i} for i in range(n)}
    dist_copy = dists.copy()
    np.fill_diagonal(dist_copy, np.inf)

    for _ in range(n - 1):
        if len(alive) < 2:
            break
        # Find minimum distance
        min_val = np.inf
        min_i, min_j = 0, 0
        for ii in range(len(alive)):
            for jj in range(ii + 1, len(alive)):
                d = dist_copy[alive[ii], alive[jj]]
                if d < min_val:
                    min_val = d
                    min_i, min_j = ii, jj

        merge_dists.append(float(min_val))

        # Merge: keep alive[min_i], remove alive[min_j]
        keep = alive[min_i]
        remove = alive[min_j]
        # Update distances (single linkage = min)
        for k in alive:
            if k != keep and k != remove:
                dist_copy[keep, k] = min(dist_copy[keep, k], dist_copy[remove, k])
                dist_copy[k, keep] = dist_copy[keep, k]
        alive.pop(min_j)

    lifetimes = sorted(merge_dists, reverse=True) if merge_dists else []

    dimensions = {
        0: {
            "n_features": len(lifetimes),
            "lifetimes": lifetimes[:20],
            "mean_lifetime": float(np.mean(lifetimes)) if lifetimes else 0.0,
            "max_lifetime": float(np.max(lifetimes)) if lifetimes else 0.0,
        }
    }

    return {"dimensions": dimensions, "n_points": len(points), "source": "fallback"}


def _feature_type(dim: int) -> str:
    return {0: "component", 1: "loop", 2: "void"}.get(dim, f"dim-{dim}")
