"""Drift lens — detect what changed between time periods."""

from __future__ import annotations

import numpy as np

from ..types import Snapshot


def compare(a: Snapshot, b: Snapshot) -> dict:
    """Compare two snapshots. Returns structural and centroid-level changes.

    Matches clusters across snapshots by centroid similarity,
    then reports which appeared, disappeared, grew, shrunk, or moved.
    """
    if not a.centroids or not b.centroids:
        return {
            "appeared": len(b.centroids),
            "disappeared": len(a.centroids),
            "moved": [],
            "grew": [],
            "shrunk": [],
            "structural": _structural_diff(a, b),
        }

    # Build similarity matrix between a's and b's centroids
    a_mat = np.array(a.centroids)
    b_mat = np.array(b.centroids)
    sim_matrix = a_mat @ b_mat.T  # cosine sim (normalized vectors)

    # Match: for each cluster in a, find best match in b (and vice versa)
    match_threshold = 0.7
    a_to_b = {}
    for i in range(len(a.centroids)):
        best_j = int(np.argmax(sim_matrix[i]))
        if sim_matrix[i, best_j] >= match_threshold:
            a_to_b[i] = best_j

    matched_b = set(a_to_b.values())

    appeared = [j for j in range(len(b.centroids)) if j not in matched_b]
    disappeared = [i for i in range(len(a.centroids)) if i not in a_to_b]

    moved = []
    grew = []
    shrunk = []

    for i, j in a_to_b.items():
        sim = float(sim_matrix[i, j])
        weight_a = a.weights[i]
        weight_b = b.weights[j]
        weight_delta = weight_b - weight_a

        if sim < 0.95:  # centroid shifted meaningfully
            moved.append({
                "from_index": i,
                "to_index": j,
                "similarity": sim,
                "centroid_shift": 1 - sim,
            })

        if weight_delta > 0:
            grew.append({
                "index_a": i,
                "index_b": j,
                "weight_a": weight_a,
                "weight_b": weight_b,
                "delta": weight_delta,
            })
        elif weight_delta < 0:
            shrunk.append({
                "index_a": i,
                "index_b": j,
                "weight_a": weight_a,
                "weight_b": weight_b,
                "delta": weight_delta,
            })

    return {
        "appeared": len(appeared),
        "disappeared": len(disappeared),
        "moved": sorted(moved, key=lambda x: x["centroid_shift"], reverse=True),
        "grew": sorted(grew, key=lambda x: x["delta"], reverse=True),
        "shrunk": sorted(shrunk, key=lambda x: x["delta"]),
        "structural": _structural_diff(a, b),
    }


def detect(snapshots: list[Snapshot], min_snapshots: int = 2) -> list[dict]:
    """Detect drift across a series of snapshots.

    Returns a list of drift events between consecutive pairs,
    sorted by magnitude (most significant first).
    """
    if len(snapshots) < min_snapshots:
        return []

    events = []
    for i in range(len(snapshots) - 1):
        diff = compare(snapshots[i], snapshots[i + 1])
        magnitude = (
            diff["appeared"]
            + diff["disappeared"]
            + len(diff["moved"])
            + len(diff["grew"])
            + len(diff["shrunk"])
        )
        events.append({
            "from_label": snapshots[i].label,
            "to_label": snapshots[i + 1].label,
            "from_time": snapshots[i].timestamp,
            "to_time": snapshots[i + 1].timestamp,
            "magnitude": magnitude,
            "diff": diff,
        })

    return sorted(events, key=lambda x: x["magnitude"], reverse=True)


def _structural_diff(a: Snapshot, b: Snapshot) -> dict:
    """High-level structural changes between snapshots."""
    return {
        "core_delta": b.n_core - a.n_core,
        "potential_delta": b.n_potential - a.n_potential,
        "outlier_delta": b.n_outliers - a.n_outliers,
        "total_weight_a": sum(a.weights),
        "total_weight_b": sum(b.weights),
    }
