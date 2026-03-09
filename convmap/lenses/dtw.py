"""DTW lens — dynamic time warping for precise sequence comparison.

Compares conversations at the chunk level, aligning their trajectories
through embedding space. Handles stretching, compression, and misalignment.

Only used on small subsets (O(n*m) per pair). The other lenses identify
which conversations to compare; DTW tells you exactly how they align.
"""

from __future__ import annotations

import numpy as np

from ..types import EmbeddedConversation


def distance(a: EmbeddedConversation, b: EmbeddedConversation) -> float:
    """DTW distance between two conversations based on chunk embeddings.

    Uses cosine distance (1 - cosine_similarity) as the point-to-point metric.
    """
    seq_a = _get_embeddings(a)
    seq_b = _get_embeddings(b)

    if len(seq_a) == 0 or len(seq_b) == 0:
        return float("inf")

    cost_matrix = _dtw_matrix(seq_a, seq_b)
    return float(cost_matrix[-1, -1])


def alignment(a: EmbeddedConversation, b: EmbeddedConversation) -> dict:
    """Full DTW alignment between two conversations.

    Returns the warping path, per-step costs, and overall distance.
    The path shows which chunks in A correspond to which chunks in B.
    """
    seq_a = _get_embeddings(a)
    seq_b = _get_embeddings(b)

    if len(seq_a) == 0 or len(seq_b) == 0:
        return {"distance": float("inf"), "path": [], "step_costs": []}

    cost_matrix = _dtw_matrix(seq_a, seq_b)
    path = _traceback(cost_matrix, seq_a, seq_b)

    return {
        "distance": float(cost_matrix[-1, -1]),
        "path": path,
        "len_a": len(seq_a),
        "len_b": len(seq_b),
        "compression_ratio": len(path) / max(len(seq_a), len(seq_b)),
    }


def pairwise(conversations: list[EmbeddedConversation]) -> np.ndarray:
    """Compute pairwise DTW distance matrix for a set of conversations.

    Only use on small subsets (< 100 conversations).
    """
    n = len(conversations)
    matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            d = distance(conversations[i], conversations[j])
            matrix[i, j] = d
            matrix[j, i] = d

    return matrix


def most_similar(
    target: EmbeddedConversation,
    candidates: list[EmbeddedConversation],
    k: int = 5,
) -> list[dict]:
    """Find the k most similar conversations to a target using DTW distance."""
    scored = []
    for candidate in candidates:
        d = distance(target, candidate)
        scored.append({
            "id": candidate.id,
            "distance": d,
            "metadata": candidate.metadata,
        })

    scored.sort(key=lambda x: x["distance"])
    return scored[:k]


def _get_embeddings(conv: EmbeddedConversation) -> np.ndarray:
    """Extract chunk embeddings as a matrix."""
    embs = [c.embedding for c in conv.chunks if c.embedding is not None]
    if not embs:
        return np.array([])
    return np.array(embs)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (norm_a * norm_b))


def _dtw_matrix(seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
    """Compute the DTW accumulated cost matrix."""
    n, m = len(seq_a), len(seq_b)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = _cosine_distance(seq_a[i - 1], seq_b[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return cost[1:, 1:]  # return without the padding row/col


def _traceback(cost_matrix: np.ndarray, seq_a: np.ndarray, seq_b: np.ndarray) -> list[dict]:
    """Trace back through the cost matrix to find the optimal warping path."""
    n, m = cost_matrix.shape
    i, j = n - 1, m - 1
    path = []

    while i > 0 or j > 0:
        d = _cosine_distance(seq_a[i], seq_b[j])
        path.append({"i": int(i), "j": int(j), "cost": d})

        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = [
                (cost_matrix[i - 1, j - 1], i - 1, j - 1),
                (cost_matrix[i - 1, j], i - 1, j),
                (cost_matrix[i, j - 1], i, j - 1),
            ]
            _, i, j = min(candidates, key=lambda x: x[0])

    path.append({"i": 0, "j": 0, "cost": _cosine_distance(seq_a[0], seq_b[0])})
    path.reverse()
    return path
