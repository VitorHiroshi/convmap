"""HDC lens — hyperdimensional computing signatures for noise-resilient comparison.

Encodes chunk embeddings into a single high-dimensional bipolar signature
using binding (role association) and bundling (superposition). The signature
is resilient to noise — corrupt 30% of the input and the output still
reconstructs.

Three encoding modes:
  - bundle: orderless superposition (what's in the call)
  - positional: permutation-based (sequence preserved exactly)
  - phase: bind chunks to early/mid/late role vectors (the sweet spot)
"""

from __future__ import annotations

import numpy as np

from ..types import Chunk

# Default HDC dimension. High enough for mathematical properties to hold.
DEFAULT_HDC_DIM = 10_000


def _project(embedding: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """Project a dense embedding into bipolar HDC space."""
    with np.errstate(all="ignore"):
        projected = projection @ embedding.astype(np.float64)
        result = np.sign(np.nan_to_num(projected, nan=0.0, posinf=1.0, neginf=-1.0))
    result[result == 0] = 1.0  # break ties to +1
    return result.astype(np.float32)


def _make_projection(embed_dim: int, hdc_dim: int, seed: int = 42) -> np.ndarray:
    """Create a random projection matrix (embed_dim → hdc_dim). Deterministic.

    Scaled by 1/sqrt(embed_dim) to prevent overflow in matmul.
    Uses float64 for numerical stability.
    """
    rng = np.random.RandomState(seed)
    scale = 1.0 / np.sqrt(embed_dim)
    return rng.randn(hdc_dim, embed_dim) * scale


def _bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bind two HDC vectors (element-wise multiply). Encodes association."""
    return a * b


def _bundle(vectors: list[np.ndarray]) -> np.ndarray:
    """Bundle HDC vectors (element-wise add + sign). Encodes superposition."""
    summed = np.sum(vectors, axis=0)
    return np.sign(summed).astype(np.float32)


def _permute(v: np.ndarray, shifts: int = 1) -> np.ndarray:
    """Circular permutation. Encodes position."""
    return np.roll(v, shifts)


class HDCEncoder:
    """Encodes chunk embeddings into HDC signatures."""

    def __init__(self, embed_dim: int = 1024, hdc_dim: int = DEFAULT_HDC_DIM, seed: int = 42):
        self.embed_dim = embed_dim
        self.hdc_dim = hdc_dim
        self._projection = _make_projection(embed_dim, hdc_dim, seed)

        # Phase role vectors (deterministic)
        rng = np.random.RandomState(seed + 1)
        self.EARLY = np.sign(rng.randn(hdc_dim)).astype(np.float32)
        self.MID = np.sign(rng.randn(hdc_dim)).astype(np.float32)
        self.LATE = np.sign(rng.randn(hdc_dim)).astype(np.float32)

    def signature_bundle(self, chunks: list[Chunk]) -> np.ndarray:
        """Orderless bundle — captures content, ignores sequence."""
        hdc_vecs = [self._project_chunk(c) for c in chunks if c.embedding is not None]
        if not hdc_vecs:
            return np.zeros(self.hdc_dim, dtype=np.float32)
        return _bundle(hdc_vecs)

    def signature_positional(self, chunks: list[Chunk]) -> np.ndarray:
        """Positional encoding — each chunk permuted by its index."""
        hdc_vecs = []
        for c in chunks:
            if c.embedding is None:
                continue
            projected = self._project_chunk(c)
            hdc_vecs.append(_permute(projected, shifts=c.index))
        if not hdc_vecs:
            return np.zeros(self.hdc_dim, dtype=np.float32)
        return _bundle(hdc_vecs)

    def signature_phase(self, chunks: list[Chunk]) -> np.ndarray:
        """Phase-aware encoding — binds chunks to early/mid/late role vectors."""
        valid = [c for c in chunks if c.embedding is not None]
        if not valid:
            return np.zeros(self.hdc_dim, dtype=np.float32)

        n = len(valid)
        hdc_vecs = []
        for i, c in enumerate(valid):
            projected = self._project_chunk(c)
            phase = self._get_phase(i, n)
            hdc_vecs.append(_bind(phase, projected))

        return _bundle(hdc_vecs)

    def query(self, signature: np.ndarray, role: np.ndarray, concept: np.ndarray) -> float:
        """Probe a signature: unbind the role, compare to concept.

        Example: query(sig, EARLY, embed("frustrated")) → how frustrated was the early part?
        """
        unbound = _bind(role, signature)  # unbind by re-binding (self-inverse for bipolar)
        return float(np.dot(unbound, concept) / (self.hdc_dim + 1e-8))

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two HDC signatures."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _project_chunk(self, chunk: Chunk) -> np.ndarray:
        """Project a chunk's embedding into HDC space."""
        return _project(chunk.embedding, self._projection)

    def _get_phase(self, index: int, total: int) -> np.ndarray:
        """Map a chunk index to its phase role vector."""
        if total <= 1:
            return self.EARLY
        position = index / (total - 1)
        if position < 0.33:
            return self.EARLY
        elif position < 0.67:
            return self.MID
        else:
            return self.LATE
