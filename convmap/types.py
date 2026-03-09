from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Turn:
    speaker: str
    text: str


@dataclass
class Chunk:
    text: str
    index: int
    embedding: np.ndarray | None = None


@dataclass
class Conversation:
    id: str
    turns: list[Turn]
    metadata: dict = field(default_factory=dict)


@dataclass
class EmbeddedConversation:
    id: str
    chunks: list[Chunk]
    metadata: dict = field(default_factory=dict)


@dataclass
class MicroCluster:
    """A density-based micro-cluster in the embedding space."""

    centroid: np.ndarray
    weight: float
    radius: float
    created_at: float
    updated_at: float
    count: int = 0
    label: str | None = None

    def similarity(self, point: np.ndarray) -> float:
        """Cosine similarity to a point."""
        norm_a = np.linalg.norm(self.centroid)
        norm_b = np.linalg.norm(point)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(self.centroid, point) / (norm_a * norm_b))


@dataclass
class Snapshot:
    """A frozen distribution snapshot for drift detection."""

    centroids: list[np.ndarray]
    weights: list[float]
    counts: list[int]
    n_core: int
    n_potential: int
    n_outliers: int
    timestamp: float
    label: str | None = None


@dataclass
class MapState:
    """Snapshot of the current map state. Read-only for lenses."""

    core_clusters: list[MicroCluster]
    potential_clusters: list[MicroCluster]
    outlier_buffer: list[tuple[np.ndarray, dict]]
    recent_vectors: list[tuple[np.ndarray, dict]]
    snapshots: list[Snapshot]
    dimensions: int
    timestamp: float
