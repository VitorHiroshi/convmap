from __future__ import annotations

import time

import numpy as np

from .types import MicroCluster, MapState, EmbeddedConversation


class Engine:
    """DenStream-inspired streaming map engine.

    Maintains three tiers of density:
      - Core micro-clusters: established patterns with enough weight.
      - Potential micro-clusters: emerging patterns gaining weight.
      - Outlier buffer: points that don't fit anywhere yet.

    Points that accumulate in the outlier buffer form potential clusters.
    Potential clusters that gain enough weight get promoted to core.
    Core clusters that decay below threshold get demoted.

    All operations use cosine similarity on normalized vectors.
    """

    def __init__(
        self,
        dimensions: int = 1024,
        epsilon: float = 0.3,
        mu: float = 5.0,
        beta: float = 0.3,
        decay: float = 0.001,
        max_recent: int = 10_000,
        maintenance_interval: int = 1000,
    ):
        """
        Args:
            dimensions: embedding vector size.
            epsilon: similarity threshold for merging (1 - epsilon = min cosine sim).
            mu: minimum weight to be a core cluster.
            beta: fraction of mu — minimum weight for a potential cluster to survive.
            decay: weight decay rate per maintenance cycle.
            max_recent: bounded buffer size for recent vectors.
            maintenance_interval: run maintenance every N ingestions.
        """
        self.dimensions = dimensions
        self.epsilon = epsilon
        self.mu = mu
        self.beta = beta
        self.decay = decay
        self.max_recent = max_recent
        self.maintenance_interval = maintenance_interval

        self.core_clusters: list[MicroCluster] = []
        self.potential_clusters: list[MicroCluster] = []
        self.outlier_buffer: list[tuple[np.ndarray, dict]] = []
        self.recent_vectors: list[tuple[np.ndarray, dict]] = []
        self._step = 0

    def ingest(self, conversation: EmbeddedConversation) -> None:
        """Ingest an embedded conversation into the map."""
        embeddings = [c.embedding for c in conversation.chunks if c.embedding is not None]
        if not embeddings:
            return

        # Mean pool chunks into a single conversation vector
        conv_vector = np.mean(embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(conv_vector)
        if norm < 1e-8:
            return
        conv_vector = conv_vector / norm

        meta = {
            "id": conversation.id,
            "n_chunks": len(embeddings),
            **conversation.metadata,
        }

        self._ingest_vector(conv_vector, meta)
        self._step += 1

        if self._step % self.maintenance_interval == 0:
            self._maintain()

    def ingest_vector(self, vector: np.ndarray, metadata: dict | None = None) -> None:
        """Ingest a raw pre-computed vector directly."""
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            return
        normalized = (vector / norm).astype(np.float32)
        self._ingest_vector(normalized, metadata or {})
        self._step += 1

        if self._step % self.maintenance_interval == 0:
            self._maintain()

    def _ingest_vector(self, vector: np.ndarray, metadata: dict) -> None:
        """Core ingestion: try core → potential → outlier buffer."""
        now = time.time()

        if self._try_merge(vector, self.core_clusters, now):
            pass
        elif self._try_merge(vector, self.potential_clusters, now):
            pass
        else:
            self.outlier_buffer.append((vector, metadata))

        # Bounded recent buffer
        self.recent_vectors.append((vector, metadata))
        if len(self.recent_vectors) > self.max_recent:
            self.recent_vectors.pop(0)

    def _try_merge(
        self, vector: np.ndarray, clusters: list[MicroCluster], now: float
    ) -> bool:
        """Try to merge a vector into the nearest cluster within threshold."""
        if not clusters:
            return False

        similarities = np.array([mc.similarity(vector) for mc in clusters])
        best_idx = int(np.argmax(similarities))
        min_similarity = 1 - self.epsilon

        if similarities[best_idx] < min_similarity:
            return False

        mc = clusters[best_idx]
        mc.count += 1
        mc.weight += 1
        lr = 1.0 / mc.count
        mc.centroid = mc.centroid * (1 - lr) + vector * lr
        # Re-normalize
        mc.centroid = mc.centroid / (np.linalg.norm(mc.centroid) + 1e-8)
        mc.updated_at = now
        return True

    def _maintain(self) -> None:
        """Periodic maintenance: decay, promote, demote, cluster outliers."""
        now = time.time()

        # Decay all weights
        for mc in self.core_clusters + self.potential_clusters:
            mc.weight *= 1 - self.decay

        # Demote weak core clusters
        still_core = []
        for mc in self.core_clusters:
            if mc.weight >= self.mu:
                still_core.append(mc)
            else:
                self.potential_clusters.append(mc)
        self.core_clusters = still_core

        # Promote strong potential clusters
        still_potential = []
        for mc in self.potential_clusters:
            if mc.weight >= self.mu:
                self.core_clusters.append(mc)
            elif mc.weight >= self.beta * self.mu:
                still_potential.append(mc)
            # else: weight too low, cluster dies
        self.potential_clusters = still_potential

        # Try to form new potential clusters from outliers
        self._process_outliers(now)

    def _process_outliers(self, now: float) -> None:
        """Form potential clusters from accumulated outliers."""
        if not self.outlier_buffer:
            return

        for vector, _meta in self.outlier_buffer:
            merged = self._try_merge(vector, self.potential_clusters, now)
            if not merged:
                self.potential_clusters.append(
                    MicroCluster(
                        centroid=vector.copy(),
                        weight=1.0,
                        radius=self.epsilon,
                        created_at=now,
                        updated_at=now,
                        count=1,
                    )
                )

        self.outlier_buffer = []

    @property
    def state(self) -> MapState:
        """Current map state snapshot for lenses."""
        return MapState(
            core_clusters=list(self.core_clusters),
            potential_clusters=list(self.potential_clusters),
            outlier_buffer=list(self.outlier_buffer),
            recent_vectors=list(self.recent_vectors),
            dimensions=self.dimensions,
            timestamp=time.time(),
        )

    @property
    def summary(self) -> dict:
        """Quick summary of the map state."""
        return {
            "core_clusters": len(self.core_clusters),
            "potential_clusters": len(self.potential_clusters),
            "outlier_buffer": len(self.outlier_buffer),
            "recent_vectors": len(self.recent_vectors),
            "total_ingested": self._step,
        }
