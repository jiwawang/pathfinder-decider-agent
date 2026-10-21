"""Weak patch voting from slide-level supervision (README section 6).

A *patch bank* holds prototype (candidate) embeddings from TRAINING slides only; every prototype
inherits its slide's label as a weak label. A candidate's vote is the label distribution of its
k nearest bank neighbours (cosine). This is weak supervision, not a true patch label.

Leakage rules enforced here (README section 7):
  * the bank is built from the training fold only (caller passes training slide ids);
  * a training slide's own prototypes are excluded when voting for that slide (leave-one-slide-out).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


@dataclass
class PatchBank:
    embeddings: np.ndarray      # [M, D] unit-norm
    labels: np.ndarray          # [M] int
    slide_ids: np.ndarray       # [M] str (owner slide, for leave-one-slide-out)
    k: int = 15

    @classmethod
    def from_candidates(cls, cands: dict[str, dict], train_slide_ids: list[str], k: int) -> "PatchBank":
        embs, labs, owners = [], [], []
        for sid in train_slide_ids:
            c = cands[sid]
            embs.append(c["embedding"])
            labs.append(np.full(len(c["embedding"]), int(c["label"]), dtype=np.int64))
            owners.append(np.full(len(c["embedding"]), sid, dtype=object))
        return cls(_unit(np.concatenate(embs, 0)), np.concatenate(labs), np.concatenate(owners), k)

    def vote(self, query: np.ndarray, exclude_slide_id: str | None = None) -> dict[str, np.ndarray]:
        """Vote statistics for query embeddings [Q, D].

        Returns vote_p1, vote_p0, vote_confidence (= |p1-0.5|*2), mean_neighbor_similarity,
        nearest_neighbor_similarity, and the neighbour slide ids (for traceability).
        """
        mask = np.ones(len(self.labels), dtype=bool)
        if exclude_slide_id is not None:
            mask &= self.slide_ids != exclude_slide_id
        if (self.slide_ids[mask] == exclude_slide_id).any():
            raise RuntimeError("leave-one-slide-out exclusion failed")
        E, L = self.embeddings[mask], self.labels[mask]
        k = int(min(self.k, len(L)))
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute").fit(E)
        dist, idx = nn.kneighbors(_unit(query.astype(np.float32)))
        sim = 1.0 - dist
        nl = L[idx]                                   # [Q, k]
        p1 = nl.mean(axis=1).astype(np.float32)
        return dict(
            vote_p1=p1,
            vote_p0=(1.0 - p1).astype(np.float32),
            vote_confidence=(np.abs(p1 - 0.5) * 2).astype(np.float32),
            mean_neighbor_similarity=sim.mean(axis=1).astype(np.float32),
            nearest_neighbor_similarity=sim[:, 0].astype(np.float32),
            neighbor_slide_ids=self.slide_ids[mask][idx],
        )
