"""Within-slide clustering -> candidate patches (README section 5).

Each slide's patch embeddings are clustered with KMeans (K configurable). For every cluster the
real patch closest to the centroid becomes the candidate; synthetic centroids are never used as
evidence. Representativeness = cosine(candidate_embedding, centroid).
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def build_candidates(embeddings: np.ndarray, coords: np.ndarray, num_clusters: int,
                     seed: int = 0, n_init: int = 3, minibatch_threshold: int = 60000) -> dict:
    """Return a dict of per-candidate arrays plus slide-level summaries.

    Keys: patch_index, coord_x, coord_y, cluster_id, cluster_size, cluster_fraction,
          distance_to_centroid, representativeness, embedding, all_patch_mean, n_patches,
          cluster_assignment (per patch, for visualisation).
    """
    n = embeddings.shape[0]
    k = int(min(num_clusters, n))
    X = embeddings.astype(np.float32)
    if n > minibatch_threshold:
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=n_init, batch_size=4096)
    else:
        km = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
    assign = km.fit_predict(X)
    centroids = km.cluster_centers_.astype(np.float32)

    rows = []
    for c in range(k):
        idx = np.flatnonzero(assign == c)
        if idx.size == 0:
            continue
        d = np.linalg.norm(X[idx] - centroids[c], axis=1)
        j = idx[int(np.argmin(d))]
        rep = float(_unit(X[j]) @ _unit(centroids[c]))
        rows.append((j, c, idx.size, float(d.min()), rep))

    rows.sort(key=lambda r: -r[2])  # largest clusters first (stable, deterministic)
    pidx = np.array([r[0] for r in rows], dtype=np.int32)
    return dict(
        patch_index=pidx,
        coord_x=coords[pidx, 0].astype(np.int32),
        coord_y=coords[pidx, 1].astype(np.int32),
        cluster_id=np.array([r[1] for r in rows], dtype=np.int32),
        cluster_size=np.array([r[2] for r in rows], dtype=np.int32),
        cluster_fraction=np.array([r[2] / n for r in rows], dtype=np.float32),
        distance_to_centroid=np.array([r[3] for r in rows], dtype=np.float32),
        representativeness=np.array([r[4] for r in rows], dtype=np.float32),
        embedding=X[pidx],
        all_patch_mean=X.mean(axis=0).astype(np.float32),
        n_patches=np.int32(n),
        num_clusters=np.int32(k),
        cluster_assignment=assign.astype(np.int16),
    )
