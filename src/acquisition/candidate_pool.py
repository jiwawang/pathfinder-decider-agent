"""Cheap candidate pool (WP001 sections 6-7): KMeans on standardised CHEAP features, never on UNI.

This module deliberately has no import path to any UNI store; tests assert that building a pool
performs zero UNI calls.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sd = features.mean(0), features.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return ((features - mu) / sd).astype(np.float32), mu, sd


def build_cheap_pool(features: np.ndarray, coords: np.ndarray, feature_names: list[str], pool_size: int,
                     seed: int = 0, n_init: int = 3) -> dict:
    """Return per-candidate arrays. Candidates are real patches nearest to each cheap-feature centroid."""
    n = features.shape[0]
    k = int(min(pool_size, n))
    Z, mu, sd = standardize(features)
    km = (MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=n_init, batch_size=4096) if n > 60000
          else KMeans(n_clusters=k, random_state=seed, n_init=n_init))
    assign = km.fit_predict(Z)
    C = km.cluster_centers_.astype(np.float32)
    rows = []
    for c in range(k):
        idx = np.flatnonzero(assign == c)
        if idx.size == 0:
            continue
        d = np.linalg.norm(Z[idx] - C[c], axis=1)
        j = int(idx[np.argmin(d)])
        rows.append((j, c, int(idx.size), float(d.min()), float(d.mean())))
    rows.sort(key=lambda r: -r[2])
    pidx = np.array([r[0] for r in rows], dtype=np.int32)
    dmin = np.array([r[3] for r in rows], dtype=np.float32)
    dmean = np.array([r[4] for r in rows], dtype=np.float32)
    ti, ei = feature_names.index("tissue_fraction"), feature_names.index("entropy")
    return dict(
        patch_index=pidx,
        coord_x=coords[pidx, 0].astype(np.int32), coord_y=coords[pidx, 1].astype(np.int32),
        cluster_id=np.array([r[1] for r in rows], dtype=np.int32),
        cluster_size=np.array([r[2] for r in rows], dtype=np.int32),
        cluster_fraction=np.array([r[2] / n for r in rows], dtype=np.float32),
        representativeness=np.exp(-dmin / np.maximum(dmean, 1e-6)).astype(np.float32),  # 1 = sits on centroid
        cheap_vec=Z[pidx],                       # standardised cheap features of the candidate
        tissue_fraction=features[pidx, ti].astype(np.float32),
        entropy=features[pidx, ei].astype(np.float32),
        norm_x=features[pidx, feature_names.index("normalized_x")].astype(np.float32),
        norm_y=features[pidx, feature_names.index("normalized_y")].astype(np.float32),
        n_patches=np.int32(n), pool_size=np.int32(len(pidx)),
        cluster_assignment=assign.astype(np.int16),
        feature_mean=mu.astype(np.float32), feature_std=sd.astype(np.float32),
    )
