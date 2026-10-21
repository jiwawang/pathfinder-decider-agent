"""Per-round shortlist from the cheap candidate pool (WP001 sections 8-9).

Four deterministic exploration strategies each nominate one not-yet-acquired candidate:
  A  largest unexplored cheap cluster (representative of a big region)
  B  highest cheap-feature novelty versus the acquired evidence
  C  spatially farthest from the acquired evidence
  D  under-covered neighbourhood: fewest acquired candidates among its cheap-space neighbours
The heuristic only proposes; the router decides. Every option records why it was shortlisted.
"""
from __future__ import annotations

import numpy as np

from decision_patterns.adaptive_expensive_observation_router import Option

STRATEGIES = {"A": "largest unexplored tissue cluster", "B": "highest cheap-feature novelty",
              "C": "spatially farthest from acquired evidence", "D": "under-covered neighbourhood"}


class ShortlistBuilder:
    def __init__(self, pool: dict, k_neighbors: int = 8):
        self.pool = pool
        self.n = int(pool["pool_size"])
        self.Z = pool["cheap_vec"].astype(np.float32)
        self.xy = np.stack([pool["norm_x"], pool["norm_y"]], 1).astype(np.float32)
        d = np.linalg.norm(self.Z[:, None, :] - self.Z[None, :, :], axis=-1)
        self._cheap_scale = float(np.median(d[np.triu_indices(self.n, 1)])) if self.n > 1 else 1.0
        self._nbrs = np.argsort(d, axis=1)[:, 1:k_neighbors + 1]

    def stats(self, i: int, acquired: list[int]) -> dict[str, float]:
        p = self.pool
        if acquired:
            nov = float(np.linalg.norm(self.Z[acquired] - self.Z[i], axis=1).min() / self._cheap_scale)
            dist = float(np.linalg.norm(self.xy[acquired] - self.xy[i], axis=1).min())
        else:
            nov, dist = 1.0, 1.0
        return dict(cluster_coverage_percent=round(100 * float(p["cluster_fraction"][i]), 1),
                    cheap_novelty=round(min(nov, 1.0), 2), distance_from_nearest_acquired=round(min(dist, 1.0), 2),
                    tissue_fraction=round(float(p["tissue_fraction"][i]), 2), entropy=round(float(p["entropy"][i]), 2),
                    representativeness=round(float(p["representativeness"][i]), 2))

    def exploration_score(self, i: int, acquired: list[int]) -> float:
        s = self.stats(i, acquired)
        p = self.pool
        return float(s["cheap_novelty"] + s["distance_from_nearest_acquired"]
                     + p["cluster_fraction"][i] / p["cluster_fraction"].max()
                     + s["tissue_fraction"] * s["entropy"])

    def build(self, acquired: list[int], k: int = 4, min_tissue: float = 0.25) -> list[Option]:
        p = self.pool
        remaining = [i for i in range(self.n) if i not in set(acquired)]
        # Prefer real tissue: cheap novelty is otherwise maximised by background / pen-mark patches.
        tissue_rich = [i for i in remaining if p["tissue_fraction"][i] >= min_tissue]
        if tissue_rich:
            remaining = tissue_rich
        if not remaining:
            return []
        rem = np.array(remaining)
        acq = np.array(acquired, dtype=int)
        rank = {}
        rank["A"] = rem[np.argsort(-p["cluster_fraction"][rem], kind="stable")]
        if len(acq):
            nov = np.linalg.norm(self.Z[rem][:, None, :] - self.Z[acq][None, :, :], axis=-1).min(1)
            dist = np.linalg.norm(self.xy[rem][:, None, :] - self.xy[acq][None, :, :], axis=-1).min(1)
            cover = np.isin(self._nbrs[rem], acq).mean(1)
            rank["B"] = rem[np.argsort(-nov, kind="stable")]
            rank["C"] = rem[np.argsort(-dist, kind="stable")]
            rank["D"] = rem[np.lexsort((-(p["tissue_fraction"][rem] * p["entropy"][rem]), cover))]
        else:  # nothing acquired yet: novelty/distance are undefined, fall back to content + size
            rank["B"] = rem[np.argsort(-(p["tissue_fraction"][rem] * p["entropy"][rem]), kind="stable")]
            rank["C"] = rem[np.argsort(-np.abs(self.xy[rem] - self.xy[rem].mean(0)).sum(1), kind="stable")]
            rank["D"] = rem[np.argsort(-p["representativeness"][rem], kind="stable")]
        chosen: list[int] = []
        options: list[Option] = []
        for letter in list(STRATEGIES)[:k]:
            pick = next((int(i) for i in rank[letter] if int(i) not in chosen), None)
            if pick is None:
                continue
            chosen.append(pick)
            s = self.stats(pick, acquired)
            desc = (f"{STRATEGIES[letter]}; coverage {s['cluster_coverage_percent']}%, novelty {s['cheap_novelty']}, "
                    f"distance {s['distance_from_nearest_acquired']}, tissue {s['tissue_fraction']}, entropy {s['entropy']}")
            options.append(Option(letter, desc, payload=pick, score=self.exploration_score(pick, acquired)))
        return options
