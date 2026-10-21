"""Lazy / live UNI feature stores (WP001 sections 2, 3, 22).

The acquisition policy never sees a UNI embedding before it decides to ACQUIRE it. Both stores expose
exactly one way to obtain an embedding, `acquire(slide_id, patch_index)`, which returns a single row
and counts one UNI call. There is no bulk accessor on purpose; the cached arrays are private.

  LazyUNIFeatureStore  cached simulation: memory-maps the existing feature .npy files and copies out
                       one row per call (the 25k-row arrays are never materialised for the policy).
  LiveUNIFeatureStore  demo / verification: crops the patch from the WSI with OpenSlide and runs the
                       UNI encoder on it, so unselected patches are never encoded at all.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


class UnacquiredAccessError(RuntimeError):
    """Raised when code tries to read an embedding that was never acquired."""


@dataclass
class AcquisitionBudget:
    min_calls: int
    target_calls: int
    max_calls: int
    uni_calls: int = 0
    acquired_patch_ids: list[int] = field(default_factory=list)

    def record(self, patch_index: int) -> None:
        if self.uni_calls >= self.max_calls:
            raise RuntimeError("UNI budget exhausted")
        self.uni_calls += 1
        self.acquired_patch_ids.append(int(patch_index))

    @property
    def can_stop(self) -> bool:
        return self.uni_calls >= self.min_calls

    @property
    def must_stop(self) -> bool:
        return self.uni_calls >= self.max_calls

    @property
    def target_reached(self) -> bool:
        return self.uni_calls >= self.target_calls


class _StoreBase:
    def __init__(self) -> None:
        self.calls: dict[str, int] = defaultdict(int)
        self.acquired: dict[str, list[int]] = defaultdict(list)
        self.latency_ms: list[float] = []

    @property
    def total_calls(self) -> int:
        return int(sum(self.calls.values()))

    def embedding_of(self, slide_id: str, patch_index: int) -> np.ndarray:
        """Re-read an embedding that WAS acquired (e.g. for pooling); raises otherwise."""
        if int(patch_index) not in self.acquired[slide_id]:
            raise UnacquiredAccessError(f"{slide_id}: patch {patch_index} was never acquired")
        return self._row(slide_id, int(patch_index))

    def acquire(self, slide_id: str, patch_index: int) -> np.ndarray:
        t0 = time.perf_counter()
        row = self._row(slide_id, int(patch_index))
        self.latency_ms.append((time.perf_counter() - t0) * 1000.0)
        self.calls[slide_id] += 1
        self.acquired[slide_id].append(int(patch_index))
        return row

    def _row(self, slide_id: str, patch_index: int) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


class LazyUNIFeatureStore(_StoreBase):
    """Cached simulation over the existing per-slide feature .npy files (row order == h5 coords)."""

    def __init__(self, npy_paths: dict[str, str], l2_normalize: bool = True):
        super().__init__()
        self._paths = dict(npy_paths)
        self._arrays: dict[str, np.memmap] = {}
        self._l2 = l2_normalize

    def _row(self, slide_id: str, patch_index: int) -> np.ndarray:
        if slide_id not in self._arrays:
            self._arrays[slide_id] = np.load(self._paths[slide_id], mmap_mode="r")
        v = np.array(self._arrays[slide_id][patch_index], dtype=np.float32)  # one row, copied
        if self._l2:
            v /= max(float(np.linalg.norm(v)), 1e-8)
        return v

    def n_patches(self, slide_id: str) -> int:
        if slide_id not in self._arrays:
            self._arrays[slide_id] = np.load(self._paths[slide_id], mmap_mode="r")
        return int(self._arrays[slide_id].shape[0])


class LiveUNIFeatureStore(_StoreBase):
    """On-demand encoder: open the WSI, crop the patch, encode it. Nothing is precomputed.
    `encoder` selects the patch encoder from src/encoders/registry.py (hackathon default elsewhere: hoptimus0)."""

    def __init__(self, wsi_paths: dict[str, str], coords: dict[str, np.ndarray], patch_size: int = 224,
                 device: str = "cuda", l2_normalize: bool = True, encoder: str = "uni"):
        super().__init__()
        import openslide
        import torch
        from src.encoders.registry import build_encoder
        self._wsi_paths, self._coords, self._ps, self._device, self._l2 = wsi_paths, coords, patch_size, device, l2_normalize
        self._slides: dict[str, openslide.OpenSlide] = {}
        self._openslide = openslide
        self._torch = torch
        self.model, self.tf, self.spec = build_encoder(encoder, device)
        self.encoder = encoder

    def crop(self, slide_id: str, patch_index: int):
        if slide_id not in self._slides:
            self._slides[slide_id] = self._openslide.OpenSlide(self._wsi_paths[slide_id])
        x, y = self._coords[slide_id][patch_index]
        return self._slides[slide_id].read_region((int(x), int(y)), 0, (self._ps, self._ps)).convert("RGB")

    def _row(self, slide_id: str, patch_index: int) -> np.ndarray:
        from src.encoders.registry import encode
        return encode(self.model, self.spec, self.tf(self.crop(slide_id, patch_index)).unsqueeze(0), self._device, self._l2)[0]

    def n_patches(self, slide_id: str) -> int:
        return int(len(self._coords[slide_id]))
