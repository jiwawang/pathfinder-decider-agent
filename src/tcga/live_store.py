"""Live UNI store for TCGA: same as LiveUNIFeatureStore but with the MPP-matched per-slide window."""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.acquisition.lazy_uni_store import LiveUNIFeatureStore
from src.tcga.reader import read_patch


class TCGALiveUNIFeatureStore(LiveUNIFeatureStore):
    def __init__(self, wsi_paths, coords, windows: dict[str, int], device: str = "cuda", l2_normalize: bool = True, encoder: str = "uni"):
        super().__init__(wsi_paths, coords, patch_size=224, device=device, l2_normalize=l2_normalize, encoder=encoder)
        self._windows = dict(windows)

    def crop(self, slide_id: str, patch_index: int):
        if slide_id not in self._slides:
            self._slides[slide_id] = self._openslide.OpenSlide(self._wsi_paths[slide_id])
        x, y = self._coords[slide_id][patch_index]
        return Image.fromarray(read_patch(self._slides[slide_id], int(x), int(y), self._windows[slide_id]))
