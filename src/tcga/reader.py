"""TCGA slide reading rule (WP005 plan section 8; same as D001 / wsi_patch_dataset.slide_mpp).

* MPP only: `openslide.mpp-x` / `openslide.mpp-y`; missing or |x - y| > 1 % -> None (the slide is skipped, never guessed).
  Magnification fields are not consulted.
* Level-0 window = round(TARGET_PX * TARGET_MPP / mpp), resampled with PIL LANCZOS to TARGET_PX, so every patch
  covers TARGET_PX * TARGET_MPP = 112 um whatever the scanner resolution.
One code path feeds the cheap features, the UNI cache and the live UNI store.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

TARGET_PX = 224      # UNI input / OBR grid
TARGET_MPP = 0.5     # um per pixel at the reference scale (OBR record)
MPP_WARN_RANGE = (0.2, 0.6)


def slide_mpp(slide) -> float | None:
    p = slide.properties
    x, y = p.get("openslide.mpp-x"), p.get("openslide.mpp-y")
    if x is None or y is None:
        return None
    try:
        x, y = float(x), float(y)
    except ValueError:
        return None
    if x <= 0 or y <= 0 or abs(x - y) / x > 0.01:
        return None
    return x


def window_px(mpp: float, target_px: int = TARGET_PX, target_mpp: float = TARGET_MPP) -> int:
    return int(round(target_px * target_mpp / mpp))


def read_patch(slide, x: int, y: int, window: int, target_px: int = TARGET_PX) -> np.ndarray:
    """RGB uint8 [target_px, target_px, 3]: level-0 window at (x, y), LANCZOS-resampled."""
    img = slide.read_region((int(x), int(y)), 0, (int(window), int(window))).convert("RGB")
    if window != target_px:
        img = img.resize((target_px, target_px), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def mpp_status(mpp: float | None) -> str:
    if mpp is None:
        return "skip: mpp missing or x/y inconsistent (>1%)"
    if not (MPP_WARN_RANGE[0] <= mpp <= MPP_WARN_RANGE[1]):
        return f"warn: mpp {mpp:.4f} outside {MPP_WARN_RANGE}"
    return "ok"
