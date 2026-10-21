"""Cheap, UNI-free per-patch observations (WP001 section 5).

The slides in this cohort are single-level SVS files (no pyramid), so there is no low-magnification
level to read. Each tissue patch is therefore read once at level 0 and immediately downsampled to
`work_size` px before any statistic is computed; the cost is dominated by JPEG tile decoding and is
orders of magnitude below a ViT-L forward pass. Nothing here imports or touches UNI features.

Features per patch (FEATURE_NAMES order):
  tissue_fraction, mean/std of R,G,B, mean/std of H,S,V, entropy, laplacian_variance, edge_density,
  normalized_x, normalized_y, local_tissue_density
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is only needed for the extraction script
    cv2 = None

FEATURE_NAMES = [
    "tissue_fraction",
    "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
    "mean_h", "mean_s", "mean_v", "std_h", "std_s", "std_v",
    "entropy", "laplacian_variance", "edge_density",
    "normalized_x", "normalized_y", "local_tissue_density",
]


def patch_features(rgb: np.ndarray, work_size: int = 56) -> np.ndarray:
    """Colour / texture statistics of one RGB uint8 patch (any size), computed at `work_size` px."""
    if rgb.shape[0] != work_size:
        rgb = cv2.resize(rgb, (work_size, work_size), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    tissue = float((hsv[..., 1] > 20).mean())                      # saturation threshold, as in tissue_mask
    rgbf, hsvf = rgb.reshape(-1, 3).astype(np.float32) / 255.0, hsv.reshape(-1, 3).astype(np.float32) / 255.0
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum() / 8.0)  # normalised to [0, 1]
    lap = float(cv2.Laplacian(gray, cv2.CV_32F).var() / 1e4)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).mean())
    return np.array([tissue, *rgbf.mean(0), *rgbf.std(0), *hsvf.mean(0), *hsvf.std(0),
                     entropy, lap, edge_density, 0.0, 0.0, 0.0], dtype=np.float32)


def add_spatial_features(feats: np.ndarray, coords: np.ndarray, level0_wh: tuple[int, int],
                         patch_size: int, radius_patches: int = 2) -> np.ndarray:
    """Fill normalized_x / normalized_y / local_tissue_density (share of grid neighbours that are tissue)."""
    w0, h0 = level0_wh
    feats = feats.copy()
    feats[:, FEATURE_NAMES.index("normalized_x")] = coords[:, 0] / max(w0 - patch_size, 1)
    feats[:, FEATURE_NAMES.index("normalized_y")] = coords[:, 1] / max(h0 - patch_size, 1)
    gx, gy = coords[:, 0] // patch_size, coords[:, 1] // patch_size
    occ = set(zip(gx.tolist(), gy.tolist()))
    r = radius_patches
    n_cells = (2 * r + 1) ** 2 - 1
    dens = np.array([sum((x + dx, y + dy) in occ for dx in range(-r, r + 1) for dy in range(-r, r + 1)
                         if (dx, dy) != (0, 0)) / n_cells for x, y in zip(gx.tolist(), gy.tolist())],
                    dtype=np.float32)
    feats[:, FEATURE_NAMES.index("local_tissue_density")] = dens
    return feats


def extract_slide(wsi_path: str, coords: np.ndarray, patch_size: int = 224, work_size: int = 56) -> np.ndarray:
    """All cheap features for one slide's patch grid. Reads each patch once; never touches UNI."""
    import openslide
    s = openslide.OpenSlide(wsi_path)
    try:
        w0, h0 = s.level_dimensions[0]
        rows = []
        for x, y in coords:
            rgb = np.asarray(s.read_region((int(x), int(y)), 0, (patch_size, patch_size)).convert("RGB"))
            rows.append(patch_features(rgb, work_size))
    finally:
        s.close()
    feats = np.stack(rows).astype(np.float32)
    return add_spatial_features(feats, coords.astype(np.int64), (w0, h0), patch_size)
