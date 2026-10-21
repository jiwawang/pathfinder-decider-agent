"""WP005 Task 0 adapter tests: MPP rule, window arithmetic, cache order, patient isolation, lazy access."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.acquisition.lazy_uni_store import LazyUNIFeatureStore, UnacquiredAccessError  # noqa: E402
from src.evaluation.splits import make_folds  # noqa: E402
from src.tcga.evaluation import multiclass_metrics  # noqa: E402
from src.tcga.reader import TARGET_PX, mpp_status, read_patch, slide_mpp, window_px  # noqa: E402


class FakeSlide:
    def __init__(self, props, size=(2000, 2000)):
        self.properties = props; self.size = size

    def read_region(self, loc, level, dims):
        from PIL import Image
        return Image.new("RGBA", dims, (120, 60, 160, 255))


def test_mpp_rule_only_openslide_mpp_no_guessing():
    assert slide_mpp(FakeSlide({"openslide.mpp-x": "0.2525", "openslide.mpp-y": "0.2525", "aperio.AppMag": "20"})) == 0.2525   # AppMag ignored
    assert slide_mpp(FakeSlide({"openslide.objective-power": "40"})) is None                                                   # no MPP -> skip
    assert slide_mpp(FakeSlide({"openslide.mpp-x": "0.25", "openslide.mpp-y": "0.26"})) is None                               # x/y > 1 % -> skip
    assert slide_mpp(FakeSlide({"openslide.mpp-x": "0.25", "openslide.mpp-y": "0.2515"})) == 0.25                             # within 1 %
    assert mpp_status(None).startswith("skip") and mpp_status(0.25) == "ok" and mpp_status(1.2).startswith("warn")


def test_window_arithmetic_112um():
    assert window_px(0.5) == 224 and window_px(0.25) == 448 and window_px(0.2325) == 482 and window_px(0.23) == 487
    for mpp in (0.5, 0.25, 0.2325):
        assert abs(window_px(mpp) * mpp - TARGET_PX * 0.5) < 2 * mpp      # 112 um within one pixel


def test_read_patch_resamples_to_224():
    rgb = read_patch(FakeSlide({}), 0, 0, 482)
    assert rgb.shape == (224, 224, 3) and rgb.dtype == np.uint8 and (rgb[0, 0] == (120, 60, 160)).all()


def test_cache_rows_follow_h5_order_and_lazy_access(tmp_path):
    E = np.random.default_rng(0).normal(size=(50, 8)).astype(np.float32); np.save(tmp_path / "s.npy", E)
    store = LazyUNIFeatureStore({"s": str(tmp_path / "s.npy")}, l2_normalize=True)
    v = store.acquire("s", 7); assert np.allclose(v, E[7] / np.linalg.norm(E[7]))
    with pytest.raises(UnacquiredAccessError):
        store.embedding_of("s", 8)


def test_task0_patient_isolation_three_classes():
    meta = pd.DataFrame({"slide_id": [f"TCGA-{i:02d}-{j:04d}-01Z-00-DX1" for i, j in enumerate(range(180))],
                         "patient_id": [f"TCGA-{i:02d}-{j:04d}" for i, j in enumerate(range(180))], "label": [i % 3 for i in range(180)]})
    assert meta.patient_id.is_unique
    for tr, te in make_folds(meta, 5, 0):
        assert not set(meta.patient_id[tr]) & set(meta.patient_id[te])
        assert set(meta.label.values[te]) == {0, 1, 2}


def test_multiclass_metrics_shape():
    y = np.array([0, 1, 2, 0, 1, 2]); P = np.eye(3)[y] * 0.7 + 0.1          # rows sum to 1
    m = multiclass_metrics(y, P); assert m["auroc"] == 1.0 and m["accuracy"] == 1.0 and "auroc_class2" in m


def test_task_slide_table_matches_frozen_manifest():
    """Data check for WP006: every slide in a task table is a frozen-manifest member with the manifest's label,
    one slide per patient, and provenance columns present. Skips when the table does not exist yet."""
    import os
    root = Path(__file__).resolve().parents[1]
    for task, man in (("task1", "lung"), ("task2", "kidney")):
        table = root / "metadata" / f"tcga_{task}_slides.csv"
        if not table.exists():
            continue
        t = pd.read_csv(table); m = pd.read_csv(root / "metadata" / f"tcga_{man}_manifest.csv").set_index("file_name")
        assert t.patient_id.is_unique and t.slide_id.is_unique
        for r in t.itertuples():
            fn = r.slide_id + ".svs"
            assert fn in m.index, f"{fn} not in frozen manifest"
            assert int(m.loc[fn, "task_label"]) == int(r.label) and m.loc[fn, "patient_id"] == r.patient_id
            assert r.gdc_file_id == m.loc[fn, "gdc_file_id"] and r.md5 == m.loc[fn, "md5"]
            if hasattr(r, "wsi_path"):   # path columns are dropped in the public export
                assert os.path.exists(r.wsi_path) and os.path.exists(r.h5_path)


def test_encoder_registry_specs():
    """WP011: the hackathon encoder is H-optimus-0 (Apache-2.0, 1536-d); UNI stays available as the historical option."""
    from src.encoders.registry import ENCODERS, spec
    h, u = spec("hoptimus0"), spec("uni")
    assert h.dim == 1536 and h.hub.endswith("bioptimus/H-optimus-0") and "Apache" in h.license and h.autocast_fp16
    assert u.dim == 1024 and "NC" in u.license
    assert set(ENCODERS) >= {"hoptimus0", "uni"}
