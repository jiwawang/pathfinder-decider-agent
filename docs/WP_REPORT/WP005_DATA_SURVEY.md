# WP005 — Phase 1 data survey (TCGA cross-task generalization)

Branch `experiment/tcga-generalization` (from `main` @ `4530fe3`, tag `v2.1-audited`). No classification run,
no bulk download, no core-code change. Frozen protocol: `configs/tcga_frozen.yaml`. Frozen manifests:
`metadata/tcga_lung_manifest.csv` (sha256 ed3669d420aaf785…), `metadata/tcga_kidney_manifest.csv`
(43c6d8b3233c3e67…); raw GDC responses in `metadata/tcga_gdc_raw/` (queried 2026-10-18).

## 1. What is available (GDC, open access, Diagnostic Slide, Primary Tumor, SVS)

| Project | files | patients | patients with >1 DX slide | total size |
|---|---:|---:|---:|---:|
| TCGA-LUAD (label 0) | 540 | 478 | 20 | 413 GB |
| TCGA-LUSC (label 1) | 512 | 478 | 10 | 409 GB |
| TCGA-KIRC (label 0) | 518 | 513 | 4 | 474 GB |
| TCGA-KIRP (label 1) | 299 | 275 | 11 | 308 GB |

All four projects have far more eligible patients than the 100-per-class target, so both main tasks are feasible
on patient-level 5-fold CV with ≥ 20 patients per class per fold. GDC fields actually present: project_id,
case submitter_id (patient barcode), case_id, sample_id, sample barcode, slide_id, slide barcode (DX number),
file_id, file_name, file_size, md5sum, experimental_strategy, sample_type, tissue_source_site (id + name),
section_location. `data_format` is SVS for every file.

## 2. Frozen selection rule and manifests

One slide per patient = the lowest DX number (file-name sort); eligible patients shuffled with seed 0 and the
first 100 per class kept (`scripts/tcga/build_tcga_manifest.py`). This is the same rule as
`<TCGA_ROOT>/tcga_select_download.py`, which produced the slides already on disk, so those are the first 60 of
the same ordering. Multi-DX patients in the selections: lung 7, kidney 4 (one slide each kept). Manifests are
frozen before any result and are never overwritten by the script.

| Task | rows | labels | already local | to download |
|---|---:|---|---:|---:|
| lung (LUAD 0 / LUSC 1) | 200 | 100 / 100 | 60 (all LUAD) | 140 files, **107 GB** |
| kidney (KIRC 0 / KIRP 1) | 200 | 100 / 100 | 60 (all KIRC) | 140 files, **135 GB** |

## 3. Local cache

`<TCGA_ROOT>/{LUAD,KIRC,COAD}/svs`: 60 diagnostic slides each (48 / 49 / 36 GB), md5-verified against GDC (2 of 2
checked), with TRIDENT 20x/448 px tissue coordinates (not reused: different field size and tissue rule). No
LUSC or KIRP files exist locally. Disk: `/data` has 2.6 TB free; the full plan needs ~242 GB of slides plus
~5 GB of cached embeddings. Space is not a constraint.

## 4. Download throughput — the actual constraint

Measured now from this host: a single stream 0.5 MB/s (14 MB in 30 s), six parallel range requests 0.4 MB/s
aggregate (connections timed out). At this rate 242 GB takes 6–8 days; even the 60-per-class minimum of the
missing classes (LUSC 60 ≈ 47 GB, KIRP 60 ≈ 58 GB) takes ~2.5–3 days. The earlier 157 GB cache was downloaded
with the same API, so the limit is likely transient or per-connection; it must be re-probed (and `gdc-client`
tried) before committing to a schedule.

## 5. Format / magnification adaptation needed

* OBR (the audited cohort): single-level generic-TIFF SVS with **no MPP metadata**; 0.5 µm/px comes from the
  earlier project's record `/data/obr_mpp.csv` ("20x, ~0.5 mpp, verified"). So the reference physical field is
  224 px × 0.5 µm = 112 µm per patch — recorded as an assumption, not a measurement.
* TCGA: Aperio SVS with 3–4 pyramid levels, MPP present (local LUAD: 0.252 µm × 27, 0.253 × 8, 0.248 × 4,
  0.233 "20x" × 3 …; KIRC 0.25 ± 0.003). Mostly 40x.
* Adapter (data layer only, `tcga_adapter: mpp_matched` in the frozen config): per slide read a level-0 window of
  `round(224 × 0.5 / mpp)` px (482 px at 0.2325 µm, 448 px at 0.25 µm) and resize it to 224 before the cheap
  features and before UNI, so every patch covers ~112 µm as on OBR. Tissue detection keeps the existing
  saturation rule; the grid stride equals the window. The live store and cheap-feature extractor need a
  per-slide window size (compat layer), nothing else changes. Verified on one 40x LUAD slide: 20 ms per patch
  for read + resize + cheap features.
* Expected size: ~3.2k tissue patches per TCGA slide (vs 25k on OBR); cheap features ≈ 64 s/slide (one worker);
  full-slide UNI for the cached simulation and the Full-UNI baseline ≈ 10 s GPU + ~50 s IO per slide → ≈ 7 h for
  400 slides single-process, ~3 h on two GPUs. Cached embeddings ≈ 13 MB/slide.

## 6. Site confounding (tissue source site, from the frozen manifests)

| Task | sites | shared by both classes | class-0-only | class-1-only | χ² p | Cramér's V | slides from single-class sites |
|---|---:|---:|---:|---:|---:|---:|---:|
| lung | 35 | 16 | 8 | 11 | 1e-4 | 0.61 | 26% |
| kidney | 27 | 9 | 4 | 14 | 1e-12 | 0.75 | 34% |

Kidney is strongly site-confounded (MSKCC 34 KIRC / 7 KIRP, Pittsburgh 20 / 1, MD Anderson 10 / 0, Erlangen 0 / 9);
lung less so but still significant. Classification performance on either task cannot be read as pure tumour
morphology; a site-grouped sensitivity analysis (grouping folds by TSS) must accompany the main result, and the
main experiment stays frozen as specified. File sizes per class are balanced (0.79 / 0.79 GB; 0.90 / 0.97 GB).

## 7. BRCA ER status (optional third task)

The GDC `cases` mapping has no estrogen-receptor field; ER status exists only inside the 1,183 BRCA
`Clinical Supplement` BCR XML files (`breast_carcinoma_estrogen_receptor_status`). It could be parsed from the
official XML (not guessed), but it needs an extra download/parsing step and label-completeness check. Not pursued
in this round; recorded as feasible-but-deferred.

## 8. Answers required by section 18

* Eligible WSIs: 540 / 512 / 518 / 299 (LUAD / LUSC / KIRC / KIRP); independent patients 478 / 478 / 513 / 275.
* Data to download for the frozen 100-per-class plan: 107 GB (lung) + 135 GB (kidney) = 242 GB; local cache
  covers 60 LUAD and 60 KIRC.
* Formats: Aperio SVS, mostly 40x (0.25 µm/px) with pyramids; needs the MPP-matched 448/482-px window → 224 resize.
* Both main tasks are feasible in data terms; the practical blocker is GDC throughput (0.4–0.5 MB/s measured).
* Disk: ~250 GB needed, 2.6 TB free.

## 9. Decision needed before Phase 2

1. Download plan: full 242 GB (6–8 days at the measured rate) **or** a first round with 60 per class
   (download only LUSC-60 ≈ 47 GB and KIRP-60 ≈ 58 GB, ~2.5–3 days at this rate, faster if throughput recovers).
   Either way the frozen manifests stay; a 60-per-class round would use the first 60 selection ranks of each
   class (a prefix of the frozen list), stated in advance.
2. Confirm the 112 µm MPP-matched adapter as the fixed physical scale (it rests on the OBR 0.5 µm/px record).
