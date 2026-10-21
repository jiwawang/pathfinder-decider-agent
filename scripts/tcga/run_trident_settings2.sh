#!/usr/bin/env bash
# WP005 Task 0: TRIDENT segmentation + 224-px coords into a NEW job_dir per cancer type (D001 convention).
#   bash scripts/tcga/run_trident_settings2.sh COAD 0      # project, GPU id
# seg: --segmenter hest (no pen-mark/artifact/hole removal; same as settings_1)
# coords: --mag 20 --patch_size 224 --overlap 0 --min_tissue_proportion 0.5 (threshold in the dir name on purpose)
set -euo pipefail
P="${1:?project (COAD|KIRC|LUAD)}"; G="${2:-0}"
E=~/miniconda3/envs/trident; R="$E/bin/python $E/src/TRIDENT/run_batch_of_slides.py"
W=<TCGA_ROOT>/$P/svs; J=<TCGA_ROOT>/$P/Patch_Trident_settings_2_hest
mkdir -p "$J"
$R --task seg    --wsi_dir "$W" --job_dir "$J" --gpus "$G" --segmenter hest --skip_errors
$R --task coords --wsi_dir "$W" --job_dir "$J" --mag 20 --patch_size 224 --overlap 0 \
   --min_tissue_proportion 0.5 --coords_dir 20x_224px_0px_overlap_tissue0.5 --skip_errors
echo "[trident] $P done"
