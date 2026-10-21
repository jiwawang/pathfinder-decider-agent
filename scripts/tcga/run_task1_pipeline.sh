#!/usr/bin/env bash
# WP006 Task 1 end-to-end after the download: TRIDENT (both GPUs) -> slide table -> cheap features -> UNI cache
# (both GPUs, from both ends) -> frozen benchmark -> live benchmark. Sequencing by PID (never pgrep by name).
#   bash scripts/tcga/run_task1_pipeline.sh configs/tcga_task1.yaml LUAD LUSC
set -uo pipefail
CFG="${1:-configs/tcga_task1.yaml}"; A="${2:-LUAD}"; B="${3:-LUSC}"
PY=python; L=results/_logs
until grep -q "\[download\] done" $L/download_lung_full.log; do sleep 60; done
echo "[pipeline] download finished $(date)"
bash scripts/tcga/run_trident_settings2.sh $B 0 > $L/trident_${B}_full.log 2>&1 & P0=$!
bash scripts/tcga/run_trident_settings2.sh $A 1 > $L/trident_${A}_full.log 2>&1 & P1=$!
wait $P0 $P1; echo "[pipeline] trident done $(date)"
$PY scripts/tcga/build_task0_slide_table.py --config $CFG 2>&1 | grep -E "^\[task" 
OMP_NUM_THREADS=2 $PY scripts/tcga/extract_cheap_features_tcga.py --config $CFG --workers 12 > $L/cheap_task1.log 2>&1
echo "[pipeline] cheap done $(date): $(grep -c 'cheap-tcga\] TCGA' $L/cheap_task1.log) slides"
CUDA_VISIBLE_DEVICES=0 $PY scripts/tcga/cache_uni_embeddings.py --config $CFG --workers 8 > $L/uni_task1_gpu0.log 2>&1 & P0=$!
CUDA_VISIBLE_DEVICES=1 $PY scripts/tcga/cache_uni_embeddings.py --config $CFG --workers 8 --reverse > $L/uni_task1_gpu1.log 2>&1 & P1=$!
wait $P0 $P1; echo "[pipeline] uni cache done $(date)"
$PY scripts/tcga/build_task0_slide_table.py --config $CFG 2>&1 | grep -E "^\[task"
$PY scripts/tcga/write_layer_readme.py > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=1 $PY scripts/tcga/run_task0_cv.py --config $CFG > $L/task1_full.log 2>&1
echo "[pipeline] benchmark done $(date)"
CUDA_VISIBLE_DEVICES=0 $PY scripts/tcga/bench_task0_live.py --config $CFG --n-per-class 2 > $L/task1_live_bench.log 2>&1
echo "[pipeline] all done $(date)"
