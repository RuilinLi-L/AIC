#!/usr/bin/env bash
set -eEuo pipefail
PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
PYTHON_BIN="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
DATA_DIR="${AIC_DATA_DIR:-${PROJECT_DIR}/data}"
MODEL_DIR="${AIC_MODEL_DIR:-${PROJECT_DIR}/clip-ViT-B-32}"
PIPELINE_DIR="${AIC_PIPELINE_DIR:-${OUTPUT_ROOT}/v17_pipeline}"
STAGE="${AIC_V17_STAGE:-validate}"
RECIPE="${AIC_V17_RECIPE:?Set AIC_V17_RECIPE to agreement_recovery or dynamic_prototype}"
GPU="${AIC_GPU:?Set an explicit physical GPU index}"
BATCH="${AIC_BATCH_SIZE:-256}"
case "$STAGE" in validate|refit) ;; *) echo 'Invalid V17 training stage' >&2; exit 2;; esac
case "$RECIPE" in agreement_recovery|dynamic_prototype) ;; *) echo 'Invalid V17 recipe' >&2; exit 2;; esac
[[ "$GPU" =~ ^[0-9]+$ && "$BATCH" =~ ^(128|256)$ ]] || { echo 'Invalid GPU or batch (128/256)' >&2; exit 2; }
RUN_DIR="${AIC_RUN_DIR:-${OUTPUT_ROOT}/v17_${RECIPE}}"
if [[ "$STAGE" == refit ]]; then RUN_DIR="${AIC_REFIT_DIR:-${OUTPUT_ROOT}/v17_refit}"; fi
mkdir -p "$RUN_DIR" "$PIPELINE_DIR"
cd "$PROJECT_DIR"
support() { "$PYTHON_BIN" v17_pipeline_support.py "$@" --root "$OUTPUT_ROOT"; }
status() { support state --stage "$STAGE" --status "$1" --state-file "${RECIPE}_${STAGE}_status.json"; }
exec 8>"${RUN_DIR}/.train.lock"
flock -n 8 || { echo 'This V17 training run is already active' >&2; exit 2; }
trap 'status failed; echo "V17 training failed: $RECIPE/$STAGE" >&2' ERR
trap 'status interrupted; exit 130' INT TERM
trap 'rc=$?; if ((rc != 0 && rc != 130)); then status failed; fi' EXIT
export AIC_PIPELINE_STARTED_AT="${AIC_PIPELINE_STARTED_AT:-$(support init)}"
support official-check --report "${AIC_OFFICIAL_CHECK:-${RUN_DIR}/official_check.json}" \
  --recipe "$RECIPE" --model-dir "$MODEL_DIR"
status waiting_gpu
exec 5>"${PIPELINE_DIR}/gpu_${GPU}.lock"
flock -w "${AIC_GPU_TIMEOUT:-172800}" 5
support wait-gpu --gpu "$GPU" --minimum "${AIC_MIN_FREE_MIB:-32768}" \
  --timeout "${AIC_GPU_TIMEOUT:-172800}" --interval "${AIC_POLL_INTERVAL:-30}" \
  --state-file "${RECIPE}_${STAGE}_gpu.json"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AIC_PIN_MEMORY=0
args=(--recipe "$RECIPE" --stage "$STAGE" --train-dir "${DATA_DIR}/train"
  --data-manifest "${AIC_MANIFEST:-${OUTPUT_ROOT}/v7/dataset_manifest_v7.json}"
  --model-dir "$MODEL_DIR" --output-dir "$RUN_DIR" --device cuda
  --batch-size "$BATCH" --gradient-accumulation "$((256 / BATCH))" --workers "${AIC_WORKERS:-8}"
  --eval-batch-size "${AIC_EVAL_BATCH_SIZE:-128}" --prefetch-factor 1)
if [[ "$STAGE" == refit ]]; then
  args+=(--selection-json "${AIC_SELECTION:-${OUTPUT_ROOT}/v17_selection.json}")
else
  args+=(--feature-cache "${AIC_FEATURE_CACHE:-${OUTPUT_ROOT}/v13_expanded/cache/frozen.npy}")
fi
if [[ -f "${RUN_DIR}/resume_latest.pt" ]]; then args+=(--resume "${RUN_DIR}/resume_latest.pt"); fi
if [[ "$BATCH" == 128 ]]; then
  [[ -n "${AIC_BENCHMARK_REPORT:-}" ]] || { echo '128 requires a recipe benchmark recording actual 256 OOM' >&2; exit 2; }
  settings="$(support benchmark-settings --benchmark "$AIC_BENCHMARK_REPORT" --recipe "$RECIPE")"
  [[ "$settings" == '8 128' ]] || { echo 'Benchmark does not authorize 128x2' >&2; exit 2; }
fi
status running
echo "v17_launch recipe=$RECIPE stage=$STAGE gpu=$GPU microbatch=$BATCH effective_batch=256 schedule_epochs=24" | tee -a "${RUN_DIR}/train.log"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u train_v17.py "${args[@]}" 2>&1 | tee -a "${RUN_DIR}/train.log"
status complete
