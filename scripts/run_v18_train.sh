#!/usr/bin/env bash
set -eEuo pipefail
PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
PYTHON_BIN="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
DATA_DIR="${AIC_DATA_DIR:-${PROJECT_DIR}/data}"
MODEL_DIR="${AIC_MODEL_DIR:-${PROJECT_DIR}/clip-ViT-B-32}"
PIPELINE_DIR="${AIC_PIPELINE_DIR:-${OUTPUT_ROOT}/v18_pipeline}"
STAGE="${AIC_V18_STAGE:-validate}"
RECIPE="${AIC_V18_RECIPE:?Set AIC_V18_RECIPE to an active frozen recipe}"
GPU="${AIC_GPU:?Set an explicit physical GPU index}"
BATCH="${AIC_BATCH_SIZE:-256}"
case "$STAGE" in validate|refit) ;; *) echo 'Invalid V18 training stage' >&2; exit 2;; esac
case "$RECIPE" in resolution384|rank32|matched_control) ;; *) echo 'Invalid V18 recipe' >&2; exit 2;; esac
[[ "$STAGE" != refit || "$RECIPE" != matched_control ]] || { echo 'Matched control cannot refit' >&2; exit 2; }
[[ "$GPU" =~ ^[0-9]+$ && "$BATCH" =~ ^(128|256)$ ]] || { echo 'Invalid GPU or batch (128/256)' >&2; exit 2; }
RUN_DIR="${AIC_RUN_DIR:-${OUTPUT_ROOT}/v18_${RECIPE}}"
if [[ "$STAGE" == refit ]]; then RUN_DIR="${AIC_REFIT_DIR:-${OUTPUT_ROOT}/v18_refit}"; fi
mkdir -p "$RUN_DIR" "$PIPELINE_DIR"
cd "$PROJECT_DIR"
support() { "$PYTHON_BIN" v18_pipeline_support.py "$@" --root "$OUTPUT_ROOT"; }
status() { support state --stage "$STAGE" --status "$1" --state-file "${RECIPE}_${STAGE}_status.json"; }
exec 8>"${RUN_DIR}/.train.lock"
flock -n 8 || { echo 'This V18 training run is already active' >&2; exit 2; }
trap 'status failed; echo "V18 training failed: $RECIPE/$STAGE" >&2' ERR
trap 'status interrupted; exit 130' INT TERM
trap 'rc=$?; if ((rc != 0 && rc != 130)); then status failed; fi' EXIT
export AIC_PIPELINE_STARTED_AT="${AIC_PIPELINE_STARTED_AT:-$(support init)}"
export AIC_RESOURCE_JSON="${AIC_RESOURCE_JSON:-${PIPELINE_DIR}/resource_plan.json}"
settings="$(support resource-settings --recipe "$RECIPE" --resource-json "$AIC_RESOURCE_JSON")"
read -r frozen_workers frozen_batch <<< "$settings"
[[ "$BATCH" == "$frozen_batch" && "${AIC_WORKERS:-8}" == "$frozen_workers" ]] || {
  echo 'Training settings differ from frozen resource branch' >&2; exit 2;
}
if [[ "$STAGE" == validate ]]; then
  [[ "$RUN_DIR" == "$(support resource-field --recipe "$RECIPE" --field run_dir)" ]] || {
    echo 'Validation directory differs from frozen resource branch' >&2; exit 2;
  }
fi
official="$(support resource-field --recipe "$RECIPE" --field official_check)"
support official-check --report "$official" \
  --recipe "$RECIPE" --model-dir "$MODEL_DIR"
status waiting_gpu
exec 5>"${PIPELINE_DIR}/gpu_${GPU}.lock"
flock -w "${AIC_GPU_TIMEOUT:-172800}" 5
support wait-gpu --gpu "$GPU" --minimum "${AIC_MIN_FREE_MIB:-32768}" \
  --timeout "${AIC_GPU_TIMEOUT:-172800}" --interval "${AIC_POLL_INTERVAL:-30}" \
  --state-file "${RECIPE}_${STAGE}_gpu.json"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AIC_PIN_MEMORY=0
args=(--recipe "$RECIPE" --stage "$STAGE" --resource-json "$AIC_RESOURCE_JSON" --train-dir "${DATA_DIR}/train"
  --data-manifest "${AIC_MANIFEST:-${OUTPUT_ROOT}/v7/dataset_manifest_v7.json}"
  --model-dir "$MODEL_DIR" --output-dir "$RUN_DIR" --device cuda
  --batch-size "$BATCH" --gradient-accumulation "$((256 / BATCH))" --workers "${AIC_WORKERS:-8}"
  --eval-batch-size "${AIC_EVAL_BATCH_SIZE:-128}" --prefetch-factor 1)
if [[ "$STAGE" == refit ]]; then
  args+=(--selection-json "${AIC_SELECTION:-${OUTPUT_ROOT}/v18_selection.json}")
else
  args+=(--feature-cache "${AIC_FEATURE_CACHE:-${OUTPUT_ROOT}/v13_expanded/cache/frozen.npy}")
fi
if [[ -f "${RUN_DIR}/resume_latest.pt" ]]; then args+=(--resume "${RUN_DIR}/resume_latest.pt"); fi
status running
echo "v18_launch recipe=$RECIPE stage=$STAGE gpu=$GPU microbatch=$BATCH effective_batch=256 schedule_epochs=24" | tee -a "${RUN_DIR}/train.log"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u train_v18.py "${args[@]}" 2>&1 | tee -a "${RUN_DIR}/train.log"
status complete
