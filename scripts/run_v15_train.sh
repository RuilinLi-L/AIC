#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
PYTHON_BIN="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
DATA_DIR="${AIC_DATA_DIR:-${PROJECT_DIR}/data}"
MODEL_DIR="${AIC_MODEL_DIR:-${PROJECT_DIR}/clip-ViT-B-32}"
STAGE="${AIC_V15_STAGE:-validate}"
GPU="${AIC_GPU:?Set AIC_GPU to a physical GPU index}"
BATCH="${AIC_BATCH_SIZE:-256}"
case "$STAGE" in validate|refit) ;; *) echo "Invalid V15 training stage" >&2; exit 2;; esac
if [[ ! "$GPU" =~ ^[0-9]+$ || ! "$BATCH" =~ ^(128|256)$ ]]; then
  echo "GPU must be nonnegative and microbatch must be 256 or OOM-approved 128" >&2; exit 2
fi
ACCUM="$((256 / BATCH))"
RUN_DIR="${OUTPUT_ROOT}/v15_expanded_mlp"
if [[ "$STAGE" == refit ]]; then RUN_DIR="${OUTPUT_ROOT}/v15_refit"; fi
mkdir -p "$RUN_DIR"
exec 8>"${RUN_DIR}/.train.lock"
flock -n 8 || { echo "This V15 run is already active" >&2; exit 2; }
FREE="$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
MINIMUM="${AIC_MIN_FREE_MIB:-32768}"
if [[ ! "$FREE" =~ ^[0-9]+$ || ! "$MINIMUM" =~ ^[1-9][0-9]*$ ]] || ((FREE < MINIMUM)); then
  echo "GPU $GPU free=$FREE MiB; required=$MINIMUM MiB" >&2; exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export AIC_PIN_MEMORY="${AIC_PIN_MEMORY:-0}"
args=(--recipe expanded_mlp --stage "$STAGE" --train-dir "${DATA_DIR}/train"
  --data-manifest "${AIC_MANIFEST:-${OUTPUT_ROOT}/v7/dataset_manifest_v7.json}"
  --model-dir "$MODEL_DIR" --output-dir "$RUN_DIR" --device cuda
  --batch-size "$BATCH" --gradient-accumulation "$ACCUM" --workers "${AIC_WORKERS:-8}"
  --eval-batch-size "${AIC_EVAL_BATCH_SIZE:-128}" --prefetch-factor 1)
if [[ "$STAGE" == refit ]]; then
  args+=(--selection-json "${AIC_SELECTION:-${OUTPUT_ROOT}/v15_selection.json}")
else
  args+=(--feature-cache "${AIC_FEATURE_CACHE:-${OUTPUT_ROOT}/v13_expanded/cache/frozen.npy}")
fi
if [[ -f "${RUN_DIR}/resume_latest.pt" ]]; then args+=(--resume "${RUN_DIR}/resume_latest.pt"); fi
cd "$PROJECT_DIR"
echo "v15_launch stage=$STAGE gpu=$GPU batch=$BATCH accumulation=$ACCUM workers=${AIC_WORKERS:-8} prefetch=1 pin_memory=$AIC_PIN_MEMORY" | tee -a "${RUN_DIR}/train.log"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u train_v15.py "${args[@]}" 2>&1 | tee -a "${RUN_DIR}/train.log"
