#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
PYTHON_BIN="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
DATA_DIR="${AIC_DATA_DIR:-${PROJECT_DIR}/data}"
MODEL_DIR="${AIC_MODEL_DIR:-${PROJECT_DIR}/clip-ViT-B-32}"
RECIPE="${1:-strength}"
STAGE="${AIC_V13_STAGE:-validate}"
GPU="${AIC_GPU:?Set AIC_GPU to your chosen physical GPU index}"
BATCH="${AIC_BATCH_SIZE:-256}"
if [[ ! "${GPU}" =~ ^[0-9]+$ || ! "${BATCH}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid GPU index or batch size" >&2; exit 2
fi
if (( 256 % BATCH != 0 )); then echo "Batch size must divide 256" >&2; exit 2; fi
ACCUM="${AIC_GRAD_ACCUM:-$((256 / BATCH))}"
if [[ ! "${ACCUM}" =~ ^[1-9][0-9]*$ ]] || (( BATCH * ACCUM != 256 )); then
  echo "Keep effective batch 256" >&2; exit 2
fi
case "${RECIPE}" in strength|dynamic|expanded) ;; *) echo "Unknown recipe" >&2; exit 2 ;; esac
case "${STAGE}" in validate|refit) ;; *) echo "Stage must be validate or refit" >&2; exit 2 ;; esac
RUN_DIR="${OUTPUT_ROOT}/v13_${RECIPE}"
if [[ "${STAGE}" == refit ]]; then RUN_DIR="${OUTPUT_ROOT}/v13_refit"; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
FREE="$(nvidia-smi --id="${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
MINIMUM="${AIC_MIN_FREE_MIB:-24576}"
if [[ ! "${MINIMUM}" =~ ^[1-9][0-9]*$ || ! "${FREE}" =~ ^[0-9]+$ ]] || (( FREE < MINIMUM )); then
  echo "GPU ${GPU}: free=${FREE} MiB, required launch threshold=${MINIMUM} MiB" >&2; exit 2
fi
args=(--recipe "${RECIPE}" --stage "${STAGE}" --train-dir "${DATA_DIR}/train"
  --data-manifest "${AIC_MANIFEST:-${OUTPUT_ROOT}/v7/dataset_manifest_v7.json}"
  --model-dir "${MODEL_DIR}" --output-dir "${RUN_DIR}" --device cuda
  --batch-size "${BATCH}" --gradient-accumulation "${ACCUM}" --workers "${AIC_WORKERS:-16}"
  --eval-batch-size "${AIC_EVAL_BATCH_SIZE:-256}" --prefetch-factor "${AIC_PREFETCH_FACTOR:-2}")
if [[ "${STAGE}" == refit ]]; then
  args+=(--selection-json "${AIC_SELECTION:-${OUTPUT_ROOT}/v13_selection.json}")
fi
if [[ -n "${AIC_FEATURE_CACHE:-}" ]]; then args+=(--feature-cache "${AIC_FEATURE_CACHE}"); fi
if [[ -f "${RUN_DIR}/resume_latest.pt" ]]; then args+=(--resume "${RUN_DIR}/resume_latest.pt"); fi
mkdir -p "${RUN_DIR}"
cd "${PROJECT_DIR}"
echo "v13_launch recipe=${RECIPE} stage=${STAGE} physical_gpu=${GPU} batch=${BATCH} accumulation=${ACCUM}" | tee -a "${RUN_DIR}/train.log"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u train_v13.py "${args[@]}" 2>&1 | tee -a "${RUN_DIR}/train.log"
